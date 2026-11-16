"""
MODULE 2: PYTORCH CONDITIONAL GENERATIVE MODEL FOR PROTAC LINKER INPAINTING
Implements:
1. Anchor Feature Embedding Network (projects Warhead & E3 ligase anchor vectors into latent condition space).
2. Conditional Variational Autoencoder (C-VAE) with Latent Reparameterization and Autoregressive/Fragment Decoder.
3. Diverse PROTAC Linker Grammar across lengths (6 to 22 atoms):
   - Flexible Alkyl chains
   - Polyethylene Glycol (PEG / O-alkyl) units
   - Rigid alkynyl spacers (-C#C-)
   - Conformationally restricted heterocycles (piperazine, piperidine, 1,2,3-triazole, phenyl)
4. Chemical Assembly Engine: Synthetically stitches [Warhead]--[Linker]--[E3 Recruiter] into sanitized PROTACs.
"""

import os
import math
import random
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
try:
    from rdkit.Chem import rdFingerprintGenerator
    HAS_FP_GEN = True
except ImportError:
    HAS_FP_GEN = False


# Linker vocabulary of modular chemical building blocks for TPD heterobifunctional linkers
LINKER_VOCABULARY = [
    "<SOS>", "<EOS>", "<PAD>",
    # Linear Alkyl Units
    "CC", "CCC", "CCCC", "CCCCC",
    # PEG (Polyethylene Glycol) Units
    "CCOCC", "COCCO", "CCOCCNC", "OCCOCC", "COCO",
    # Rigid / Alkynyl units
    "C#C", "CC#CC", "C#CCC#C",
    # Nitrogenous / Amide spacers
    "CNC(=O)", "C(=O)NC", "CCNC(=O)", "C(=O)NCC", "NC(=O)C",
    # Conformationally restricted scaffolds
    "N1CCNCC1",        # Piperazine
    "C1CCN(CC1)",      # Piperidine
    "c1cn(nn1)",       # 1,2,3-Triazole (CuAAC click product)
    "c1ccc(cc1)",      # Phenyl ring
    "c1ccncc1",        # Pyridine spacer
    "C1COCCN1",        # Morpholine-like
]

VOCAB_TO_IDX = {tok: idx for idx, tok in enumerate(LINKER_VOCABULARY)}
IDX_TO_VOCAB = {idx: tok for idx, tok in enumerate(LINKER_VOCABULARY)}
VOCAB_SIZE = len(LINKER_VOCABULARY)


def featurize_anchor_smiles(smiles: str, n_bits: int = 128) -> np.ndarray:
    """Generate dense Morgan topological fingerprint vector for anchor conditioning."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    if HAS_FP_GEN:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        fp = gen.GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class AnchorConditioningEncoder(nn.Module):
    """
    Encodes bi-partite anchor information:
    - Warhead anchor vector (chemical features + attachment handle)
    - E3 recruiter anchor vector (chemical features + attachment handle)
    - Target linker length condition (e.g. 6 to 22 heavy atoms)
    """

    def __init__(self, fp_dim: int = 128, length_dim: int = 1, cond_dim: int = 64):
        super().__init__()
        # Total input: 2 fingerprints + 1 normalized target length
        in_dim = (fp_dim * 2) + length_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128) if False else nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.Tanh()
        )

    def forward(self, warhead_vec: torch.Tensor, e3_vec: torch.Tensor, target_len: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([warhead_vec, e3_vec, target_len], dim=-1)
        return self.net(combined)


class LinkerDecoderRNN(nn.Module):
    """
    Conditional Recurrent Decoder that autoregressively generates linker fragment tokens
    conditioned on latent representation z and anchor context c.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 48, hidden_dim: int = 96,
                 latent_dim: int = 32, cond_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB_TO_IDX["<PAD>"])
        self.in_proj = nn.Linear(embed_dim + latent_dim + cond_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_seq: torch.Tensor, z: torch.Tensor, c: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        # x_seq: (B, T)
        embeds = self.embedding(x_seq)  # (B, T, embed_dim)
        B, T, _ = embeds.shape
        
        # Tile z and c across sequence length T
        zc = torch.cat([z, c], dim=-1).unsqueeze(1).expand(-1, T, -1)  # (B, T, latent_dim + cond_dim)
        inputs = self.in_proj(torch.cat([embeds, zc], dim=-1))
        
        output, hidden = self.gru(inputs, hidden)
        logits = self.out_head(output)
        return logits, hidden


class ConditionalLinkerVAE(nn.Module):
    """
    Full Conditional Variational Autoencoder (C-VAE) for PROTAC Linker Inpainting.
    q(z | x, c) -> N(mu, sigma^2)
    p(x | z, c) -> generated linker token sequence
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, fp_dim: int = 128,
                 latent_dim: int = 32, cond_dim: int = 64, hidden_dim: int = 96):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        # Conditioning Network
        self.anchor_encoder = AnchorConditioningEncoder(fp_dim=fp_dim, length_dim=1, cond_dim=cond_dim)

        # Recognition (Encoder) Network: q(z | x, c)
        self.enc_embedding = nn.Embedding(vocab_size, 48, padding_idx=VOCAB_TO_IDX["<PAD>"])
        self.enc_rnn = nn.GRU(48, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2 + cond_dim, latent_dim)

        # Generative (Decoder) Network: p(x | z, c)
        self.decoder = LinkerDecoderRNN(vocab_size=vocab_size, embed_dim=48, hidden_dim=hidden_dim,
                                        latent_dim=latent_dim, cond_dim=cond_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq: torch.Tensor, warhead_fp: torch.Tensor, e3_fp: torch.Tensor, target_len: torch.Tensor):
        c = self.anchor_encoder(warhead_fp, e3_fp, target_len)
        embeds = self.enc_embedding(x_seq)
        _, h_n = self.enc_rnn(embeds)
        # Concat bidirectional states: (B, hidden_dim * 2)
        h_enc = torch.cat([h_n[0], h_n[1]], dim=-1)
        h_c = torch.cat([h_enc, c], dim=-1)
        mu = self.fc_mu(h_c)
        logvar = self.fc_logvar(h_c)
        z = self.reparameterize(mu, logvar)
        logits, _ = self.decoder(x_seq, z, c)
        return logits, mu, logvar

    @torch.no_grad()
    def sample_linker(self, warhead_fp: torch.Tensor, e3_fp: torch.Tensor, target_len: float,
                      temperature: float = 0.85, max_steps: int = 8) -> List[str]:
        """Autoregressively sample a novel linker fragment token sequence."""
        self.eval()
        t_len_tensor = torch.tensor([[target_len / 22.0]], dtype=torch.float32)
        c = self.anchor_encoder(warhead_fp, e3_fp, t_len_tensor)
        z = torch.randn((1, self.latent_dim))

        curr_token = torch.tensor([[VOCAB_TO_IDX["<SOS>"]]], dtype=torch.long)
        hidden = None
        generated_tokens = []

        for _ in range(max_steps):
            embeds = self.decoder.embedding(curr_token)
            zc = torch.cat([z, c], dim=-1).unsqueeze(1)
            inp = self.decoder.in_proj(torch.cat([embeds, zc], dim=-1))
            output, hidden = self.decoder.gru(inp, hidden)
            logits = self.decoder.out_head(output[:, -1, :]) / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1).squeeze(0)

            # Mask out SOS and PAD
            probs[VOCAB_TO_IDX["<SOS>"]] = 0.0
            probs[VOCAB_TO_IDX["<PAD>"]] = 0.0
            probs = probs / probs.sum()

            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            token_str = IDX_TO_VOCAB[next_token_idx]
            if token_str == "<EOS>":
                break
            generated_tokens.append(token_str)
            curr_token = torch.tensor([[next_token_idx]], dtype=torch.long)

        return generated_tokens


class ProtacChemicalAssembler:
    """
    Synthesizes and chemically validates intact PROTAC heterobifunctionals by coupling:
    [Warhead Core (Exit Vector)] -- [Linker Fragment] -- [E3 Recruiter Core (Exit Vector)]
    """

    def __init__(self, warhead_smiles: str, e3_smiles: str):
        self.warhead_smiles = warhead_smiles
        self.e3_smiles = e3_smiles

    def assemble_protac(self, linker_tokens: List[str]) -> Optional[Dict[str, Any]]:
        """
        Assemble the tripartite PROTAC via bio-conjugation:
        - Warhead: JQ1-acid derivative
        - Linker: parsed modular tokens
        - E3 Ligase: 4-amino-pomalidomide derivative
        """
        if not linker_tokens:
            return None

        # Robust independent ring-indexed cores
        jq1_head = "Cc1sc2nc(CC(=O)N"
        jq1_tail = ")nc(-c3ccc(Cl)cc3)c2c1C"
        pom_core = "c4cccc5C(=O)N(C6CCC(=O)NC6=O)C(=O)c45"

        candidate_linkers = self._format_chemically_sound_linkers(linker_tokens)

        for candidate_l in candidate_linkers:
            full_smiles = f"{jq1_head}{candidate_l}{pom_core}{jq1_tail}"
            mol = Chem.MolFromSmiles(full_smiles)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                    linker_heavy_atoms = self._count_linker_span(candidate_l)
                    return {
                        "full_smiles": canonical_smiles,
                        "linker_fragment": candidate_l,
                        "linker_tokens": linker_tokens,
                        "linker_length_atoms": linker_heavy_atoms,
                        "scaffold_type": self._classify_scaffold(candidate_l)
                    }
                except Exception:
                    continue

        return None

    def _format_chemically_sound_linkers(self, tokens: List[str]) -> List[str]:
        """Format token sequence into synthetically viable chemical bridge chains with independent ring numbering."""
        raw_body = "".join(tokens)
        # Adapt heterocyclic ring numbers to prevent clash with JQ1 (rings 1,2,3) and Pom (rings 4,5,6)
        clean_body = raw_body.replace("c1cn(nn1)", "c7cn(nn7)").replace("N1CCNCC1", "N7CCNCC7").replace("c1ccc(cc1)", "c7ccc(cc7)")
        cand1 = f"CC{clean_body}CC(=O)N"
        cand2 = f"C{clean_body}C(=O)N"
        cand3 = f"CCOCC{clean_body}CCOCC(=O)N"
        cand4 = f"CC#CC{clean_body}CC(=O)N"
        cand5 = f"CC7CCN(CC7){clean_body}CC(=O)N"
        cand6 = f"CCc7cn(nn7){clean_body}COCCC(=O)N"
        return [cand6, cand1, cand3, cand4, cand5, cand2]

    def _count_linker_span(self, linker_smiles: str) -> int:
        """Count heavy atoms in the connecting linker bridge."""
        mol = Chem.MolFromSmiles(linker_smiles)
        if mol is not None:
            return mol.GetNumHeavyAtoms()
        return len(linker_smiles) // 2

    def _classify_scaffold(self, linker_smiles: str) -> str:
        """Classify the structural class of the linker."""
        if "c1cn(nn1)" in linker_smiles:
            return "1,2,3-Triazole Rigid Linker"
        elif "1CCN" in linker_smiles or "N1CCN" in linker_smiles:
            return "Piperazine / Heterocyclic Constrained"
        elif "#" in linker_smiles:
            return "Alkynyl Rigid / Semi-Rigid"
        elif "OCC" in linker_smiles or "COCCO" in linker_smiles:
            return "PEG (Polyethylene Glycol) Flexible"
        else:
            return "Aliphatic Alkyl / Hybrid"


class GenerativeProtacPipeline:
    """Master controller that trains/initializes the C-VAE and generates candidate linkers."""

    def __init__(self, warhead_smiles: str, e3_smiles: str, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.warhead_smiles = warhead_smiles
        self.e3_smiles = e3_smiles
        self.warhead_fp = torch.from_numpy(featurize_anchor_smiles(warhead_smiles)).float().unsqueeze(0)
        self.e3_fp = torch.from_numpy(featurize_anchor_smiles(e3_smiles)).float().unsqueeze(0)

        # Instantiate C-VAE model
        self.model = ConditionalLinkerVAE()
        self.assembler = ProtacChemicalAssembler(warhead_smiles, e3_smiles)
        self._pretrain_synthetic_prior()

    def _pretrain_synthetic_prior(self):
        """
        Calibrate C-VAE weights with a compact prior optimization
        over established PROTAC linker synthetic grammars (PEG, alkyl, alkynes, triazoles).
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Synthetic prior exemplars for TPD linkers
        prior_sequences = [
            ["CC", "CCOCC", "CC"],
            ["CCC", "COCCO", "CCC"],
            ["C#C", "CCOCC", "CNC(=O)"],
            ["N1CCNCC1", "CCOCC", "CC"],
            ["c1cn(nn1)", "CCOCC", "CCC"],
            ["CC", "CCCC", "CCOCC"],
            ["C1CCN(CC1)", "CC", "CNC(=O)"],
            ["c1ccc(cc1)", "CCOCC", "CC"],
        ]

        for epoch in range(12):
            total_loss = 0.0
            for seq in prior_sequences:
                # Convert tokens to indices
                tok_indices = [VOCAB_TO_IDX["<SOS>"]] + [VOCAB_TO_IDX[t] for t in seq if t in VOCAB_TO_IDX] + [VOCAB_TO_IDX["<EOS>"]]
                x = torch.tensor([tok_indices], dtype=torch.long)
                t_len = torch.tensor([[float(len(seq) * 3)]], dtype=torch.float32)

                optimizer.zero_grad()
                logits, mu, logvar = self.model(x, self.warhead_fp, self.e3_fp, t_len)
                
                # Cross-entropy reconstruction loss
                shift_logits = logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE)
                shift_targets = x[:, 1:].contiguous().view(-1)
                recon_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=VOCAB_TO_IDX["<PAD>"])

                # KL Divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + (0.01 * kl_loss)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def generate_candidates(self, num_samples: int = 50,
                            target_lengths: Optional[List[int]] = None,
                            temperature: float = 0.85) -> List[Dict[str, Any]]:
        """
        Sample diverse candidates across requested linker lengths and scaffold types.
        """
        if target_lengths is None:
            target_lengths = [8, 10, 12, 14, 16, 18, 20]

        generated_protacs = []
        seen_smiles = set()

        for i in range(num_samples):
            target_len = random.choice(target_lengths)
            # Sample tokens from model
            tokens = self.model.sample_linker(
                self.warhead_fp, self.e3_fp, target_len=target_len,
                temperature=temperature, max_steps=random.randint(2, 5)
            )

            if not tokens:
                # Fallback to grammar template
                tokens = random.choice([
                    ["CCOCC", "CC"],
                    ["N1CCNCC1", "CCOCC"],
                    ["c1cn(nn1)", "CC"],
                    ["C#C", "CCOCC"],
                    ["CCCC", "COCCO"]
                ])

            # Chemically stitch into intact PROTAC
            protac_dict = self.assembler.assemble_protac(tokens)
            if protac_dict is not None:
                smi = protac_dict["full_smiles"]
                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    protac_dict["candidate_id"] = f"PROTAC-AML-{len(generated_protacs) + 1:03d}"
                    generated_protacs.append(protac_dict)

        return generated_protacs

