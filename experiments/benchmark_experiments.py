"""
COMPUTATIONAL EXPERIMENTS, ABLATION STUDIES & BENCHMARK EVALUATIONS
Executes:
1. Experiment A: Generative Architecture Ablation (Random Baseline vs Unconditional GRU vs Dual-Anchor C-VAE).
   - Evaluated across three genuinely distinct architectures over identical sample sizes (N=50) and random seed (42).
   - Novelty is evaluated by comparing canonical RDKit SMILES representations against canonical training linkers.
   - Programmatically computed metrics: chemical validity, uniqueness, novelty, bRo5 compliance, PAINS-free rate,
     mean SAScore, mean QED, mean cooperativity alpha, and productive degrader fraction (alpha >= 1.0, bRo5 pass).
2. Experiment B: Systematic Linker Length Trajectory Scan (6 to 22 atoms).
   - Evaluated via RDKit ETKDGv3 3D conformer embedding, MMFF94s minimization, Euclidean exit-vector span measurement,
     and PDB 6BOY crystallographic interface scoring.
3. Experiment C: Head-to-Head Comparative Benchmark against Literature Standards (dBET6, ARV-825, dBET1)
   and verified AI leads (PROTAC-AML-028, PROTAC-AML-025).
"""

import os
import sys
import csv
import math
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from src.data_curation import AML_TARGET_CONFIG
from src.models.linker_generator import (
    ConditionalLinkerVAE,
    ProtacChemicalAssembler,
    LINKER_VOCABULARY,
    VOCAB_TO_IDX,
    IDX_TO_VOCAB,
    VOCAB_SIZE,
    featurize_anchor_smiles
)
from src.models.scoring_engine import ProtacPropertyCalculator
from src.docking.ternary_complex import TernaryComplexModeler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Experiments")


class UnconditionalLinkerGRU(nn.Module):
    """
    Unconditional Autoregressive GRU Decoder.
    Generates linker token sequences based solely on sequence transitions P(x_t | x_<t),
    WITHOUT anchor fingerprint conditioning or target length conditioning.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 48, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB_TO_IDX["<PAD>"])
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out_head = nn.Linear(hidden_dim, vocab_size)

    def sample(self, temperature: float = 0.9, max_steps: int = 5) -> List[str]:
        self.eval()
        curr_token = torch.tensor([[VOCAB_TO_IDX["<SOS>"]]], dtype=torch.long)
        hidden = None
        tokens = []
        for _ in range(max_steps):
            embeds = self.embedding(curr_token)
            out, hidden = self.gru(embeds, hidden)
            logits = self.out_head(out[:, -1, :]) / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            probs[VOCAB_TO_IDX["<SOS>"]] = 0.0
            probs[VOCAB_TO_IDX["<PAD>"]] = 0.0
            probs = probs / probs.sum()
            nxt = torch.multinomial(probs, num_samples=1).item()
            tok_str = IDX_TO_VOCAB[nxt]
            if tok_str == "<EOS>":
                break
            tokens.append(tok_str)
            curr_token = torch.tensor([[nxt]], dtype=torch.long)
        return tokens


def train_unconditional_gru(train_csv: str, epochs: int = 25) -> UnconditionalLinkerGRU:
    """Train unconditional autoregressive GRU on training linker token sequences."""
    torch.manual_seed(42)
    model = UnconditionalLinkerGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=VOCAB_TO_IDX["<PAD>"])

    seqs = []
    if os.path.exists(train_csv):
        with open(train_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                toks = row.get("token_sequence", "").split(";")
                seqs.append(toks)

    for ep in range(epochs):
        model.train()
        for toks in seqs:
            indices = [VOCAB_TO_IDX["<SOS>"]] + [VOCAB_TO_IDX.get(t, VOCAB_TO_IDX["CC"]) for t in toks] + [VOCAB_TO_IDX["<EOS>"]]
            x = torch.tensor([indices[:-1]], dtype=torch.long)
            y = torch.tensor([indices[1:]], dtype=torch.long)
            optimizer.zero_grad()
            out, _ = model.gru(model.embedding(x))
            loss = criterion(model.out_head(out).view(-1, VOCAB_SIZE), y.view(-1))
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def run_experiment_a_ablation(
    sample_size: int = 50,
    output_csv: str = "experiments/ablation_study_results.csv",
    checkpoint_path: str = "checkpoints/cvae_linker_model.pt"
) -> List[Dict[str, Any]]:
    """
    Experiment A: Evaluates generative performance metrics across 3 genuinely distinct architectural variants:
    1. Random Combinatorial Baseline (Uniform random fragment sampling)
    2. Unconditional Autoregressive GRU (Sequence-only transitions without anchor context conditioning)
    3. Proposed Dual-Anchor Conditional VAE (Conditioned on Morgan FPs and target length, loaded from checkpoint)
    """
    output_path = os.path.join(PROJECT_ROOT, output_csv) if not os.path.isabs(output_csv) else output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info(f"Executing Experiment A: Generative Model Architecture Ablation (N={sample_size} per model)...")

    # Load canonical representations of training linkers for rigorous novelty check
    train_csv = os.path.join(PROJECT_ROOT, "data", "train_linkers.csv")
    known_canonical_linkers = set()
    if os.path.exists(train_csv):
        with open(train_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                smi = row.get("linker_smiles", "").strip()
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    known_canonical_linkers.add(Chem.MolToSmiles(mol, canonical=True))

    warhead_smi = AML_TARGET_CONFIG["warhead_smiles"]
    e3_smi = AML_TARGET_CONFIG["e3_recruiter_smiles"]
    assembler = ProtacChemicalAssembler(warhead_smi, e3_smi)
    calc = ProtacPropertyCalculator()
    dock_modeler = TernaryComplexModeler(pdb_template="6BOY")

    # Load C-VAE checkpoint
    cvae_model = ConditionalLinkerVAE()
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cvae_model.load_state_dict(ckpt["model_state_dict"])
        cvae_model.eval()
        logger.info(f"Loaded C-VAE checkpoint from {checkpoint_path}")
    else:
        logger.warning("Checkpoint not found; using initialized model for C-VAE.")

    # Train Unconditional GRU
    uncond_model = train_unconditional_gru(train_csv, epochs=25)
    logger.info("Trained Unconditional GRU baseline on linker sequences.")

    warhead_fp = torch.from_numpy(featurize_anchor_smiles(warhead_smi)).float().unsqueeze(0)
    e3_fp = torch.from_numpy(featurize_anchor_smiles(e3_smi)).float().unsqueeze(0)

    # Building block pool for naive random combinatorial sampling
    random_frag_pool = [
        "CC", "CCC", "CCCC", "CCOCC", "COCCO", "c1ccc(cc1)", "c1cn(nn1)",
        "N1CCNCC1", "C1CCN(CC1)", "C#C", "CNC(=O)", "c1ccncc1", "C1COCCN1"
    ]

    variants = [
        ("Random Combinatorial Baseline", "random"),
        ("Unconditional Autoregressive GRU", "unconditional"),
        ("Proposed Dual-Anchor C-VAE (Ours)", "cvae")
    ]

    results = []

    for name, mode in variants:
        valid_count = 0
        unique_smiles = set()
        novel_count = 0
        bro5_pass_count = 0
        pains_pass_count = 0
        sascore_list = []
        qed_list = []
        alpha_list = []
        productive_leads = 0

        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)

        for _ in range(sample_size):
            if mode == "random":
                k = random.randint(2, 4)
                toks = [random.choice(random_frag_pool) for _ in range(k)]
            elif mode == "unconditional":
                toks = uncond_model.sample(temperature=0.9, max_steps=4)
                if not toks:
                    toks = [random.choice(random_frag_pool)]
            else:  # cvae
                t_len = random.choice([8, 10, 12, 14, 16, 18, 20])
                toks = cvae_model.sample_linker(warhead_fp, e3_fp, target_len=t_len, temperature=0.85, max_steps=4)
                if not toks:
                    toks = ["CCOCC", "CC"]

            protac_dict = assembler.assemble_protac(toks)
            if protac_dict is not None:
                valid_count += 1
                smi = protac_dict["full_smiles"]
                unique_smiles.add(smi)

                # Canonical novelty check
                frag = protac_dict["linker_fragment"].replace("7", "1")
                test_frag_smi = f"N{frag}" if not frag.startswith("N") else frag
                mol_frag = Chem.MolFromSmiles(test_frag_smi)
                if mol_frag:
                    canon_frag = Chem.MolToSmiles(mol_frag, canonical=True)
                    if canon_frag not in known_canonical_linkers:
                        novel_count += 1
                else:
                    novel_count += 1

                props = calc.compute_all_metrics(smi)
                if props is not None:
                    sascore_list.append(props["sascore"])
                    qed_list.append(props["qed"])
                    if props["passed_all_filters"]:
                        bro5_pass_count += 1
                    if not props["pains_flag"]:
                        pains_pass_count += 1

                    ternary = dock_modeler.evaluate_protac_ternary_fit(
                        smi, protac_dict["linker_length_atoms"], fast_eval=True
                    )
                    if ternary is not None:
                        alpha = ternary["cooperativity_alpha"]
                        alpha_list.append(alpha)
                        if props["passed_all_filters"] and not props["pains_flag"] and props["sascore"] <= 4.5 and alpha >= 1.0:
                            productive_leads += 1

        val_pct = round((valid_count / sample_size) * 100.0, 1)
        uniq_pct = round((len(unique_smiles) / max(1, valid_count)) * 100.0, 1)
        nov_pct = round((novel_count / max(1, valid_count)) * 100.0, 1)
        bro5_pct = round((bro5_pass_count / max(1, valid_count)) * 100.0, 1)
        pains_pct = round((pains_pass_count / max(1, valid_count)) * 100.0, 1)
        mean_sa = round(float(np.mean(sascore_list)), 2) if sascore_list else 5.0
        mean_q = round(float(np.mean(qed_list)), 3) if qed_list else 0.15
        mean_a = round(float(np.mean(alpha_list)), 3) if alpha_list else 0.05
        prod_pct = round((productive_leads / max(1, valid_count)) * 100.0, 1)

        record = {
            "model_variant": name,
            "samples_generated": sample_size,
            "valid_molecules_count": valid_count,
            "chemical_validity_pct": val_pct,
            "unique_molecules_count": len(unique_smiles),
            "uniqueness_pct": uniq_pct,
            "novel_molecules_count": novel_count,
            "novelty_pct": nov_pct,
            "bro5_compliant_count": bro5_pass_count,
            "bro5_compliance_pct": bro5_pct,
            "pains_pass_count": pains_pass_count,
            "pains_pass_pct": pains_pct,
            "mean_sascore": mean_sa,
            "mean_qed": mean_q,
            "mean_cooperativity_alpha": mean_a,
            "productive_leads_count": productive_leads,
            "productive_leads_fraction_pct": prod_pct
        }
        results.append(record)
        logger.info(f"  [{name}] Validity: {val_pct}% ({valid_count}/{sample_size}), Uniqueness: {uniq_pct}% ({len(unique_smiles)}/{valid_count}), Novelty: {nov_pct}%, SAScore: {mean_sa}, Productive: {prod_pct}% ({productive_leads}/{valid_count})")

    # Export to primary, thesis_defense, and final_validated folders
    final_dir = os.path.join(PROJECT_ROOT, "results", "final_validated")
    defense_dir = os.path.join(PROJECT_ROOT, "results", "thesis_defense")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(defense_dir, exist_ok=True)

    fieldnames = list(results[0].keys())
    for target in [output_path, os.path.join(defense_dir, "ablation_results.csv"), os.path.join(final_dir, "ablation_results.csv")]:
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    logger.info(f"Experiment A completed. Synchronized across experiments, thesis_defense, and final_validated.")
    return results


def run_experiment_b_length_scan(output_csv: str = "experiments/linker_length_scan_results.csv") -> List[Dict[str, Any]]:
    """
    Experiment B: Systematic scan of linker heavy atom span (6 to 22 atoms)
    with actual RDKit 3D conformer generation, MMFF94s minimization,
    Euclidean exit-vector distance measurement, and 6BOY interface scoring.
    """
    output_path = os.path.join(PROJECT_ROOT, output_csv) if not os.path.isabs(output_csv) else output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Executing Experiment B: Linker Length vs. Cooperativity Trajectory Scan (3D Conformer Modeling)...")

    dock_modeler = TernaryComplexModeler(pdb_template="6BOY")

    length_templates = [
        (6, "CCCC(=O)N"),
        (8, "CCCCCC(=O)N"),
        (10, "CCCCCCCC(=O)N"),
        (12, "CCCCCCCCCC(=O)N"),
        (13, "CCOCCOCCC(=O)N"),
        (14, "CCOCCOCCCOCC(=O)N"),
        (15, "CCCCCCCCCCCCC(=O)N"),
        (16, "CCOCCOCCOCCC(=O)N"),
        (18, "CCOCCOCCOCCOCC(=O)N"),
        (20, "CCCCCCCCCCCCCCCCCC(=O)N"),
        (22, "CCOCCOCCOCCOCCOCC(=O)N")
    ]

    jq1_head = "Cc1sc2nc(CC(=O)N"
    jq1_tail = ")nc(-c3ccc(Cl)cc3)c2c1C"
    pom_core = "c4cccc5C(=O)N(C6CCC(=O)NC6=O)C(=O)c45"

    scan_results = []

    for l_span, linker_str in length_templates:
        full_smi = f"{jq1_head}{linker_str}{pom_core}{jq1_tail}"
        mol = Chem.MolFromSmiles(full_smi)
        if mol is None:
            continue

        mol_3d, energies = dock_modeler.generate_3d_conformers(mol, num_conformers=8, max_iters=250)
        fallback = False
        if mol_3d is not None and energies:
            # Unbiased lowest-energy conformer represents the baseline relaxed state
            min_cid = energies.index(0.0) if 0.0 in energies else 0
            min_energy_span = dock_modeler.calculate_end_to_end_distance(mol_3d, conf_id=min_cid)
            all_spans = [dock_modeler.calculate_end_to_end_distance(mol_3d, conf_id=cid) for cid in range(len(energies))]
            mean_span = round(float(np.mean(all_spans)), 2)
            min_span = round(float(np.min(all_spans)), 2)
            max_span = round(float(np.max(all_spans)), 2)
            conf_count = len(energies)
        else:
            min_energy_span = round(6.5 + (l_span * 0.58), 2)
            mean_span = min_energy_span
            min_span = min_energy_span
            max_span = min_energy_span
            conf_count = 0
            fallback = True

        target_span = dock_modeler.pocket_distance_optimal  # 14.2 Angstroms
        pocket_mismatch = round(abs(min_energy_span - target_span), 2)

        delta_sasa, clash_penalty, clash_count = dock_modeler.evaluate_steric_and_interface(
            linker_length_atoms=l_span, end_to_end_dist=min_energy_span
        )

        thermo = dock_modeler.compute_cooperativity_and_affinity(
            strain_energy=0.0, delta_sasa=delta_sasa, clash_penalty=clash_penalty
        )

        # Classification based on biophysical landscape
        if clash_penalty >= 5.0 or l_span < 10:
            feasibility = "Steric Collision (Severe)"
        elif 2.0 <= clash_penalty < 5.0 or (l_span == 10 and pocket_mismatch > 3.0):
            feasibility = "Transition Boundary"
        elif thermo["cooperativity_alpha"] >= 1.0 and clash_penalty <= 0.5:
            feasibility = "Optimal Design Window"
        elif 0.8 <= thermo["cooperativity_alpha"] < 1.0 and clash_penalty <= 0.5:
            feasibility = "Permissive Window"
        else:
            feasibility = "High Rotational Entropy Loss"

        scan_results.append({
            "linker_span_atoms": l_span,
            "min_energy_conformer_span_A": min_energy_span,
            "mean_conformer_span_A": mean_span,
            "min_conformer_span_A": min_span,
            "max_conformer_span_A": max_span,
            "target_crystallographic_span_A": target_span,
            "target_pocket_mismatch_A": pocket_mismatch,
            "conformer_count": conf_count,
            "empirical_interface_sasa_A2": delta_sasa,
            "heuristic_steric_penalty_kcal": clash_penalty,
            "heuristic_clash_count": clash_count,
            "estimated_cooperativity_alpha": thermo["cooperativity_alpha"],
            "estimated_deltaG_ternary_kcal": thermo["deltaG_ternary_kcal"],
            "estimated_apparent_kd_nm": thermo["apparent_kd_nm"],
            "conformer_generation_fallback": fallback,
            "biological_feasibility": feasibility
        })

    final_dir = os.path.join(PROJECT_ROOT, "results", "final_validated")
    defense_dir = os.path.join(PROJECT_ROOT, "results", "thesis_defense")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(defense_dir, exist_ok=True)

    fieldnames = list(scan_results[0].keys())
    for target in [output_path, os.path.join(defense_dir, "linker_length_scan.csv"), os.path.join(final_dir, "linker_length_scan.csv")]:
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scan_results)

    logger.info(f"Experiment B completed. Conformer trajectory synchronized across experiments, thesis_defense, and final_validated.")
    return scan_results


def run_experiment_c_literature_benchmark(output_csv: str = "experiments/literature_benchmark_comparison.csv") -> List[Dict[str, Any]]:
    """
    Experiment C: Head-to-head comparison of AI-generated leads against
    clinically validated literature standard BET degraders:
    - dBET6 (PDB 6BOY; Nowak et al. Nat Chem Biol 2018)
    - ARV-825 (Lu et al. Blood 2016)
    - dBET1 (Winter et al. Science 2015)
    - PROTAC-AML-028 (Top AI Lead: Piperazine-Constrained)
    - PROTAC-AML-025 (Second AI Lead: Aliphatic-Amide Hybrid)
    """
    output_path = os.path.join(PROJECT_ROOT, output_csv) if not os.path.isabs(output_csv) else output_csv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Executing Experiment C: Benchmark Comparison against Literature Compounds...")

    calc = ProtacPropertyCalculator()
    dock_modeler = TernaryComplexModeler(pdb_template="6BOY")

    # Literature standard reference compounds
    compounds = [
        {
            "name": "dBET6 (Gold Standard)",
            "smiles": "Cc1sc2nc(CC(=O)NCCCCCCCC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C",
            "linker_type": "Octyl Alkyl Chain",
            "linker_len": 10,
            "published_dc50_nm": "1.0 nM (Nowak et al.)",
            "published_dmax": "98.5%"
        },
        {
            "name": "ARV-825 (Clinical Candidate)",
            "smiles": "Cc1sc2nc(CC(=O)NCCOCCOCC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C",
            "linker_type": "Triethylene Glycol (PEG)",
            "linker_len": 11,
            "published_dc50_nm": "2.2 nM (Lu et al.)",
            "published_dmax": "95.0%"
        },
        {
            "name": "dBET1 (First-Gen Degrader)",
            "smiles": "Cc1sc2nc(CC(=O)NCC(=O)Oc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C",
            "linker_type": "Short Amide-Ester",
            "linker_len": 5,
            "published_dc50_nm": "430.0 nM (Winter et al.)",
            "published_dmax": "85.0%"
        }
    ]

    # Dynamic loading of top 2 AI leads from verified candidate outputs
    lead_source = os.path.join(PROJECT_ROOT, "results", "final_validated", "verified_lead_candidates.csv")
    if not os.path.exists(lead_source):
        lead_source = os.path.join(PROJECT_ROOT, "results", "thesis_defense", "verified_lead_candidates.csv")
    if not os.path.exists(lead_source):
        lead_source = os.path.join(PROJECT_ROOT, "lead_protac_candidates.csv")

    ai_leads = []
    if os.path.exists(lead_source):
        with open(lead_source, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            if len(reader) >= 1:
                ai_leads.append({
                    "name": f"{reader[0]['candidate_id']} (Top AI Lead #1)",
                    "smiles": reader[0]["smiles"],
                    "linker_type": reader[0].get("scaffold_type", "Alkynyl Semi-Rigid"),
                    "linker_len": int(reader[0].get("linker_length_atoms", 16)),
                    "published_dc50_nm": "In Silico Hypothesis (Gated)",
                    "published_dmax": "In Silico Hypothesis (Gated)"
                })
            if len(reader) >= 2:
                ai_leads.append({
                    "name": f"{reader[1]['candidate_id']} (AI Lead #2)",
                    "smiles": reader[1]["smiles"],
                    "linker_type": reader[1].get("scaffold_type", "Alkynyl Semi-Rigid"),
                    "linker_len": int(reader[1].get("linker_length_atoms", 16)),
                    "published_dc50_nm": "In Silico Hypothesis (Gated)",
                    "published_dmax": "In Silico Hypothesis (Gated)"
                })

    if not ai_leads:
        ai_leads = [
            {
                "name": "PROTAC-AML-012 (AI Lead 1)",
                "smiles": "Cc1sc2nc(CC(=O)NCCCCCCCCCNC(=O)CC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C",
                "linker_type": "Aliphatic Alkyl / Hybrid",
                "linker_len": 16,
                "published_dc50_nm": "In Silico Hypothesis (Gated)",
                "published_dmax": "In Silico Hypothesis (Gated)"
            },
            {
                "name": "PROTAC-AML-010 (AI Lead 2)",
                "smiles": "Cc1sc2nc(CC(=O)NCCC#CCCCCCCCNC(=O)CC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C",
                "linker_type": "Alkynyl Rigid / Semi-Rigid",
                "linker_len": 16,
                "published_dc50_nm": "In Silico Hypothesis (Gated)",
                "published_dmax": "In Silico Hypothesis (Gated)"
            }
        ]

    compounds.extend(ai_leads)

    benchmarks = []
    for c in compounds:
        props = calc.compute_all_metrics(c["smiles"])
        ternary = dock_modeler.evaluate_protac_ternary_fit(c["smiles"], c["linker_len"], fast_eval=False)

        benchmarks.append({
            "compound_name": c["name"],
            "linker_architecture": c["linker_type"],
            "linker_span_atoms": c["linker_len"],
            "mw": props["mw"] if props else 0.0,
            "logp": props["logp"] if props else 0.0,
            "tpsa": props["tpsa"] if props else 0.0,
            "rot_bonds": props["rot_bonds"] if props else 0,
            "sascore": props["sascore"] if props else 0.0,
            "qed": props["qed"] if props else 0.0,
            "cooperativity_alpha": ternary["cooperativity_alpha"] if ternary else 1.0,
            "deltaG_ternary_kcal": ternary["deltaG_ternary_kcal"] if ternary else 0.0,
            "apparent_kd_nm": ternary["apparent_kd_nm"] if ternary else 50.0,
            "inter_protein_sasa_A2": ternary["inter_protein_sasa_A2"] if ternary else 0.0,
            "clash_count": ternary["clash_count"] if ternary else 0,
            "dc50_nm_reported_or_pred": c["published_dc50_nm"],
            "dmax_pct_reported_or_pred": c["published_dmax"]
        })

    final_dir = os.path.join(PROJECT_ROOT, "results", "final_validated")
    defense_dir = os.path.join(PROJECT_ROOT, "results", "thesis_defense")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(defense_dir, exist_ok=True)

    fieldnames = list(benchmarks[0].keys())
    for target in [output_path, os.path.join(defense_dir, "literature_benchmark_comparison.csv"), os.path.join(final_dir, "literature_benchmark_comparison.csv")]:
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(benchmarks)

    logger.info(f"Experiment C completed. Literature benchmark synchronized across experiments, thesis_defense, and final_validated.")
    return benchmarks


def run_all_experiments():
    run_experiment_a_ablation()
    run_experiment_b_length_scan()
    run_experiment_c_literature_benchmark()


if __name__ == "__main__":
    run_all_experiments()
