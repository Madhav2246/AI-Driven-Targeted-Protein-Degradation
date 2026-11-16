"""
NEURAL NETWORK TRAINING PIPELINE FOR PROTAC LINKER GENERATION
Trains Conditional Variational Autoencoder (C-VAE) on curated PROTAC-DB linker dataset.
Outputs:
1. Model checkpoints: checkpoints/cvae_linker_model.pt
2. Comprehensive training logs: logs/training_history.csv
3. Quantitative loss curves (Reconstruction, KL Divergence, Validation Loss)
"""

import os
import sys
import csv
import logging
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Ensure project root is accessible
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.linker_generator import (
    ConditionalLinkerVAE,
    VOCAB_TO_IDX,
    IDX_TO_VOCAB,
    VOCAB_SIZE,
    featurize_anchor_smiles
)
from src.data_curation import AML_TARGET_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Train-CVAE")


class ProtacLinkerDataset(Dataset):
    """PyTorch Dataset for Modular PROTAC Linker Sequences with Anchor Conditioning."""

    def __init__(self, csv_path: str, warhead_smiles: str, e3_smiles: str, max_seq_len: int = 10):
        self.samples = []
        self.max_seq_len = max_seq_len
        self.warhead_fp = torch.from_numpy(featurize_anchor_smiles(warhead_smiles)).float()
        self.e3_fp = torch.from_numpy(featurize_anchor_smiles(e3_smiles)).float()

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_tokens = row["token_sequence"].split(";")
                target_len = float(row.get("heavy_atom_count", len(raw_tokens) * 3))
                
                # Tokenize
                tok_ids = [VOCAB_TO_IDX["<SOS>"]]
                for t in raw_tokens:
                    clean_t = t.strip()
                    if clean_t in VOCAB_TO_IDX:
                        tok_ids.append(VOCAB_TO_IDX[clean_t])
                tok_ids.append(VOCAB_TO_IDX["<EOS>"])

                # Pad to max_seq_len
                if len(tok_ids) < self.max_seq_len:
                    tok_ids += [VOCAB_TO_IDX["<PAD>"]] * (self.max_seq_len - len(tok_ids))
                else:
                    tok_ids = tok_ids[:self.max_seq_len]

                self.samples.append({
                    "token_tensor": torch.tensor(tok_ids, dtype=torch.long),
                    "target_len": torch.tensor([target_len / 22.0], dtype=torch.float32),
                    "raw_smiles": row.get("linker_smiles", ""),
                    "linker_type": row.get("linker_type", "General")
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "tokens": item["token_tensor"],
            "target_len": item["target_len"],
            "warhead_fp": self.warhead_fp,
            "e3_fp": self.e3_fp
        }


def train_cvae(epochs: int = 25, batch_size: int = 4, lr: float = 1e-3,
               train_csv: str = "data/train_linkers.csv",
               val_csv: str = "data/val_linkers.csv",
               checkpoint_dir: str = "checkpoints",
               logs_dir: str = "logs") -> List[Dict[str, Any]]:
    """Execute complete training and validation cycle."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    warhead_smi = AML_TARGET_CONFIG["warhead_smiles"]
    e3_smi = AML_TARGET_CONFIG["e3_recruiter_smiles"]

    logger.info(f"Loading training data from {train_csv}...")
    train_dataset = ProtacLinkerDataset(train_csv, warhead_smi, e3_smi)
    val_dataset = ProtacLinkerDataset(val_csv, warhead_smi, e3_smi)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ConditionalLinkerVAE(vocab_size=VOCAB_SIZE, fp_dim=128, latent_dim=32, cond_dim=64, hidden_dim=96)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = []
    best_val_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "cvae_linker_model.pt")

    logger.info(f"Starting C-VAE Training for {epochs} Epochs on {len(train_dataset)} training linkers...")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_recon_loss = 0.0
        train_kl_loss = 0.0
        total_train_loss = 0.0

        for batch in train_loader:
            tokens = batch["tokens"]
            target_len = batch["target_len"]
            warhead_fp = batch["warhead_fp"]
            e3_fp = batch["e3_fp"]

            optimizer.zero_grad()
            logits, mu, logvar = model(tokens, warhead_fp, e3_fp, target_len)

            shift_logits = logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE)
            shift_targets = tokens[:, 1:].contiguous().view(-1)

            recon_loss = F.cross_entropy(shift_logits, shift_targets, ignore_index=VOCAB_TO_IDX["<PAD>"])
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / tokens.size(0)

            # Warm-up KL annealing factor beta
            beta = min(1.0, epoch / 10.0) * 0.02
            loss = recon_loss + (beta * kl_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            total_train_loss += loss.item()

        num_train_batches = max(1, len(train_loader))
        avg_train_loss = total_train_loss / num_train_batches
        avg_recon_loss = train_recon_loss / num_train_batches
        avg_kl_loss = train_kl_loss / num_train_batches

        # Validation Phase
        model.eval()
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch["tokens"]
                target_len = batch["target_len"]
                warhead_fp = batch["warhead_fp"]
                e3_fp = batch["e3_fp"]

                logits, mu, logvar = model(tokens, warhead_fp, e3_fp, target_len)
                shift_logits = logits[:, :-1, :].contiguous().view(-1, VOCAB_SIZE)
                shift_targets = tokens[:, 1:].contiguous().view(-1)

                v_recon = F.cross_entropy(shift_logits, shift_targets, ignore_index=VOCAB_TO_IDX["<PAD>"])
                v_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / tokens.size(0)
                v_loss = v_recon + (0.02 * v_kl)

                val_recon_loss += v_recon.item()
                val_kl_loss += v_kl.item()
                total_val_loss += v_loss.item()

        num_val_batches = max(1, len(val_loader))
        avg_val_loss = total_val_loss / num_val_batches
        avg_val_recon = val_recon_loss / num_val_batches
        avg_val_kl = val_kl_loss / num_val_batches

        curr_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_recon_loss": round(avg_recon_loss, 4),
            "train_kl_loss": round(avg_kl_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_recon_loss": round(avg_val_recon, 4),
            "val_kl_loss": round(avg_val_kl, 4),
            "learning_rate": f"{curr_lr:.6f}"
        }
        history.append(epoch_record)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "vocab": VOCAB_TO_IDX
            }, best_model_path)
            best_flag = " [*BEST CHECKPOINT SAVED]"
        else:
            best_flag = ""

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}) | Val Loss: {avg_val_loss:.4f}{best_flag}")

    # Export training history to CSV
    log_csv = os.path.join(logs_dir, "training_history.csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["epoch", "train_loss", "train_recon_loss", "train_kl_loss", "val_loss", "val_recon_loss", "val_kl_loss", "learning_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    logger.info(f"Training completed successfully! Model checkpoint: {best_model_path}")
    logger.info(f"Training log exported: {log_csv}")
    return history


if __name__ == "__main__":
    train_cvae()

