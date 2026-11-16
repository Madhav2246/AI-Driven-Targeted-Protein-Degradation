"""
COMPREHENSIVE AUTOMATED UNIT TESTS & BIOPHYSICAL VALIDATION SUITE
Tests all core modules:
1. Data curation & RDKit standardization (MoleculeStandardizer, ChEMBLDataCurator)
2. Exit vector detection (ProtacAnchorSystem)
3. Synthetic accessibility scoring (calculate_sascore)
4. PROTAC property calculator & bRo5 filtering (ProtacPropertyCalculator)
5. PyTorch C-VAE generative assembly (GenerativeProtacPipeline, ProtacChemicalAssembler)
6. Ternary complex modeling & mass-action kinetics (TernaryKineticsEngine)
7. Checkpoint loading and metadata validation
8. C-VAE latent inference and vocabulary compliance
9. Dynamic chemotype assembly and scaffold classification
10. PDB 6BOY crystallographic reference parameters
11. Robustness against malformed/invalid SMILES inputs
12. Quantitative descriptor consistency across leads
"""

import os
import sys
import pytest
import torch
from rdkit import Chem

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_curation import MoleculeStandardizer, ChEMBLDataCurator, ProtacAnchorSystem, AML_TARGET_CONFIG
from src.utils.sascorer import calculate_sascore
from src.models.scoring_engine import ProtacPropertyCalculator
from src.models.linker_generator import (
    GenerativeProtacPipeline, ProtacChemicalAssembler,
    LINKER_VOCABULARY, VOCAB_TO_IDX, featurize_anchor_smiles
)
from src.docking.ternary_complex import (
    TernaryComplexModeler, CRYSTAL_REFERENCE_PDB, POCKET_DISTANCE_OPTIMAL
)
from src.docking.ternary_kinetics import TernaryKineticsEngine


def test_standardizer_salt_removal():
    std = MoleculeStandardizer()
    salt_smi = "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)O)n2.Cl.O"
    clean_smi = std.standardize(salt_smi)
    assert clean_smi is not None
    assert "." not in clean_smi  # Salts and solvent stripped
    mol = Chem.MolFromSmiles(clean_smi)
    assert mol.GetNumHeavyAtoms() == 22


def test_exit_vector_detection():
    anchor_sys = ProtacAnchorSystem()
    spec = anchor_sys.get_tripartite_specification()
    assert spec["target"] == "BRD4"
    assert spec["e3_ligase"] == "CEREBLON"
    assert len(spec["warhead"]["detected_vectors"]) > 0
    assert len(spec["e3_recruiter"]["detected_vectors"]) > 0


def test_sascore_bounds():
    simple_peg = Chem.MolFromSmiles("CCOCCOCC(=O)O")
    score_peg = calculate_sascore(simple_peg)
    assert 1.0 <= score_peg <= 3.5  # Simple PEG is highly accessible

    strained_polycycle = Chem.MolFromSmiles("C12CC3CC(C1)CC(C2)C3")  # Adamantane
    score_strained = calculate_sascore(strained_polycycle)
    assert score_strained > score_peg


def test_bro5_protac_calculator():
    calc = ProtacPropertyCalculator()
    lead_smi = "Cc1sc2nc(CC(=O)NCCc3cn(COCCC(=O)Nc4cccc5c4C(=O)N(C4CCC(=O)NC4=O)C5=O)nn3)nc(-c3ccc(Cl)cc3)c2c1C"
    res = calc.compute_all_metrics(lead_smi)
    assert res is not None
    assert 700.0 <= res["mw"] <= 1100.0
    assert 1.0 <= res["logp"] <= 6.0
    assert res["pains_flag"] is False
    assert res["alert_flag"] is False
    assert res["passed_all_filters"] is True


def test_generative_assembly_validity():
    pipeline = GenerativeProtacPipeline(
        warhead_smiles="Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)O)n2",
        e3_smiles="O=C1c2c(cccc2N)C(=O)N1C1CCC(=O)NC1=O",
        random_seed=42
    )
    candidates = pipeline.generate_candidates(num_samples=5, target_lengths=[12, 14])
    assert len(candidates) > 0
    for cand in candidates:
        mol = Chem.MolFromSmiles(cand["full_smiles"])
        assert mol is not None
        assert cand["linker_length_atoms"] >= 6


def test_ternary_kinetics_hook_effect():
    engine = TernaryKineticsEngine(kd_poi_nm=50.0, kd_e3_nm=150.0)
    sim = engine.simulate_dose_response(alpha=3.65)
    assert sim["dc50_nm"] < 15.0  # Nanomolar degradation potency
    assert sim["d_max_percent"] > 25.0  # Peak instantaneous ternary complex equilibrium fraction
    assert sim["therapeutic_window_fold"] > 100.0


def test_cvae_checkpoint_loading_and_metadata():
    ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints", "cvae_linker_model.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint file missing: {ckpt_path}"
    
    pipeline = GenerativeProtacPipeline(
        warhead_smiles=AML_TARGET_CONFIG["warhead_smiles"],
        e3_smiles=AML_TARGET_CONFIG["e3_recruiter_smiles"],
        checkpoint_path=ckpt_path,
        random_seed=42
    )
    meta = pipeline.checkpoint_metadata
    assert meta.get("loaded_successfully") is True
    assert meta.get("epoch") == 11
    assert meta.get("val_loss") is not None
    assert meta["val_loss"] < 2.5


def test_cvae_inference_latent_and_vocab():
    pipeline = GenerativeProtacPipeline(
        warhead_smiles=AML_TARGET_CONFIG["warhead_smiles"],
        e3_smiles=AML_TARGET_CONFIG["e3_recruiter_smiles"],
        checkpoint_path="checkpoints/cvae_linker_model.pt",
        random_seed=42
    )
    tokens = pipeline.model.sample_linker(
        pipeline.warhead_fp, pipeline.e3_fp, target_len=14, temperature=0.8
    )
    assert isinstance(tokens, list)
    for tok in tokens:
        assert tok in VOCAB_TO_IDX, f"Token {tok} not in linker vocabulary"


def test_dynamic_chemotype_assembly():
    assembler = ProtacChemicalAssembler(
        AML_TARGET_CONFIG["warhead_smiles"],
        AML_TARGET_CONFIG["e3_recruiter_smiles"]
    )
    
    # 1. Alkyl tokens -> Aliphatic Alkyl scaffold
    p1 = assembler.assemble_protac(["CCCC", "CCCC"])
    assert p1 is not None
    assert p1["scaffold_type"] == "Aliphatic Alkyl / Hybrid"
    assert "c7cn(nn7)" not in p1["linker_fragment"]

    # 2. PEG tokens -> PEG Flexible scaffold
    p2 = assembler.assemble_protac(["CCOCC", "COCCO"])
    assert p2 is not None
    assert p2["scaffold_type"] == "PEG (Polyethylene Glycol) Flexible"

    # 3. Alkyne tokens -> Alkynyl Rigid scaffold
    p3 = assembler.assemble_protac(["C#C", "CCCC"])
    assert p3 is not None
    assert p3["scaffold_type"] == "Alkynyl Rigid / Semi-Rigid"

    # 4. Triazole tokens -> 1,2,3-Triazole scaffold
    p4 = assembler.assemble_protac(["c1cn(nn1)", "CCOCC"])
    assert p4 is not None
    assert p4["scaffold_type"] == "1,2,3-Triazole Rigid Linker"


def test_pdb_6boy_reference_parameters():
    assert CRYSTAL_REFERENCE_PDB == "6BOY"
    assert POCKET_DISTANCE_OPTIMAL == 14.2
    modeler = TernaryComplexModeler(pdb_template="6BOY")
    assert modeler.pocket_distance_optimal == 14.2


def test_invalid_smiles_robustness():
    calc = ProtacPropertyCalculator()
    assert calc.compute_all_metrics("invalid_chemical_string_123") is None
    assert calc.compute_all_metrics("") is None

    dock = TernaryComplexModeler(pdb_template="6BOY")
    assert dock.evaluate_protac_ternary_fit("invalid_chemical_string_123", 12) is None
    assert dock.evaluate_protac_ternary_fit("", 12) is None


def test_exact_lead_descriptor_consistency():
    calc = ProtacPropertyCalculator()
    lead_smi = "Cc1sc2nc(CC(=O)NCCCCCCCC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C"
    props = calc.compute_all_metrics(lead_smi)
    assert props is not None
    assert 700.0 <= props["mw"] <= 850.0
    assert 2.0 <= props["logp"] <= 6.0
    assert 150.0 <= props["tpsa"] <= 250.0
    assert props["pains_flag"] is False
    assert props["sascore"] <= 4.0
