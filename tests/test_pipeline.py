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
import numpy as np
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
    """
    Rigorously validates that all candidates in verified_lead_candidates.csv:
    1. Parse into chemically valid RDKit molecules.
    2. Recalculated descriptors (MW, LogP, TPSA, SAScore, QED, PAINS, alerts) match CSV values.
    3. Conformers in verified_lead_structures.sdf match candidate SMILES and possess valid 3D coordinates.
    """
    import csv
    import pytest

    calc = ProtacPropertyCalculator()
    csv_path = os.path.join(PROJECT_ROOT, "results", "final_validated", "verified_lead_candidates.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "results", "thesis_defense", "verified_lead_candidates.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "lead_protac_candidates.csv")

    assert os.path.exists(csv_path), f"Candidate CSV not found at {csv_path}"

    candidates = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            candidates.append(row)

    assert len(candidates) >= 5, f"Expected at least 5 prioritized leads, found {len(candidates)}"

    candidate_ids = [r["candidate_id"] for r in candidates]
    assert len(candidate_ids) == len(set(candidate_ids)), f"Duplicate candidate IDs found: {candidate_ids}"
    smiles_set = [r["smiles"] for r in candidates]
    assert len(smiles_set) == len(set(smiles_set)), f"Duplicate SMILES found in candidate CSV"

    for row in candidates:
        cid = row["candidate_id"]
        smi = row["smiles"]
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Could not parse SMILES for candidate {cid}"

        # Recalculate all physicochemical descriptors
        recalculated = calc.compute_all_metrics(smi)
        assert recalculated is not None

        # Compare with CSV values using tight numerical tolerances
        assert recalculated["mw"] == pytest.approx(float(row["mw"]), abs=0.1), f"MW mismatch on {cid}"
        assert recalculated["logp"] == pytest.approx(float(row["logp"]), abs=0.05), f"LogP mismatch on {cid}"
        assert recalculated["tpsa"] == pytest.approx(float(row["tpsa"]), abs=0.1), f"TPSA mismatch on {cid}"
        assert recalculated["rot_bonds"] == int(row["rot_bonds"]), f"RotBonds mismatch on {cid}"
        assert recalculated["sascore"] == pytest.approx(float(row["sascore"]), abs=0.05), f"SAScore mismatch on {cid}"
        assert recalculated["qed"] == pytest.approx(float(row["qed"]), abs=0.01), f"QED mismatch on {cid}"

        csv_pains = row["pains_flag"].strip().lower() == "true"
        csv_alert = row["alert_flag"].strip().lower() == "true"
        assert recalculated["pains_flag"] == csv_pains, f"PAINS mismatch on {cid}"
        assert recalculated["alert_flag"] == csv_alert, f"Alert flag mismatch on {cid}"

    # Validate corresponding SDF structures
    sdf_path = os.path.join(PROJECT_ROOT, "results", "final_validated", "verified_lead_structures.sdf")
    if not os.path.exists(sdf_path):
        sdf_path = os.path.join(PROJECT_ROOT, "results", "thesis_defense", "verified_lead_structures.sdf")

    assert os.path.exists(sdf_path), f"SDF structure file not found: {sdf_path}"
    suppl = Chem.SDMolSupplier(sdf_path)
    sdf_mols = [m for m in suppl if m is not None]
    assert len(sdf_mols) == len(candidates), f"SDF count ({len(sdf_mols)}) != CSV lead count ({len(candidates)})"

    sdf_names = [m.GetProp("_Name") for m in sdf_mols]
    assert set(candidate_ids) == set(sdf_names), "Candidate IDs in SDF do not match CSV"

    csv_smiles_map = {r["candidate_id"]: r["smiles"] for r in candidates}
    for m in sdf_mols:
        cid = m.GetProp("_Name")
        assert m.GetNumConformers() >= 1, f"SDF molecule {cid} lacks 3D conformer"
        conf = m.GetConformer(0)
        assert conf.Is3D(), f"Conformer {cid} is not flagged as 3D"
        pos = conf.GetPositions()
        assert not np.all(pos == 0.0), f"Conformer {cid} atom positions are all zero"
        assert np.all(np.isfinite(pos)), f"Conformer {cid} contains NaN/infinite coordinates"

        # Check constitutional isomorphism against candidate SMILES
        smi_prop = m.GetProp("SMILES")
        m_no_h = Chem.RemoveHs(m)
        s_sdf = Chem.MolToSmiles(m_no_h, isomericSmiles=False, canonical=True)
        s_ref = Chem.MolToSmiles(Chem.MolFromSmiles(csv_smiles_map[cid]), isomericSmiles=False, canonical=True)
        assert s_sdf == s_ref, f"SDF structure constitutional mismatch on {cid}"


def test_unclipped_cooperativity_and_apparent_kd():
    """
    Verifies that:
    1. Cooperativity alpha is not artificially clamped at 150.0.
    2. Apparent Kd is continuously derived as Kd_app = 50.0 / alpha (nM).
    3. Apparent Kd values across top candidates are not identical flat artificial floors.
    """
    import csv
    import pytest

    csv_path = os.path.join(PROJECT_ROOT, "results", "final_validated", "verified_lead_candidates.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "results", "thesis_defense", "verified_lead_candidates.csv")

    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    alphas = [float(r["cooperativity_alpha"]) for r in reader]
    kds = [float(r["apparent_kd_nm"]) for r in reader]

    for a, kd in zip(alphas, kds):
        assert a != 150.0, "Cooperativity alpha still shows artificial 150.0 clipping!"
        assert a > 0.0, "Cooperativity alpha must be positive"
        expected_kd = 50.0 / a
        assert kd == pytest.approx(expected_kd, rel=1e-2), f"Kd {kd} does not match 50.0 / alpha ({expected_kd})"

    # Verify candidates do not share identical flat Kd floors
    assert len(set(round(k, 3) for k in kds)) > 1, "Apparent Kd values are artificially identical across leads!"


def test_ablation_metrics_structure():
    """
    Verifies that ablation results contain sample counts and distinct baseline performance metrics.
    """
    import csv

    csv_path = os.path.join(PROJECT_ROOT, "results", "final_validated", "ablation_results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "results", "thesis_defense", "ablation_results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "experiments", "ablation_study_results.csv")

    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3, f"Expected 3 ablation models, found {len(rows)}"
    expected_cols = [
        "model_variant", "samples_generated", "valid_molecules_count",
        "chemical_validity_pct", "unique_molecules_count", "uniqueness_pct",
        "novel_molecules_count", "novelty_pct", "bro5_compliant_count",
        "bro5_compliance_pct", "pains_pass_count", "pains_pass_pct",
        "mean_sascore", "mean_qed", "mean_cooperativity_alpha",
        "productive_leads_count", "productive_leads_fraction_pct"
    ]
    for col in expected_cols:
        assert col in rows[0], f"Missing expected column {col} in ablation study"


def test_linker_length_scan_metrics():
    """
    Verifies that linker length scan outputs conformer ensemble spans and target mismatch.
    """
    import csv

    csv_path = os.path.join(PROJECT_ROOT, "results", "final_validated", "linker_length_scan.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "results", "thesis_defense", "linker_length_scan.csv")

    assert os.path.exists(csv_path), f"File not found: {csv_path}"
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 8, f"Expected at least 8 length entries, found {len(rows)}"
    for r in rows:
        assert "mean_conformer_span_A" in r
        assert "target_crystallographic_span_A" in r
        assert float(r["target_crystallographic_span_A"]) == 14.2
        assert "target_pocket_mismatch_A" in r
