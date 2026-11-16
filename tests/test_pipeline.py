"""
COMPREHENSIVE AUTOMATED UNIT TESTS & BIOPHYSICAL VALIDATION SUITE
Tests all core modules:
1. Data curation & RDKit standardization (test_data_curation.py)
2. Synthetic accessibility scoring (test_sascorer.py)
3. PROTAC property calculator & bRo5 filtering (test_scoring_engine.py)
4. PyTorch C-VAE generative assembly (test_linker_generator.py)
5. Ternary complex modeling & mass-action kinetics (test_ternary_complex.py)
"""

import os
import sys
import pytest
from rdkit import Chem

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_curation import MoleculeStandardizer, ChEMBLDataCurator, ProtacAnchorSystem
from src.utils.sascorer import calculate_sascore
from src.models.scoring_engine import ProtacPropertyCalculator
from src.models.linker_generator import GenerativeProtacPipeline, ProtacChemicalAssembler
from src.docking.ternary_complex import TernaryComplexModeler
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
