"""
MODULE 2: MULTI-OBJECTIVE SCORING & FILTERING ENGINE
Implements:
1. Synthetic Accessibility Score (SAScore / Ertl algorithm).
2. Quantitative Estimate of Drug-likeness (QED).
3. "Beyond Rule of 5" (bRo5) physicochemical boundaries for heterobifunctional PROTACs.
4. Structural alert & PAINS (Pan-Assay Interference Compounds) elimination.
5. Composite Multi-Parameter Optimization (MPO) desirability score.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# Import internal SAScore
from src.utils.sascorer import calculate_sascore


# Structural alerts / toxicophores / reactive substructures unacceptable in TPD lead candidates
UNWANTED_SUBSTRUCTURE_SMARTS = {
    "acyl_halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]",
    "aliphatic_ester_unstable": "[CX3](=[OX1])[OX2][CX4;!$(C(C)(C)C);!$(c)]",  # highly labile linear esters
    "peroxide": "[OX2][OX2]",
    "hydrazine_unhindered": "[NX3][NX3;H2]",
    "epoxide": "[OX2]1[CX3,CX4][CX3,CX4]1",
    "aziridine": "[NX3]1[CX4][CX4]1",
    "diazo": "[NX2]=[NX2]",
    "thioester": "[CX3](=[OX1])[SX2]",
    "nitroso": "[NX2]=[OX1]",
    "heavy_metal": "[Hg,As,Pb,Cd,Cr,Tl,Sb,Te]",
}


class ProtacPropertyCalculator:
    """Computes all key physicochemical, structural, and topological properties for PROTACs."""

    def __init__(self):
        # Initialize PAINS filter catalog
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
        self.pains_catalog = FilterCatalog(params)

        # Precompile reactive group SMARTS
        self.alert_patterns = {
            name: Chem.MolFromSmarts(smarts)
            for name, smarts in UNWANTED_SUBSTRUCTURE_SMARTS.items()
            if Chem.MolFromSmarts(smarts) is not None
        }

    def evaluate_pains(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """Check for Pan-Assay Interference Compounds."""
        if mol is None:
            return True, ["Invalid molecule"]
        matches = self.pains_catalog.GetMatches(mol)
        flagged = len(matches) > 0
        reasons = [m.GetDescription() for m in matches]
        return flagged, reasons

    def evaluate_reactive_alerts(self, mol: Chem.Mol) -> Tuple[bool, List[str]]:
        """Check for toxicophoric or reactive structural alerts."""
        if mol is None:
            return True, ["Invalid molecule"]
        detected = []
        for alert_name, pattern in self.alert_patterns.items():
            if mol.HasSubstructMatch(pattern):
                detected.append(alert_name)
        return len(detected) > 0, detected

    def compute_bro5_compliance(self, props: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess compliance with empirical 'Beyond Rule of 5' (bRo5) guidelines for PROTAC degraders:
        - Molecular Weight (MW): 700 - 1150 Da (Permeability & degradation competent)
        - cLogP: 1.5 - 5.8 (Lipophilicity window avoiding insoluble precipitants)
        - TPSA: 120 - 240 Å² (Chameleonic passive permeability envelope)
        - Rotatable Bonds (NRot): 8 - 20 (Avoids excessive entropic penalty)
        - Hydrogen Bond Donors (HBD): <= 5
        - Hydrogen Bond Acceptors (HBA): <= 15
        """
        mw = props["mw"]
        logp = props["logp"]
        tpsa = props["tpsa"]
        nrot = props["rot_bonds"]
        hbd = props["hbd"]
        hba = props["hba"]

        checks = {
            "mw_pass": 650.0 <= mw <= 1200.0,
            "logp_pass": 1.0 <= logp <= 6.2,
            "tpsa_pass": 100.0 <= tpsa <= 250.0,
            "nrot_pass": 6 <= nrot <= 24,
            "hbd_pass": hbd <= 5,
            "hba_pass": hba <= 16,
        }
        
        num_passed = sum(checks.values())
        score = num_passed / len(checks)
        return {
            "checks": checks,
            "passed_fraction": round(score, 2),
            "is_bro5_compliant": num_passed >= 5,  # Allow at most 1 minor boundary stretch
        }

    def compute_all_metrics(self, smiles: str) -> Optional[Dict[str, Any]]:
        """Full multi-objective profiling of a PROTAC compound."""
        if not smiles or not smiles.strip():
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None

        # Standard RDKit Descriptors
        mw = round(Descriptors.MolWt(mol), 2)
        logp = round(Descriptors.MolLogP(mol), 2)
        tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        nrot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        num_heavy = mol.GetNumHeavyAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        # QED & SAScore
        qed_val = round(QED.default(mol), 3)
        sascore_val = round(calculate_sascore(mol), 2)

        # Filters
        pains_flag, pains_reasons = self.evaluate_pains(mol)
        alert_flag, alert_reasons = self.evaluate_reactive_alerts(mol)

        raw_props = {
            "mw": mw,
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "rot_bonds": nrot,
            "heavy_atoms": num_heavy,
            "rings": num_rings,
            "aromatic_rings": num_aromatic_rings,
            "qed": qed_val,
            "sascore": sascore_val,
        }

        bro5 = self.compute_bro5_compliance(raw_props)

        # Composite MPO Desirability Score (0.0 to 1.0)
        # 1. SAScore desirability: lower is better (1.0 -> 1.0, 10.0 -> 0.0)
        desirability_sa = max(0.0, 1.0 - (sascore_val - 1.0) / 9.0)
        # 2. QED desirability
        desirability_qed = qed_val
        # 3. bRo5 fraction
        desirability_bro5 = bro5["passed_fraction"]
        # 4. Penalty for structural alerts
        penalty = 0.0
        if pains_flag:
            penalty += 0.35
        if alert_flag:
            penalty += 0.40

        # Weighted geometric mean for MPO
        weights = {"sa": 0.35, "qed": 0.25, "bro5": 0.40}
        composite_score = (
            (desirability_sa ** weights["sa"]) *
            (desirability_qed ** weights["qed"]) *
            (desirability_bro5 ** weights["bro5"])
        ) - penalty
        composite_score = round(max(0.0, min(1.0, composite_score)), 3)

        passed_filters = (not pains_flag) and (not alert_flag) and bro5["is_bro5_compliant"] and (sascore_val <= 6.0)

        return {
            "smiles": smiles,
            "mw": mw,
            "logp": logp,
            "tpsa": tpsa,
            "rot_bonds": nrot,
            "hbd": hbd,
            "hba": hba,
            "qed": qed_val,
            "sascore": sascore_val,
            "pains_flag": pains_flag,
            "pains_reasons": pains_reasons,
            "alert_flag": alert_flag,
            "alert_reasons": alert_reasons,
            "bro5_compliance": bro5,
            "composite_score": composite_score,
            "passed_all_filters": passed_filters,
        }

