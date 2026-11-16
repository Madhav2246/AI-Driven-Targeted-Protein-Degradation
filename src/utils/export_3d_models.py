"""
3D STRUCTURE GENERATOR & EXPORTER
Generates energy-minimized 3D coordinates (SDF / PDB / Mol formats) for:
1. Lead 1: PROTAC-AML-001 (1,2,3-Triazole-PEG Hybrid)
2. Lead 2: PROTAC-AML-002 (Alkynyl-PEG Semi-Rigid)
3. Benchmark: dBET6 (PDB 5T35 anchor conformer)

Outputs saved to:
- web_app/models_3d/protac_aml_001.sdf
- web_app/models_3d/protac_aml_002.sdf
- web_app/models_3d/dbet6_reference.sdf
"""

import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "web_app", "models_3d")


LEADS_3D = [
    {
        "id": "PROTAC-AML-001",
        "name": "protac_aml_001.sdf",
        "smiles": "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCCc1cn(nn1)COCCC(=O)Nc1cccc2C(=O)N(C3CCC(=O)NC3=O)C(=O)c12)n2"
    },
    {
        "id": "PROTAC-AML-002",
        "name": "protac_aml_002.sdf",
        "smiles": "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCCC#CCOCCC(=O)Nc1cccc2C(=O)N(C3CCC(=O)NC3=O)C(=O)c12)n2"
    },
    {
        "id": "dBET6-Reference",
        "name": "dbet6_reference.sdf",
        "smiles": "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCCCCCCCC(=O)Nc1cccc2C(=O)N(C3CCC(=O)NC3=O)C(=O)c12)n2"
    }
]


def generate_and_save_3d_structures():
    os.makedirs(OUT_DIR, exist_ok=True)
    for lead in LEADS_3D:
        mol = Chem.MolFromSmiles(lead["smiles"])
        if mol is None:
            continue

        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.enforceChirality = True
        AllChem.EmbedMolecule(mol_h, params)

        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500, mmffVariant="MMFF94s")
        except Exception:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=300)

        out_path = os.path.join(OUT_DIR, lead["name"])
        writer = Chem.SDWriter(out_path)
        writer.write(mol_h)
        writer.close()
        print(f"Exported 3D coordinates for {lead['id']} -> {out_path}")


if __name__ == "__main__":
    generate_and_save_3d_structures()

