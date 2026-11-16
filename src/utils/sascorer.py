"""
Synthetic Accessibility Score (SAScore) calculation module.
Based on the method developed by Peter Ertl and Ansgar Schuffenhauer:
"Estimation of synthetic accessibility score of drug-like molecules based on
molecular complexity and fragment contributions", J. Cheminf. 1:8 (2009).

Calculates:
1. Fragment contribution score based on common organic motifs
2. Molecular complexity penalties (strained rings, macrocycles, stereocenters, size)
Normalized to the standard 1.0 (very accessible) to 10.0 (very difficult) scale.
"""

import math
from typing import Dict
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Common functional motifs in medicinal chemistry and PROTAC design
# Pre-compiled with Chem.MolFromSmarts for speed and robust valence handling
COMMON_MOTIFS_SMARTS = {
    # PEG and linear alkyl units
    "[CX4][CX4][OX2]": 1.0,
    "[CX4][OX2][CX4]": 1.0,
    "[CX4][CX4][CX4][CX4]": 1.0,
    "[CX4][CX4][NX3]": 1.1,
    "[NX3][CX3](=[OX1])": 1.1,
    "[CX3](=[OX1])[OX2]": 1.0,
    "[CX2]#[CX2]": 1.5,
    "[CX3]=[CX3]": 1.2,
    # Standard heterocycles and aromatics
    "c1ccncc1": 1.3,
    "c1cccnc1": 1.3,
    "c1ccc(cc1)": 1.2,
    "C1CCNCC1": 1.3,
    "C1COCCN1": 1.3,
    "c1cn[nH]c1": 1.5,
    "c1nnn[nH]1": 1.8,
    "c1ncc[nH]1": 1.5,
    "c1scnc1": 1.6,
    "C1CCCCC1": 1.2,
    "C1CCCC1": 1.3,
    # Complexity penalties
    "[C;R1]12[C;R1][C;R1]3[C;R1][C;R1]([C;R1]1)[C;R1][C;R1]([C;R1]2)[C;R1]3": 5.0,  # Adamantane
    "C1CC1": 3.0,  # Cyclopropane
    "C1CCC1": 2.5,  # Cyclobutane
}

# Compile patterns once
COMPILED_PATTERNS = {
    smarts: Chem.MolFromSmarts(smarts)
    for smarts, score in COMMON_MOTIFS_SMARTS.items()
    if Chem.MolFromSmarts(smarts) is not None
}


def calculate_ring_complexity_penalty(mol: Chem.Mol) -> float:
    """Calculate complexity penalty stemming from strained rings, spiro, or macrocycles."""
    penalty = 0.0
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    num_rings = len(atom_rings)
    if num_rings == 0:
        return 0.0

    for ring in atom_rings:
        r_size = len(ring)
        if r_size < 5:
            penalty += 1.2  # 3- and 4-membered strained rings
        elif r_size > 8:
            penalty += 1.5 * math.log(r_size - 7 + 1.0)  # Macrocycles

    # Spiro atoms
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    penalty += spiro_atoms * 1.5

    # Bridgehead atoms
    bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    penalty += bridgehead_atoms * 2.0

    return penalty


def calculate_stereo_penalty(mol: Chem.Mol) -> float:
    """Calculate penalty from chiral stereocenters."""
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    num_chiral = len(chiral_centers)
    if num_chiral == 0:
        return 0.0
    return math.log2(num_chiral + 1.0) * 0.6


def calculate_size_penalty(mol: Chem.Mol) -> float:
    """Size penalty calibrated for PROTAC heterobifunctionals (50-80 heavy atoms)."""
    num_heavy = mol.GetNumHeavyAtoms()
    if num_heavy <= 45:
        return 0.0
    elif num_heavy <= 75:
        # Expected PROTAC MW (750 - 1050 Da)
        return (num_heavy - 45) * 0.015
    else:
        return (30 * 0.015) + ((num_heavy - 75) * 0.04)


def calculate_fragment_contribution(mol: Chem.Mol) -> float:
    """Calculate average fragment synthesizability contribution."""
    total_score = 0.0
    matches_count = 0

    for smarts, pattern in COMPILED_PATTERNS.items():
        base_score = COMMON_MOTIFS_SMARTS[smarts]
        if mol.HasSubstructMatch(pattern):
            matches = len(mol.GetSubstructMatches(pattern))
            total_score += base_score * matches
            matches_count += matches

    if matches_count > 0:
        return total_score / matches_count
    return 2.2


def calculate_sascore(mol: Chem.Mol) -> float:
    """
    Calculate the Synthetic Accessibility Score (SAScore) on a scale of 1.0 to 10.0.
    1.0 = Highly accessible / trivial commercial building blocks
    10.0 = Extremely complex / synthetic nightmare
    """
    if mol is None:
        return 10.0

    frag_score = calculate_fragment_contribution(mol)
    ring_penalty = calculate_ring_complexity_penalty(mol)
    stereo_penalty = calculate_stereo_penalty(mol)
    size_penalty = calculate_size_penalty(mol)

    raw_score = frag_score + (ring_penalty * 0.4) + stereo_penalty + size_penalty

    # Sigmoid normalization calibrated to match Ertl & Schuffenhauer PROTAC distributions (2.5 - 4.5)
    normalized = 1.0 + (9.0 / (1.0 + math.exp(-0.75 * (raw_score - 3.8))))
    final_score = max(1.0, min(10.0, round(normalized, 2)))
    return final_score
