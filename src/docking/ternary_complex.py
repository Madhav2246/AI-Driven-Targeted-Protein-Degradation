"""
MODULE 3: TERNARY COMPLEX MODELING & BINDING AFFINITY EVALUATION
Pipeline for POI : PROTAC : E3 Ligase Biophysical Characterization
Primary Target pair: BRD4 (BD1) : PROTAC : Cereblon (CRBN) [PDB Reference: 6BOY]
Cross-Ligase Comparator: BRD4 (BD2) : PROTAC : VHL [PDB Reference: 5T35]

Key components:
1. Low-energy 3D Conformer Generation via RDKit ETKDGv3 + MMFF94s force field relaxation.
2. Euclidean inter-anchor exit vector distance measurement between chemically defined attachment atoms:
   - JQ1: C4-carboxamide carbonyl carbon (attaches to linker)
   - Pomalidomide: phthaloyl 4-amino nitrogen (attaches to linker)
3. Conformer strain energy calculation from the MMFF94s relative conformer ensemble.
4. Heuristic inter-protein contact surface area (empirical ΔSASA) and steric clash penalty evaluation.
5. Thermodynamic cooperativity factor (alpha) and predicted ternary free energy (ΔGternary).

SCIENTIFIC METHODOLOGY & LIMITATIONS NOTE:
------------------------------------------
The biophysical parameters evaluated in this module (cooperativity alpha, ternary deltaG,
apparent Kd, buried SASA, and steric clash penalty) are HEURISTIC EMPIRICAL APPROXIMATIONS
calibrated against the crystallographic geometry of the dBET6:BRD4-BD1:CRBN complex (PDB 6BOY,
Nowak et al., Nat Chem Biol 2018; optimal anchor span 14.2 +/- 2.0 Angstroms).
- RDKit conformer generation, distance measurements, and MMFF94s relative energies are actual
  small-molecule forcefield calculations.
- Protein-protein contact surface area (ΔSASA) and steric clash penalties are analytical distance-
  geometry heuristics, NOT the result of all-atom protein-protein docking, explicit-solvent
  molecular dynamics (MD), or free energy perturbation (FEP).
- The rigid-body tethering entropy penalty (ΔGrigid = +3.60 kcal/mol) is a fixed theoretical
  approximation (Page & Jencks, 1971; Janin & Finkelstein, 1993) and does not capture linker-specific
  rotational entropy loss (~0.5 - 0.7 kcal/mol per rotatable bond).
- Apparent dissociation constants (Kd,app) are derived from the mass-action formula Kd,binary / alpha
  (assuming binary Kd = 50.0 nM for JQ1-Acid). For candidates with extreme negative cooperativity
  (alpha < 1e-4), apparent Kd is undefined/unreliable due to severe steric clash and is reported as such.
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger("TernaryDocking")

# Primary crystallographic reference: BRD4 BD1 in complex with dBET6 and DDB1-CRBN
CRYSTAL_REFERENCE_PDB = "6BOY"
COMPARATOR_REFERENCE_PDB = "5T35"  # BRD4 BD2 with MZ1 and VHL (cross-ligase reference)

# In PDB 6BOY, the distance between JQ1 C4-carbonyl carbon and Pomalidomide C4-amino nitrogen
# spans an optimal crystallographic distance of 14.2 +/- 2.0 Angstroms across dBET6.
POCKET_DISTANCE_OPTIMAL = 14.2  # Angstroms (derived from PDB 6BOY)
POCKET_DISTANCE_TOLERANCE = 2.0  # Angstroms


class TernaryComplexModeler:
    """
    Models the biophysical ternary assembly:
    BRD4 Bromodomain-1 <---> PROTAC Heterobifunctional <---> Cereblon / DDB1
    calibrated to the 14.2 Angstrom crystallographic envelope of PDB 6BOY.
    """

    def __init__(self, pdb_template: str = CRYSTAL_REFERENCE_PDB):
        self.pdb_template = pdb_template
        self.pocket_distance_optimal = POCKET_DISTANCE_OPTIMAL if pdb_template == "6BOY" else 13.2
        # Standard binary binding free energies for anchor compounds
        # JQ1 binding to BRD4 BD1: Kd ~ 50 nM -> deltaG ~ -9.95 kcal/mol
        # Pomalidomide binding to CRBN: Kd ~ 150 nM -> deltaG ~ -9.30 kcal/mol
        self.deltaG_warhead_binary = -9.95  # kcal/mol
        self.deltaG_e3_binary = -9.30       # kcal/mol
        self.rt_constant = 0.5925           # kcal/mol at 298.15 K
        # Thermodynamic rigid-body entropy penalty for tethering two independent proteins
        # into a defined ternary interface (Page & Jencks 1971; Janin & Finkelstein 1993)
        self.rigid_body_entropy = 3.60      # kcal/mol

    def generate_3d_conformers(
        self, mol: Chem.Mol, num_conformers: int = 6, max_iters: int = 250
    ) -> Tuple[Optional[Chem.Mol], List[float], bool]:
        """
        Generate low-energy 3D conformations using ETKDGv3 with MMFF94s minimization.
        Returns:
            - mol_3d: RDKit Mol with 3D conformers (or None)
            - relative_energies: list of relative conformer energies (kcal/mol above min)
            - is_fallback: bool indicating whether fallback embedding was triggered
        """
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None, [], True

        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.pruneRmsThresh = 0.5
        params.enforceChirality = True

        conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_conformers, params=params)
        is_fallback = False
        if len(conf_ids) == 0:
            # Fallback embedding
            res = AllChem.EmbedMolecule(mol_3d, randomSeed=42)
            if res == -1:
                # 2D fallback coordinates if 3D embedding completely fails
                AllChem.Compute2DCoords(mol_3d)
                return mol_3d, [], True
            conf_ids = [0]
            is_fallback = True

        energies = []
        for cid in conf_ids:
            try:
                # MMFF94s force field optimization
                AllChem.MMFFOptimizeMolecule(mol_3d, confId=cid, maxIters=max_iters, mmffVariant="MMFF94s")
                ff_props = AllChem.MMFFGetMoleculeProperties(mol_3d, mmffVariant="MMFF94s")
                ff = AllChem.MMFFGetMoleculeForceField(mol_3d, ff_props, confId=cid)
                if ff is not None:
                    e = ff.CalcEnergy()
                else:
                    e = 999.0
            except Exception:
                # UFF fallback
                try:
                    AllChem.UFFOptimizeMolecule(mol_3d, confId=cid, maxIters=max_iters)
                    uff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=cid)
                    e = uff.CalcEnergy() if uff is not None else 999.0
                except Exception:
                    e = 999.0
            energies.append(e)

        min_energy = min(energies) if energies else 0.0
        relative_energies = [round(e - min_energy, 2) for e in energies]
        return mol_3d, relative_energies, is_fallback

    @staticmethod
    def find_anchor_exit_atoms(mol: Chem.Mol) -> Tuple[Optional[int], Optional[int]]:
        """
        Identify the specific attachment atoms defining the linker exit vectors:
        - JQ1: C4-carboxamide carbonyl carbon (attaches to linker)
        - Pomalidomide: phthaloyl 4-amino nitrogen (attaches to linker)
        """
        query_jq1 = Chem.MolFromSmarts("c1sc2nc(C[C:1](=[O:2])[N:3])nc(-c3ccc(Cl)cc3)c2c1C")
        query_pom = Chem.MolFromSmarts("[N:1]c1cccc2c1C(=O)N(C3CCC(=O)NC3=O)C2=O")

        jq1_matches = mol.GetSubstructMatches(query_jq1)
        pom_matches = mol.GetSubstructMatches(query_pom)

        c_idx = None
        if jq1_matches:
            for q_atom, idx in zip(query_jq1.GetAtoms(), jq1_matches[0]):
                if q_atom.GetAtomMapNum() == 1:
                    c_idx = idx
                    break

        n_idx = None
        if pom_matches:
            for q_atom, idx in zip(query_pom.GetAtoms(), pom_matches[0]):
                if q_atom.GetAtomMapNum() == 1:
                    n_idx = idx
                    break

        return c_idx, n_idx

    def calculate_end_to_end_distance(self, mol_3d: Chem.Mol, conf_id: int = 0) -> Tuple[float, bool]:
        """
        Measure Euclidean distance between chemically defined anchor exit vectors
        (JQ1 C4-carbonyl carbon to Pomalidomide 4-amino nitrogen) in the given 3D conformer.
        Returns:
            - distance_A: float (in Angstroms)
            - is_fallback_span: bool (True if exit atoms could not be resolved and topological fallback was used)
        """
        try:
            c_idx, n_idx = self.find_anchor_exit_atoms(mol_3d)
            conf = mol_3d.GetConformer(conf_id)
            if c_idx is not None and n_idx is not None:
                p1 = conf.GetAtomPosition(c_idx)
                p2 = conf.GetAtomPosition(n_idx)
                dist = float(np.linalg.norm(np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z])))
                return round(dist, 2), False
            else:
                # Fallback: topological span between farthest heavy atoms
                positions = conf.GetPositions()
                dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
                return round(float(np.percentile(dists, 80)), 2), True
        except Exception:
            return 12.5, True

    def evaluate_steric_and_interface(
        self, linker_length_atoms: int, end_to_end_dist: float
    ) -> Tuple[float, float, int]:
        """
        Heuristic interface scoring calibrated to PDB 6BOY (optimal span 14.2 A):
        1. Heuristic inter-protein contact surface area (empirical delta SASA in Angstroms^2).
           Productive ternary complex formation in 6BOY exhibits ~500 - 580 A^2 of inter-protein contact.
        2. Heuristic steric clash penalty (kcal/mol).
           Short linkers (<10 atoms) or spans < 12 A force steric overlap between BRD4 and CRBN surfaces.
           Excessively long linkers (>18 atoms) hold proteins apart, reducing interface burial.
        3. Steric clash count.
        """
        dist_delta = end_to_end_dist - self.pocket_distance_optimal

        # Optimal span yields maximum cooperative contact (~500-580 A^2 in 6BOY)
        if abs(dist_delta) <= 2.2 and 10 <= linker_length_atoms <= 18:
            inter_protein_sasa = 560.0 - (abs(dist_delta) * 35.0)
            clash_penalty = 0.2 + (abs(dist_delta) * 0.1)
            clash_count = 0
        elif dist_delta < -2.2 or linker_length_atoms < 10:
            # Overly short linker: forces collision between protein surfaces
            shortfall = abs(dist_delta) - 2.2 if dist_delta < -2.2 else 1.0
            shortfall = max(0.5, shortfall)
            inter_protein_sasa = max(80.0, 450.0 - (shortfall * 70.0))
            clash_penalty = 2.0 + (shortfall * 2.5)
            clash_count = int(math.ceil(shortfall * 1.5))
        elif dist_delta > 2.2 or linker_length_atoms > 18:
            # Overly long linker: proteins held apart, reduced interface contact
            excess = dist_delta - 2.2 if dist_delta > 2.2 else 1.0
            excess = max(0.5, excess)
            inter_protein_sasa = max(80.0, 450.0 - (excess * 40.0))
            clash_penalty = 0.5 + (excess * 0.25)
            clash_count = 0
        else:
            inter_protein_sasa = 400.0 - (abs(dist_delta) * 20.0)
            clash_penalty = 0.8
            clash_count = 0

        inter_protein_sasa = round(max(50.0, inter_protein_sasa), 1)
        clash_penalty = round(clash_penalty, 2)
        return inter_protein_sasa, clash_penalty, clash_count

    def compute_cooperativity_and_affinity(
        self, strain_energy: Optional[float], delta_sasa: float, clash_penalty: float
    ) -> Dict[str, Any]:
        """
        Compute:
        - Thermodynamic cooperativity factor alpha = exp(-delta_delta_G / RT) without artificial caps.
        - Predicted Stepwise & Total Ternary Binding Free Energies (kcal/mol)
        - Apparent ternary dissociation constant Kd,app = Kd,binary(POI) / alpha (nM)
        - Cooperativity status: Positive (alpha >= 1.05), Neutral (0.95 <= alpha < 1.05), Negative (alpha < 0.95)
        """
        actual_strain = strain_energy if strain_energy is not None else 2.5
        # Inter-protein buried contact surface provides favorable free energy:
        # ~ -0.008 kcal/mol per A^2 of buried hydrophobic surface
        deltaG_protein_contact = -(delta_sasa * 0.008)

        # Total free energy perturbation (delta delta G)
        # Includes rigid-body tethering entropy penalty, favorable contact, conformer strain & steric clash
        delta_delta_g = self.rigid_body_entropy + deltaG_protein_contact + (actual_strain * 0.25) + clash_penalty

        # Cooperativity factor alpha:
        # alpha = exp(- delta_delta_g / RT)
        # Numerical safeguard: prevent floating point overflow/underflow without arbitrary caps
        exp_arg = -delta_delta_g / self.rt_constant
        exp_arg_safe = max(-50.0, min(50.0, exp_arg))
        alpha = round(math.exp(exp_arg_safe), 4)

        # Total ternary assembly free energy (A + B + D -> ABD):
        deltaG_ternary = round(self.deltaG_warhead_binary + self.deltaG_e3_binary + delta_delta_g, 2)

        # Stepwise ternary binding free energy (BD + A -> ABD):
        deltaG_stepwise = round(self.deltaG_warhead_binary + delta_delta_g, 2)

        # Apparent ternary dissociation constant for POI recruitment:
        # Kd,app = Kd,binary(POI) / alpha (Binary JQ1 Kd = 50.0 nM)
        # If alpha < 1e-4 (severe steric clash), apparent Kd is undefined / unreliable
        if alpha >= 1e-4:
            apparent_kd_nm: Optional[float] = round(50.0 / alpha, 2)
            apparent_kd_str = f"{apparent_kd_nm:.2f} nM"
        else:
            apparent_kd_nm = None
            apparent_kd_str = ">10,000 nM (Unreliable / Severe Clash)"

        if alpha >= 1.05:
            status = "Positive Cooperativity (Synergistic)"
        elif alpha >= 0.95:
            status = "Neutral Cooperativity"
        else:
            status = "Negative Cooperativity (Antagonistic / Steric Clash)"

        return {
            "cooperativity_alpha": alpha,
            "heuristic_cooperativity_alpha": alpha,
            "cooperativity_status": status,
            "deltaG_ternary_kcal": deltaG_ternary,
            "heuristic_deltaG_ternary_kcal": deltaG_ternary,
            "deltaG_stepwise_kcal": deltaG_stepwise,
            "delta_delta_g_kcal": round(delta_delta_g, 3),
            "apparent_kd_nm": apparent_kd_nm,
            "apparent_kd_display": apparent_kd_str,
            "inter_protein_sasa_A2": delta_sasa,
            "heuristic_buried_sasa_A2": delta_sasa,
            "steric_clash_penalty_kcal": clash_penalty,
            "heuristic_clash_penalty_kcal": clash_penalty,
            "strain_energy_kcal": actual_strain if strain_energy is not None else 0.0,
            "conformer_strain_energy_kcal": strain_energy,
            "strain_calculated": strain_energy is not None,
            "is_heuristic_score": True,
        }

    def evaluate_protac_ternary_fit(
        self, smiles: str, linker_length_atoms: int, fast_eval: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        In silico ternary conformation and biophysical scoring workflow.
        Evaluates the lowest-energy conformer as the unbiased baseline state,
        records the full conformational distance ensemble, and calculates
        actual MMFF94s relative conformational strain required to span the pocket.
        """
        if not smiles or not smiles.strip():
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            return None

        # 1. 3D Conformer Generation & Forcefield relaxation
        n_confs = 3 if fast_eval else 8
        max_it = 80 if fast_eval else 250
        mol_3d, rel_energies, is_embedding_fallback = self.generate_3d_conformers(
            mol, num_conformers=n_confs, max_iters=max_it
        )

        target_span = self.pocket_distance_optimal  # 14.2 Angstroms

        if not rel_energies or mol_3d is None:
            # Conformer generation or minimization failed
            conformer_strain: Optional[float] = None
            end_to_end = 14.0
            mean_span = 14.0
            min_span = 14.0
            max_span = 14.0
            conf_count = 0
            is_span_fallback = True
        else:
            conf_count = len(rel_energies)
            # Measure exit-vector distances across all generated conformers
            span_results = [
                self.calculate_end_to_end_distance(mol_3d, conf_id=cid)
                for cid in range(conf_count)
            ]
            all_spans = [s[0] for s in span_results]
            is_span_fallback = any(s[1] for s in span_results)

            # The lowest-energy conformer (relative energy 0.0) represents the resting baseline
            min_cid = rel_energies.index(0.0) if 0.0 in rel_energies else 0
            end_to_end = all_spans[min_cid]
            mean_span = round(float(np.mean(all_spans)), 2)
            min_span = round(float(np.min(all_spans)), 2)
            max_span = round(float(np.max(all_spans)), 2)

            # Calculate MMFF94s conformational strain energy required to bridge the target pocket:
            # 1. Does the resting conformer already bridge the pocket within tolerance?
            if abs(end_to_end - target_span) <= POCKET_DISTANCE_TOLERANCE:
                conformer_strain = 0.0
            else:
                # 2. Look for conformers in the ensemble that bridge the pocket within tolerance
                bridging_conf_indices = [
                    i for i, s in enumerate(all_spans)
                    if abs(s - target_span) <= POCKET_DISTANCE_TOLERANCE
                ]
                if bridging_conf_indices:
                    # Strain is the lowest relative energy among bridging conformers
                    conformer_strain = round(float(min(rel_energies[i] for i in bridging_conf_indices)), 2)
                else:
                    # 3. If no conformer reaches the window, find the conformer with closest span
                    closest_idx = int(np.argmin([abs(s - target_span) for s in all_spans]))
                    residual_mismatch = abs(all_spans[closest_idx] - target_span) - POCKET_DISTANCE_TOLERANCE
                    harmonic_penalty = round(0.35 * (residual_mismatch ** 2), 2)
                    conformer_strain = round(rel_energies[closest_idx] + harmonic_penalty, 2)

        # 2. Steric clashes & inter-protein interface compatibility (heuristic model)
        delta_sasa, clash_penalty, clash_count = self.evaluate_steric_and_interface(
            linker_length_atoms=linker_length_atoms, end_to_end_dist=end_to_end
        )

        # 3. Cooperativity and free energy calculations
        thermo = self.compute_cooperativity_and_affinity(
            strain_energy=conformer_strain, delta_sasa=delta_sasa, clash_penalty=clash_penalty
        )

        reported_strain = conformer_strain if conformer_strain is not None else 0.0

        return {
            "smiles": smiles,
            "end_to_end_distance_A": end_to_end,
            "mean_conformer_distance_A": mean_span,
            "min_conformer_distance_A": min_span,
            "max_conformer_distance_A": max_span,
            "target_distance_A": target_span,
            "distance_mismatch_A": round(abs(end_to_end - target_span), 2),
            "conformer_count": conf_count,
            "linker_span_atoms": linker_length_atoms,
            "conformer_strain_energy_kcal": conformer_strain,
            "strain_energy_kcal": reported_strain,
            "strain_calculated": conformer_strain is not None,
            "inter_protein_sasa_A2": delta_sasa,
            "heuristic_buried_sasa_A2": delta_sasa,
            "steric_clash_penalty_kcal": clash_penalty,
            "heuristic_clash_penalty_kcal": clash_penalty,
            "clash_count": clash_count,
            "cooperativity_alpha": thermo["cooperativity_alpha"],
            "heuristic_cooperativity_alpha": thermo["heuristic_cooperativity_alpha"],
            "cooperativity_status": thermo["cooperativity_status"],
            "deltaG_ternary_kcal": thermo["deltaG_ternary_kcal"],
            "heuristic_deltaG_ternary_kcal": thermo["heuristic_deltaG_ternary_kcal"],
            "delta_delta_g_kcal": thermo["delta_delta_g_kcal"],
            "apparent_kd_nm": thermo["apparent_kd_nm"],
            "apparent_kd_display": thermo["apparent_kd_display"],
            "conformer_generation_fallback": is_embedding_fallback or is_span_fallback,
            "is_heuristic_score": True,
            "scoring_methodology": "Distance-Geometry Empirical Analytical Approximation (calibrated to PDB 6BOY)",
        }
