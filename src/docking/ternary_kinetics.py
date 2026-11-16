"""
TERNARY COMPLEX FORMATION KINETICS & HOOK EFFECT SIMULATION ENGINE
Mathematical modeling of heterobifunctional PROTAC equilibrium thermodynamics
and 3-body mass action equilibria.

System Species:
P   = Free PROTAC
T   = Free Target POI (e.g., BRD4)
E   = Free E3 Ubiquitin Ligase (e.g., CRBN)
PT  = Binary PROTAC-Target complex
PE  = Binary PROTAC-E3 complex
PTE = Functional Ternary complex [BRD4 : PROTAC : CRBN]

Governing Equilibria:
1. PT formation:   P + T  <==> PT     (Dissociation constant: Kd_T)
2. PE formation:   P + E  <==> PE     (Dissociation constant: Kd_E)
3. Ternary via PT: PT + E <==> PTE    (Dissociation constant: Kd_E / alpha)
4. Ternary via PE: PE + T <==> PTE    (Dissociation constant: Kd_T / alpha)

Where:
alpha = Thermodynamic cooperativity factor (> 1: positive synergy; < 1: negative cooperativity)
"""

import math
from typing import Dict, List, Tuple, Any
import numpy as np


class TernaryKineticsEngine:
    """
    Simulates concentration-dependent ternary complex formation and quantitative
    characterization of the bell-shaped hook effect across varying cooperativity factors.
    """

    def __init__(self, kd_poi_nm: float = 50.0, kd_e3_nm: float = 150.0,
                 total_poi_nm: float = 100.0, total_e3_nm: float = 100.0):
        self.kd_t = kd_poi_nm
        self.kd_e = kd_e3_nm
        self.t_total = total_poi_nm
        self.e_total = total_e3_nm

    def solve_equilibrium(self, p_conc_nm: float, alpha: float, max_iter: int = 100, tol: float = 1e-6) -> Dict[str, float]:
        """
        Numerically solve the coupled non-linear mass action conservation equations:
        T_total = [T] + [PT] + [PTE]
        E_total = [E] + [PE] + [PTE]
        P_total = [P] + [PT] + [PE] + [PTE]
        """
        # Initial guess: free species equal to totals
        free_t = self.t_total / (1.0 + (p_conc_nm / self.kd_t))
        free_e = self.e_total / (1.0 + (p_conc_nm / self.kd_e))

        for _ in range(max_iter):
            # Binary species
            pt = (p_conc_nm * free_t) / self.kd_t
            pe = (p_conc_nm * free_e) / self.kd_e

            # Ternary species governed by cooperativity alpha
            # [PTE] = (alpha * [P] * [T] * [E]) / (Kd_T * Kd_E)
            pte = (alpha * p_conc_nm * free_t * free_e) / (self.kd_t * self.kd_e)

            # Residual conservation errors
            f_t = free_t + pt + pte - self.t_total
            f_e = free_e + pe + pte - self.e_total

            if abs(f_t) < tol and abs(f_e) < tol:
                break

            # Damped Newton-Raphson update
            df_t_d_free_t = 1.0 + (p_conc_nm / self.kd_t) + (alpha * p_conc_nm * free_e / (self.kd_t * self.kd_e))
            df_e_d_free_e = 1.0 + (p_conc_nm / self.kd_e) + (alpha * p_conc_nm * free_t / (self.kd_t * self.kd_e))

            free_t = max(1e-12, free_t - 0.7 * (f_t / df_t_d_free_t))
            free_e = max(1e-12, free_e - 0.7 * (f_e / df_e_d_free_e))

        pt = (p_conc_nm * free_t) / self.kd_t
        pe = (p_conc_nm * free_e) / self.kd_e
        pte = (alpha * p_conc_nm * free_t * free_e) / (self.kd_t * self.kd_e)

        fraction_degraded = min(1.0, pte / self.t_total)

        return {
            "p_total_nm": p_conc_nm,
            "free_t_nm": free_t,
            "free_e_nm": free_e,
            "binary_pt_nm": pt,
            "binary_pe_nm": pe,
            "ternary_pte_nm": pte,
            "ternary_fraction": fraction_degraded,
            "percent_ternary": fraction_degraded * 100.0,
        }

    def simulate_dose_response(self, alpha: float,
                               conc_min_log10: float = -2.0,
                               conc_max_log10: float = 6.0,
                               num_points: int = 80) -> Dict[str, Any]:
        """
        Generate full bell-shaped dose-response profile across concentration gradient.
        Returns peak concentration, maximal ternary complex formed (Dmax), and DC50.
        """
        concentrations = np.logspace(conc_min_log10, conc_max_log10, num_points)
        curve_data = []

        max_pte = -1.0
        p_opt_nm = 0.0

        for conc in concentrations:
            res = self.solve_equilibrium(conc, alpha=alpha)
            curve_data.append(res)
            if res["ternary_pte_nm"] > max_pte:
                max_pte = res["ternary_pte_nm"]
                p_opt_nm = conc

        # Determine DC50 (concentration at which half-maximal ternary complex is formed on ascending limb)
        half_max = max_pte / 2.0
        dc50_nm = 0.0
        for pt in curve_data:
            if pt["ternary_pte_nm"] >= half_max:
                dc50_nm = pt["p_total_nm"]
                break

        # Determine Hook Threshold (concentration on descending limb where ternary complex drops to 50% of peak)
        c_hook_nm = 1e6
        passed_peak = False
        for pt in curve_data:
            if pt["p_total_nm"] >= p_opt_nm:
                passed_peak = True
            if passed_peak and pt["ternary_pte_nm"] <= half_max:
                c_hook_nm = pt["p_total_nm"]
                break

        therapeutic_window_fold = round(c_hook_nm / max(1e-4, dc50_nm), 1)
        d_max_percent = round((max_pte / self.t_total) * 100.0, 2)

        return {
            "alpha": alpha,
            "dc50_nm": round(dc50_nm, 2),
            "optimal_conc_nm": round(p_opt_nm, 2),
            "hook_threshold_nm": round(c_hook_nm, 2),
            "therapeutic_window_fold": therapeutic_window_fold,
            "d_max_percent": d_max_percent,
            "curve_points": curve_data,
        }

