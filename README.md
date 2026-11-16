# AI-Driven Targeted Protein Degradation (TPD) Lead Compound Generation Pipeline for Acute Myeloid Leukemia (AML) Targets via Generative PROTAC Linker Optimization

**Authors / Collaborators:** Madhav Yalamarthi and Hareesh  
**Affiliation:** Department of Computer Science / Artificial Intelligence, Amrita Vishwa Vidyapeetham, Amritapuri, India  
**Project Scope:** Final Year Project / Publication-Grade Computational Chemical Biology & Generative AI  

---

## 📌 Project Overview
This repository provides a reproducible, *in silico* computational pipeline for the de novo generative design, multi-parameter filtering, and ternary complex biophysical modeling of heterobifunctional Proteolysis Targeting Chimeras (PROTACs). The pipeline is calibrated for **Bromodomain-containing Protein 4 (BRD4)** degradation via recruitment of the **Cereblon (CRBN)** E3 ubiquitin ligase in Acute Myeloid Leukemia (AML).

```
====================================================================================================
                        PROTAC TRIPARTITE SYSTEM ARCHITECTURE
====================================================================================================
   [ BRD4 BD1 Target ]               [ GENERATIVE LINKER ]             [ Cereblon / CRBN ]
   (+)-JQ1 Carboxamide      <===>   C-VAE Sampled Bridge       <===>   4-NH-Pomalidomide
   (PDB: 6BOY / BD1 Pocket)         (6-22 atoms: Alkyl/PEG/Alkyne)     (Tri-Trp Hydrophobic Pocket)
====================================================================================================
```

> [!IMPORTANT]
> **Scientific Integrity & Scope Disclosure:**  
> All generated compounds, binding affinities, and cooperativity values are **in silico computational hypotheses** derived from distance-geometry conformer modeling (ETKDGv3) and empirical forcefield optimization (MMFF94s). No cellular $DC_{50}$, maximal degradation ($D_{\text{max}}$), or *in vivo* anti-leukemic efficacy is claimed without prospective wet-lab synthesis and SPR/HiBiT validation.

---

## 🚀 Key Modules & Research Architecture

* **Module 1: Target Curation & Tripartite Specification (`src/data_curation.py` & `data/`):**
  * `data/chembl_brd4_bioactivities.csv`: 10 validated sub-100 nM BRD4 inhibitors with experimental $IC_{50}/K_i/K_d$ bioactivities.
  * `data/protac_db_benchmark.csv`: Reference degraders (dBET1, dBET6, ARV-825) from PROTAC-DB.
  * `data/train_linkers.csv` & `val_linkers.csv`: Domain-specific dataset of **30 curated linkers** (20 train, 5 validation, 5 test) spanning PEG, alkyl, alkynyl, and triazole motifs.
* **Module 2: Generative Linker Engine (`src/models/` & `checkpoints/`):**
  * `linker_generator.py`: Dual-Anchor Conditional Variational Autoencoder (C-VAE) conditioned on dual 512-bit Morgan fingerprints of the warhead and E3 recruiter, combined with target length embeddings ($L \in [6, 22]$). Incorporates active model checkpoint `checkpoints/cvae_linker_model.pt` (Epoch 11, validation loss 1.8342).
  * `scoring_engine.py`: Multi-objective optimization (Ertl SAScore, QED, empirical beyond-Rule-of-5 criteria, and PAINS-A/B/C and toxicophore elimination).
* **Module 3: In Silico Ternary Complex Biophysics (`src/docking/`):**
  * `ternary_complex.py`: 3D conformer relaxation (ETKDGv3 + MMFF94s), Euclidean exit-vector distance measurement, MMFF94s conformational strain energy calculation, heuristic inter-protein contact area ($\Delta\text{SASA}$), steric clash penalties, and unconstrained thermodynamic cooperativity ($\alpha$).
  * `ternary_kinetics.py`: 3-body mass-action equilibrium and bell-shaped hook effect simulator.
* **Module 4: Master Execution & Defense Outputs (`run_tpd_pipeline.py` & `results/final_validated/`):**
  * `run_tpd_pipeline.py`: Master pipeline executing curation, neural sampling, filtering, 3D modeling, and lead ranking.
  * `experiments/benchmark_experiments.py`: Generative model ablation studies (Random vs Unconditional GRU vs C-VAE), unbiased linker length trajectory scan (6–22 atoms), and literature benchmark comparisons.
  * `src/utils/generate_defense_plots.py`: Publication-grade visualization script producing 5 high-resolution figures.
* **Module 5: Interactive Web Interface Demo (`web_app/`):**
  * `web_app/index.html`: Interactive browser dashboard for live candidate exploration, loss curves, and 3D molecular viewer.

---

## 📊 Top Verified Lead Heterobifunctionals (AML: BRD4–CRBN)

All lead compounds were generated from the active C-VAE checkpoint, chemically assembled with validated valency, and scored against the PDB 6BOY crystallographic pocket without artificial caps or numerical floors:

| Candidate ID | Score | Scaffold Type | Span (atoms) | MW (Da) | cLogP | TPSA (Å²) | RotB | SAScore | QED | Cooperativity ($\alpha_{\text{est}}$) | Apparent $K_d$ (nM) | Buried SASA (Å²) | Clash Penalty (kcal/mol) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PROTAC-AML-012** | **0.617** | Aliphatic Alkyl / Hybrid | 16 | 814.36 | 5.56 | 196.63 | 17 | 2.71 | 0.060 | **2.9832** | **16.76** | 557.2 | 0.21 |
| **PROTAC-AML-010** | **0.609** | Alkynyl Rigid / Semi-Rigid | 16 | 813.33 | 4.92 | 185.99 | 16 | 2.74 | 0.080 | **1.7596** | **28.42** | 528.1 | 0.29 |
| **PROTAC-AML-021** | **0.591** | Alkynyl Rigid / Semi-Rigid | 16 | 810.33 | 4.79 | 196.63 | 14 | 2.76 | 0.060 | **1.6503** | **30.30** | 524.6 | 0.30 |
| **PROTAC-AML-024** | **0.585** | Alkynyl Rigid / Semi-Rigid | 16 | 806.30 | 4.01 | 196.63 | 12 | 2.80 | 0.071 | **1.2105** | **41.31** | 507.9 | 0.35 |
| **PROTAC-AML-006** | **0.574** | Alkynyl Rigid / Semi-Rigid | 17 | 828.30 | 3.22 | 215.09 | 15 | 2.76 | 0.059 | **1.1399** | **43.86** | 504.7 | 0.36 |

### Structure of #1 Prioritized Lead (`PROTAC-AML-012`)
- **Linker Fragment:** `CCCCCCCCCNC(=O)CC(=O)N`
- **Canonical SMILES:**  
  `Cc1sc2nc(CC(=O)NCCCCCCCCCNC(=O)CC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C`
- **Biophysical Rationale:** Spans 14.28 Å in its minimum-energy conformer, matching the 14.2 Å crystallographic pocket of PDB 6BOY, inducing $557.2\text{ \AA}^2$ of inter-protein surface burial with minimal steric clash ($0.21\text{ kcal/mol}$).

---

## 📈 Validated Defense Artifacts (`results/final_validated/`)

All files are verified and synchronized across `results/final_validated/` and `results/thesis_defense/`:

1. **`verified_lead_candidates.csv`**: Physicochemical and biophysical metrics for prioritized leads with unclipped $\alpha$ and apparent $K_d$.
2. **`all_generated_candidates.csv`**: Complete candidate pool ($N=38$) for full traceability.
3. **`verified_lead_structures.sdf`**: Energy-minimized 3D conformers (ETKDGv3 + MMFF94s) with confirmed non-zero 3D coordinates.
4. **`ablation_results.csv`**: Empirical ablation benchmark ($N=50$ per model, Seed 42) across Random, GRU, and C-VAE.
5. **`linker_length_scan.csv`**: MMFF94s unconstrained conformer ensemble trajectory scan across 6–22 heavy atoms.
6. **`literature_benchmark_comparison.csv`**: Head-to-head comparison against dBET6, ARV-825, and dBET1.
7. **`pipeline_execution_metrics.json`**: Independent filter rates and sequential funnel attrition counts.
8. **`test_results.txt`**: 15-test pytest execution log (**100% pass rate**).
9. **Publication-Quality Figures (300 DPI):**
   * `training_curves.png`: C-VAE loss trajectories and Cosine Annealing schedule.
   * `ablation_comparison.png`: Architectural benchmark across validity, uniqueness, bRo5, and productive fraction.
   * `linker_length_analysis.png`: Conformer span, steric clash penalty, and interface contact SASA across lengths.
   * `lead_property_comparison.png`: 6-panel property comparison against literature references.
   * `pipeline_summary.png`: Optimization funnel and chemical scaffold diversity.

---

## 💻 Reproduction Instructions

```bash
# 1. Run Automated Unit Test Suite (15 Tests)
python -m pytest -v tests/test_pipeline.py

# 2. Execute End-to-End Pipeline & Benchmarks
python run_tpd_pipeline.py

# 3. Regenerate All 5 Publication Figures
python src/utils/generate_defense_plots.py

# 4. Launch Interactive Web Demo Interface
python web_app/server.py
```
Or open `web_app/index.html` directly in your browser.

---

## 🔬 Citations & Ground-Truth References
1. **PDB 6BOY:** Nowak, R. P. et al. *Plasticity in binding confers selectivity in ligand-induced protein degradation.* Nat. Chem. Biol. **14**, 706–714 (2018).
2. **PDB 5T35:** Gadd, M. S. et al. *Structural basis for PROTAC-mediated ternary complex formation and degradation.* Nat. Chem. Biol. **13**, 514–521 (2017).
3. **Ertl SAScore:** Ertl, P. & Schuffenhauer, A. *Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions.* J. Cheminf. **1**, 8 (2009).
4. **dBET6 Benchmark:** Nowak, R. P. et al. Nat. Chem. Biol. **14**, 706–714 (2018).
5. **ARV-825 Benchmark:** Lu, J. et al. Blood **126**, 2197 (2015).
