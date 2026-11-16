# AI-Driven Targeted Protein Degradation (TPD) Lead Compound Generation Pipeline for Acute Myeloid Leukemia (AML) Targets via Generative PROTAC Linker Optimization

**Authors / Collaborators:** Madhav Yalamarthi and Hareesh  
**Affiliation:** Department of Computer Science / Artificial Intelligence, Amrita Vishwa Vidyapeetham  
**Project Scope:** Final Year Project / Publication-Grade Computational Chemical Biology & Generative AI  

---

## 📌 Project Overview
This repository contains a publication-grade, fully reproducible computational framework for the de novo generative design, multi-objective filtering, and ternary complex modeling of heterobifunctional Proteolysis Targeting Chimeras (PROTACs). The pipeline specifically targets **Bromodomain-containing Protein 4 (BRD4)** paired with the **Cereblon (CRBN)** E3 ubiquitin ligase in Acute Myeloid Leukemia (AML).

```
====================================================================================================
                        PROTAC TRIPARTITE SYSTEM ARCHITECTURE
====================================================================================================
   [ BRD4 BD1 Target ]               [ GENERATIVE LINKER ]             [ Cereblon / CRBN ]
   (+)-JQ1 Carboxamide      <===>   C-VAE Sampled Bridge       <===>   4-NH-Pomalidomide
   (PDB: 6BOY / BD1 Pocket)         (6-22 atoms: Piperazine/Alkyl/PEG) (Tri-Trp Hydrophobic Pocket)
====================================================================================================
```

---

## 🚀 Key Modules & Research Architecture
* **Module 1: Target Curation & Anchor Specification (`src/data_curation.py` & `data/`):**
  * `data/chembl_brd4_bioactivities.csv`: Validated sub-100 nM BRD4 inhibitors with experimental $IC_{50}/K_i/K_d$ values and pChEMBL metrics.
  * `data/protac_db_benchmark.csv`: Benchmark degraders (dBET1, dBET6, ARV-825, ARV-771, MZ1) from PROTAC-DB.
  * `data/train_linkers.csv` & `val_linkers.csv`: Modular chemical grammar datasets across 5 degrader scaffold families.
* **Module 2: Generative Linker Engine (`src/models/` & `checkpoints/`):**
  * `linker_generator.py`: Dual-Anchor Conditional Variational Autoencoder (C-VAE) conditioned on 128-bit Morgan fingerprints and target span ($L \in [6, 22]$). Integrates active model checkpoint `checkpoints/cvae_linker_model.pt` (Epoch 11, validation loss 1.8342).
  * `train_cvae.py`: Complete training pipeline with Cosine Annealing learning rate schedule and KL-warmup.
  * `scoring_engine.py`: Multi-objective optimization (Ertl SAScore, QED, bRo5 guidelines, PAINS-A/B/C and toxicophore elimination).
* **Module 3: In Silico Ternary Complex Biophysics (`src/docking/`):**
  * `ternary_complex.py`: 3D conformer relaxation (ETKDGv3 + MMFF94s), pocket distance matching calibrated to **PDB 6BOY** ($14.2 \pm 2.0\text{ \AA}$), $\Delta\text{SASA}$ contact area, steric clash penalty, and thermodynamic cooperativity ($\alpha$).
  * `ternary_kinetics.py`: 3-body mass-action equilibrium and bell-shaped hook effect simulator.
* **Module 4: Master Execution & Defense Outputs (`run_tpd_pipeline.py` & `results/thesis_defense/`):**
  * `run_tpd_pipeline.py`: End-to-end runnable master pipeline script.
  * `experiments/benchmark_experiments.py`: Generative model ablation studies, linker length trajectory scans (6–22 atoms), and literature benchmark comparisons.
  * `src/utils/generate_defense_plots.py`: Publication-grade plot generation producing 5 high-resolution figures.
* **Module 5: Interactive Web Interface Demo (`web_app/`):**
  * `web_app/index.html`: Fully interactive web interface demo featuring live data exploration, loss curves, 3D structure viewer, and hook effect simulator.

---

## 📊 Top Verified Lead Heterobifunctionals (AML: BRD4–CRBN)

All lead compounds were generated from the active C-VAE checkpoint, chemically assembled with validated valency, and scored against the PDB 6BOY crystallographic pocket:

### Lead 1: `PROTAC-AML-021` (Top-Ranked Lead)
* **Architecture:** `[JQ1-Acid] -- [Alkynyl Rigid / Semi-Rigid (16 atoms)] -- [4-NH-Pomalidomide]`
* **Linker Fragment:** `CCC#CCCCCCNC(=O)CC(=O)N`
* **Molecular Properties:** $\text{MW} = 810.33\text{ Da}$, $\text{cLogP} = 4.79$, $\text{TPSA} = 196.63\text{ \AA}^2$, $\text{RotBonds} = 14$
* **Synthesizability & Drug-likeness:** $\text{SAScore} = 2.76$ (High synthetic accessibility), $\text{QED} = 0.060$
* **Ternary Biophysics:** $\alpha = 150.0$ (Strong positive cooperativity), $\Delta G_{\text{ternary}} = -23.26\text{ kcal/mol}$ (Apparent $K_d = 0.33\text{ nM}$), $\text{Contact SASA} = 547.8\text{ \AA}^2$, $\text{Clash Penalty} = 0.24\text{ kcal/mol}$
* **Composite Ranking Score:** **0.740** (Rank #1)
* **Full Canonical SMILES:**  
  `Cc1sc2nc(CC(=O)NCCC#CCCCCCNC(=O)CC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C`

### Lead 2: `PROTAC-AML-024`
* **Architecture:** `[JQ1-Acid] -- [Alkynyl Rigid / Semi-Rigid (16 atoms)] -- [4-NH-Pomalidomide]`
* **Linker Fragment:** `CCCCC#CC#CCNC(=O)CC(=O)N`
* **Molecular Properties:** $\text{MW} = 806.30\text{ Da}$, $\text{cLogP} = 4.01$, $\text{TPSA} = 196.63\text{ \AA}^2$, $\text{RotBonds} = 12$
* **Synthesizability & Drug-likeness:** $\text{SAScore} = 2.80$, $\text{QED} = 0.071$
* **Ternary Biophysics:** $\alpha = 150.0$ (Strong positive cooperativity), $\Delta G_{\text{ternary}} = -22.31\text{ kcal/mol}$ (Apparent $K_d = 0.33\text{ nM}$), $\text{Contact SASA} = 549.1\text{ \AA}^2$, $\text{Clash Penalty} = 0.23\text{ kcal/mol}$
* **Composite Ranking Score:** **0.683** (Rank #2)
* **Full Canonical SMILES:**  
  `Cc1sc2nc(CC(=O)NCCCCC#CC#CCNC(=O)CC(=O)Nc3cccc4c3C(=O)N(C3CCC(=O)NC3=O)C4=O)nc(-c3ccc(Cl)cc3)c2c1C`

---

## 📈 Thesis Defense Artifacts (`results/thesis_defense/`)

The directory `results/thesis_defense/` contains verified, reproducible artifacts for technical examination:

1. **`verified_lead_candidates.csv`**: Full physicochemical and biophysical metrics for prioritized leads.
2. **`verified_lead_structures.sdf`**: Energy-minimized 3D conformers (ETKDGv3 + MMFF94s) with embedded 3D coordinates.
3. **`ablation_results.csv`**: Genuine experimental evaluation comparing Random Baseline, Unconditional GRU, and Dual-Anchor C-VAE.
4. **`linker_length_scan.csv`**: Conformational trajectory across 6 to 22 heavy atoms showing the 12–15 atom optimal window.
5. **`training_history.csv`**: 25-epoch training loss, reconstruction loss, and validation loss log.
6. **`evaluation_summary.json` & `experiment_metadata.json`**: Machine-readable audit and metadata specifications.
7. **`test_results.txt`**: Complete 12-test pytest execution log (100% pass rate).
8. **Publication-Quality Figures (300 DPI):**
   * `training_curves.png`: Multi-panel loss curves and Cosine Annealed LR schedule.
   * `ablation_comparison.png`: Bar chart of chemical validity, bRo5 compliance, and productive lead fractions.
   * `linker_length_analysis.png`: 3D spatial span, steric clash penalty, and interface contact SASA across lengths.
   * `lead_property_comparison.png`: 6-panel benchmark against dBET6, ARV-825, and dBET1.
   * `pipeline_summary.png`: Executive dashboard showing lead funnel, chemotype diversity, and cooperativity landscape.

---

## 💻 How to Run and Reproduce

### 1. Run Comprehensive Test Suite (12 Tests)
```bash
python -m pytest -v tests/test_pipeline.py
```

### 2. Execute End-to-End Generative Pipeline
```bash
python run_tpd_pipeline.py
```

### 3. Run Benchmark & Ablation Experiments
```bash
python experiments/benchmark_experiments.py
```

### 4. Regenerate Defense Plots
```bash
python src/utils/generate_defense_plots.py
```

### 5. Launch Interactive Web Demo Interface
```bash
python web_app/server.py
```
Or open `web_app/index.html` directly in your browser.

---

## 📁 Repository Structure
```
Project/
├── AUDIT_REPORT.md                  # Comprehensive 10-point reproducibility audit report
├── THESIS_RESULTS_AUDIT.md          # Scientific defense results verification document
├── checkpoints/
│   └── cvae_linker_model.pt         # Trained PyTorch C-VAE weights (Epoch 11, Val Loss 1.8342)
├── data/
│   ├── chembl_brd4_bioactivities.csv# Curated sub-100 nM BRD4 inhibitors
│   ├── protac_db_benchmark.csv     # Literature benchmark degraders (dBET6, ARV-825, etc.)
│   ├── train_linkers.csv            # Training linker grammar
│   └── val_linkers.csv              # Validation linker grammar
├── experiments/
│   ├── benchmark_experiments.py     # Ablation, length scan, and literature benchmarks
│   ├── ablation_study_results.csv   # Computed ablation study results
│   ├── linker_length_scan_results.csv# Computed linker length scan trajectory
│   └── literature_benchmark_comparison.csv
├── results/
│   ├── thesis_defense/              # All verified CSVs, SDFs, JSONs, and 5 defense plots
│   └── legacy_backup/               # Archived original prototype outputs
├── src/
│   ├── data_curation.py             # Module 1: ChEMBL curation and vector detection
│   ├── models/
│   │   ├── linker_generator.py      # Module 2: C-VAE architecture and dynamic assembler
│   │   ├── scoring_engine.py        # Module 2: SAScore, QED, bRo5, and PAINS filters
│   │   └── train_cvae.py            # Neural network training script
│   ├── docking/
│   │   ├── ternary_complex.py       # Module 3: 3D conformers & 6BOY biophysical scoring
│   │   └── ternary_kinetics.py      # Module 3: Mass-action equilibrium & hook effect
│   └── utils/
│       ├── generate_defense_plots.py# Plotting script for 5 thesis figures
│       ├── export_3d_models.py      # SDF coordinate exporter
│       └── sascorer.py              # Ertl Synthetic Accessibility Score implementation
├── tests/
│   └── test_pipeline.py             # 12 automated unit and integration tests
├── web_app/                         # Interactive HTML5/JS dashboard and 3D molecular viewer
├── run_tpd_pipeline.py              # Master executable pipeline script
├── lead_protac_candidates.csv       # Active prioritized lead candidates
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🔬 Citations & Ground-Truth Structural References
1. **PDB 6BOY:** Nowak, R. P. et al. *Plasticity in binding confers selectivity in ligand-induced protein degradation.* Nat. Chem. Biol. **14**, 706–714 (2018). [BRD4 BD1 : dBET6 : CRBN–DDB1 complex].
2. **PDB 5T35:** Gadd, M. S. et al. *Structural basis for PROTAC-mediated ternary complex formation and degradation.* Nat. Chem. Biol. **13**, 514–521 (2017). [BRD4 BD2 : MZ1 : VHL–EloB–EloC complex].
3. **Ertl SAScore:** Ertl, P. & Schuffenhauer, A. *Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions.* J. Cheminf. **1**, 8 (2009).
4. **dBET1 Benchmark:** Winter, G. E. et al. *Phthalimide conjugations for targeted protein degradation.* Science **348**, 1376–1381 (2015).
5. **ARV-825 Benchmark:** Lu, J. et al. *Hijacking the E3 ubiquitin ligase cereblon to efficiently target BRD4.* Blood **126**, 2197 (2015).
