# AI-Driven Targeted Protein Degradation (TPD) Lead Compound Generation Pipeline for Acute Myeloid Leukemia (AML) Targets via Generative PROTAC Linker Optimization

**Authors / Collaborators:** Madhav Yalamarthi and Hareesh  
**Project Scope:** Final Year Project / Publication-Grade Computational Chemical Biology & Generative AI  

---

## 📌 Project Overview
This repository contains the end-to-end computational pipeline for the automated de novo generative design, multi-objective optimization, and ternary complex modeling of heterobifunctional Proteolysis Targeting Chimeras (PROTACs) targeting **Bromodomain-containing Protein 4 (BRD4)** paired with the **Cereblon (CRBN)** E3 ubiquitin ligase in Acute Myeloid Leukemia (AML).

```
====================================================================================================
                        PROTAC TRIPARTITE SYSTEM ARCHITECTURE
====================================================================================================
   [ BRD4 BD1 Target ]               [ GENERATIVE LINKER ]             [ Cereblon / CRBN ]
   (+)-JQ1 Carboxamide      <===>   C-VAE Sampled Bridge       <===>   4-NH-Pomalidomide
   (PDB: 5T35 / BD1 Pocket)         (6-22 atoms: Triazole/PEG)         (Tri-Trp Hydrophobic Pocket)
====================================================================================================
```

---

## 🚀 Key Modules
* **Module 1 (`src/data_curation.py` & `data/`):**
  * `data/chembl_brd4_bioactivities.csv`: 16+ curated sub-100 nM BRD4 inhibitors with experimental IC50/Ki/Kd values, pChEMBL, and PubMed references.
  * `data/protac_db_benchmark.csv`: Gold-standard benchmark degraders (dBET1, dBET6, ARV-825, ARV-771, MZ1) from PROTAC-DB.
  * `data/train_linkers.csv` & `val_linkers.csv`: Modular chemical grammar datasets.
* **Module 2 (`src/models/` & `logs/`):**
  * `linker_generator.py`: PyTorch Conditional Variational Autoencoder (C-VAE) with bidirectional GRU decoder.
  * `train_cvae.py`: Full PyTorch training script with learning rate scheduler and checkpoint saving (`checkpoints/cvae_linker_model.pt`).
  * `logs/training_history.csv`: Empirical 25-epoch training loss, reconstruction loss, and validation loss records.
  * `scoring_engine.py`: Multi-objective optimization (SAScore, QED, bRo5 guidelines, PAINS removal).
* **Module 3 (`src/docking/`):**
  * `ternary_complex.py`: 3D conformer relaxation (ETKDGv3 + MMFF94s), pocket distance matching (PDB 5T35), $\Delta\text{SASA}$ contact area, and cooperativity ($\alpha$).
  * `ternary_kinetics.py`: 3-body mass-action equilibrium and bell-shaped hook effect simulator.
* **Module 4 (`run_tpd_pipeline.py` & `experiments/`):**
  * `experiments/benchmark_experiments.py`: Generative model ablation studies, linker length trajectory scans (6-22 atoms), and literature benchmark comparisons.
  * Exported CSVs: `lead_protac_candidates.csv`, `ablation_study_results.csv`, `linker_length_scan_results.csv`, `literature_benchmark_comparison.csv`.
* **Module 5 (`web_app/` & `REPORT.md`):**
  * `web_app/index.html`: Fully interactive web interface demo featuring live data exploration, training curves, interactive PROTAC generator studio, and hook effect simulator.
  * `REPORT.md`: Comprehensive, conference-ready manuscript authored by Madhav Yalamarthi and Hareesh.

---

## 🌐 Interactive Web Interface Demo

Launch the local interactive web interface to explore datasets, inspect loss curves, and dynamically simulate PROTAC generation:
```bash
python web_app/server.py
```
Or simply double-click and open `web_app/index.html` directly in your web browser!

---

## 💻 How to Run the Pipeline via Terminal

Execute the master script in your terminal:
```bash
python run_tpd_pipeline.py
```

### Execution Flow:
1. **Initializes Tripartite System:** Loads JQ1-acid and 4-amino-pomalidomide anchor definitions.
2. **Data Curation:** Queries ChEMBL for sub-100 nM BRD4 inhibitors and standardizes chemistry.
3. **Generative Modeling:** Trains/calibrates C-VAE and generates diverse linker fragments.
4. **Multi-Objective Filtering:** Computes SAScore, QED, bRo5, and eliminates toxicophores/PAINS.
5. **Ternary Biophysics:** Calculates 3D low-energy conformations, buried interface area ($\Delta\text{SASA}$), steric clashes, and thermodynamic cooperativity ($\alpha$).
6. **Outputs Lead Compounds:** Prints an ASCII summary table to the console and exports `lead_protac_candidates.csv`.

---

## 📊 Summary of Top-2 Prioritized Leads for AML

### Lead 1: `PROTAC-AML-001`
* **Architecture:** `[JQ1-Acid] -- [(CH2)2 - (1,2,3-triazole) - CH2 - O - (CH2)2 - C(=O)NH] -- [4-NH-Pomalidomide]`
* **Linker Type:** 1,2,3-Triazole Rigid Linker (14 heavy atoms)
* **MW / cLogP / TPSA:** $915.43\text{ Da}$ / $4.12$ / $185.34\text{ \AA}^2$
* **Synthesizability (SAScore):** $2.92$ (Accessible Ertl score)
* **Ternary Cooperativity ($\alpha$):** **$3.65$** (High positive synergy; avoids hook effect)
* **Ternary Free Energy ($\Delta G_{\text{ternary}}$):** **$-20.65\text{ kcal/mol}$** (Apparent $K_d \approx 0.12\text{ nM}$)
* **Inter-Protein Contact Area ($\Delta\text{SASA}$):** $542.0\text{ \AA}^2$ (Zero steric clashes)

### Lead 2: `PROTAC-AML-002`
* **Architecture:** `[JQ1-Acid] -- [(CH2)2 - C#C - CH2 - O - (CH2)2 - C(=O)NH] -- [4-NH-Pomalidomide]`
* **Linker Type:** Alkynyl-PEG Semi-Rigid Linker (13 heavy atoms)
* **MW / cLogP / TPSA:** $898.38\text{ Da}$ / $3.94$ / $179.22\text{ \AA}^2$
* **Synthesizability (SAScore):** $2.85$
* **Ternary Cooperativity ($\alpha$):** **$3.22$**
* **Ternary Free Energy ($\Delta G_{\text{ternary}}$):** **$-20.35\text{ kcal/mol}$** (Apparent $K_d \approx 0.21\text{ nM}$)
* **Inter-Protein Contact Area ($\Delta\text{SASA}$):** $510.5\text{ \AA}^2$ (Zero steric clashes)

---

## 📁 Repository Structure
```
d:\S7\AI in Drug Design\Project\
├── data/
│   └── curated_aml_targets.csv      # Standardized ChEMBL sub-100 nM bioactivities
├── src/
│   ├── __init__.py
│   ├── data_curation.py             # Module 1: Data extraction, standardization, exit vectors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── linker_generator.py      # Module 2: PyTorch C-VAE generative model & chemical assembler
│   │   └── scoring_engine.py        # Module 2: Multi-objective filter (SAScore, QED, bRo5, PAINS)
│   ├── docking/
│   │   ├── __init__.py
│   │   └── ternary_complex.py       # Module 3: 3D Conformer relaxation, clash & cooperativity engine
│   └── utils/
│       ├── __init__.py
│       └── sascorer.py              # Ertl Synthetic Accessibility Score implementation
├── run_tpd_pipeline.py              # Module 4: Master runnable terminal execution demo
├── lead_protac_candidates.csv       # Exported top-5 synthesis-ready lead candidates
├── requirements.txt                 # Python environment requirements
├── REPORT.md                        # Module 5: Conference-ready scientific report
└── README.md                        # Documentation and project manual
```

---

## 🔬 Citations & References
1. Ertl, P. & Schuffenhauer, A. *Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions.* J. Cheminf. **1**, 8 (2009).
2. Nowak, R. P. et al. *Plasticity in binding confers selectivity in ligand-induced protein degradation.* Nat. Chem. Biol. **14**, 706–714 (2018) [PDB 5T35].
3. Gadd, M. S. et al. *Structural basis for PROTAC-mediated ternary complex formation and degradation.* Nat. Chem. Biol. **13**, 514–521 (2017).
4. Winter, G. E. et al. *Phthalimide conjugations for targeted protein degradation.* Science **348**, 1376–1381 (2015).
