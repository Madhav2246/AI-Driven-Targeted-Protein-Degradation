# RESEARCH-GRADE REPRODUCIBILITY AUDIT REPORT
## AI-Driven Targeted Protein Degradation (TPD) Pipeline for Acute Myeloid Leukemia (AML)
### Project: Generative PROTAC Linker Optimization (BRD4 BD1/BD2 – Cereblon Degradation System)
**Authors / Investigators:** Madhav Yalamarthi and Hareesh  
**Audit Role:** Senior Computational Chemist, Cheminformatics Engineer & Generative AI Researcher  
**Date:** November 2026  
**Status:** Audit Completed; Remediation Plan Active  

---

## 1. EXECUTIVE SUMMARY

An exhaustive, line-by-line research and engineering audit was performed across all modules, datasets, model checkpoints, simulation scripts, and reported artifacts within the repository. The objective of this audit is to elevate the project from an initial prototype to a publication-grade, fully reproducible computational chemical biology framework ready for rigorous academic defense.

The core algorithmic design—combining conditional generative modeling for fragment inpainting with biophysical scoring and Beyond-Rule-of-5 (bRo5) filtering—is chemically sound. However, the audit uncovered **critical discrepancies**, **disconnected checkpoints**, **hardcoded empirical experiment returns**, and **incongruent PDB crystallographic structural references** that would undermine technical credibility if defended in an academic thesis or peer review.

This document details the 10 specific audit items mandated for inspection, classifies each by severity, identifies the exact file paths and line numbers affected, explains the root causes, and outlines the required surgical remediations.

---

## 2. SYSTEMATIC INVESTIGATION OF AUDIT ITEMS

```
+==================================================================================================+
|                                      AUDIT SEVERITY SUMMARY                                      |
+==================================================================================================+
| CRITICAL ISSUES (Immediate Technical Disqualification if Defended):                              |
|   - Item 1: Pipeline disconnected from trained C-VAE checkpoint (trained on 8-sample toy prior) |
|   - Item 5: Structural ternary scoring claimed as docking/FreeSASA is heuristic distance proxy   |
|   - Item 6: PDB reference mismatch (5T35 is BRD4 BD2-VHL-MZ1, NOT BRD4 BD1-CRBN-dBET6 [6BOY])   |
|   - Item 8: Ablation study metrics in benchmark_experiments.py are hardcoded dictionary literals |
|   - Item 9: Linker length scan uses linear algebraic heuristic rather than 3D conformer modeling |
+--------------------------------------------------------------------------------------------------+
| MAJOR ISSUES (Scientific Rigor & Reproducibility Compromised):                                   |
|   - Item 2: Small curated dataset size (20 train, 5 val, 5 test linkers)                        |
|   - Item 3: Assembler prepends triazole scaffold (cand6) uniformly across all generated linkers  |
|   - Item 4: Hardcoded ring prefixing overrides sampled diverse chemotypes (alkyl/PEG/alkyne)     |
|   - Item 7: Discrepancy between lead_protac_candidates.csv (AML-015) and REPORT.md (AML-001)   |
+--------------------------------------------------------------------------------------------------+
| MODERATE ISSUES (Quality Assurance & Engineering Robustness):                                    |
|   - Item 10: Test suite passes but lacks tests for checkpoint loading, SDF export & invalid SMILES|
+==================================================================================================+
```

---

### Audit Item 1: C-VAE Model Checkpoint Utilization in Master Pipeline
* **Severity:** **CRITICAL**
* **Affected Files:** `run_tpd_pipeline.py` (lines 92–104), `src/models/linker_generator.py` (lines 288–350)
* **Finding:** 
  The project contains a fully trained PyTorch checkpoint `checkpoints/cvae_linker_model.pt` (3.3 MB, trained over 25 epochs with best validation loss of 1.8342 at epoch 11). However, the master execution script `run_tpd_pipeline.py` initializes `GenerativeProtacPipeline`, which in its `__init__` executes:
  ```python
  self.model = ConditionalLinkerVAE()
  self.assembler = ProtacChemicalAssembler(warhead_smiles, e3_smiles)
  self._pretrain_synthetic_prior()
  ```
  `_pretrain_synthetic_prior()` trains the model *in-memory* for 12 epochs over only 8 hardcoded toy token sequences (`prior_sequences`), completely discarding `checkpoints/cvae_linker_model.pt`.
* **Impact:** 
  The pipeline never evaluated the real trained neural network during candidate generation.
* **Remediation:**
  Refactor `GenerativeProtacPipeline` to accept `checkpoint_path: Optional[str] = "checkpoints/cvae_linker_model.pt"`. Implement explicit checkpoint integrity checks: verify `model_state_dict` keys, ensure vocabulary alignment, load weights, and log checkpoint metadata (epoch, validation loss). Fall back to training only if the checkpoint is absent.

---

### Audit Item 2: Training and Validation Dataset Quality, Size, and Diversity
* **Severity:** **MAJOR**
* **Affected Files:** `data/train_linkers.csv`, `data/val_linkers.csv`, `data/test_linkers.csv`
* **Finding:**
  The training set contains 20 rows, validation contains 5 rows, and test contains 5 rows (total 30 linkers). While these entries are modular representations of standard degrader linkers (e.g., PEG-2 through PEG-5, alkyl C4-C12, alkynyl spacers, 1,2,3-triazoles, and piperazine bridges), the dataset size is small for deep generative training.
* **Impact:**
  With 20 training examples and batch size 4, an epoch consists of 5 gradient updates. While sufficient for proof-of-concept token sequence calibration, this constraint must be honestly acknowledged in the thesis rather than claiming training on thousands of PROTAC linkers.
* **Remediation:**
  Document the exact dataset dimensions, source provenance (curated from PROTAC-DB and ChEMBL literature standards), and split ratios. Implement data verification tests to guarantee all training linkers parse without RDKit sanitization errors.

---

### Audit Item 3: C-VAE Architecture, Conditioning Mechanism, and Decoding
* **Severity:** **MAJOR**
* **Affected Files:** `src/models/linker_generator.py` (lines 73–205)
* **Finding:**
  1. The encoder `AnchorConditioningEncoder` projects two 128-bit Morgan fingerprints (Warhead and E3) plus a normalized target length scalar into a 64-dimensional condition vector $c$.
  2. The recognition network uses a bidirectional GRU ($hidden\_dim=96$) mapping into latent space $z \in \mathbb{R}^{32}$.
  3. The decoder autoregressively consumes $[embed(x_t), z, c]$ to predict token logits.
  4. In `sample_linker()`, if the decoder generates an empty sequence, it fell back to a hardcoded `random.choice`.
  5. In `ProtacChemicalAssembler._format_chemically_sound_linkers`, candidate template `cand6 = f"CCc7cn(nn7){clean_body}COCCC(=O)N"` was placed as the first option evaluated. Because `cand6` almost always sanitizes successfully, every generated PROTAC was assembled with a 1,2,3-triazole ring prepended, suppressing aliphatic and pure PEG designs.
* **Impact:**
  This artificial bias reduced structural diversity in generated candidates, regardless of the tokens sampled by the neural decoder.
* **Remediation:**
  Dynamically format linker assemblies according to the dominant chemotype present in the sampled token sequence. If tokens are purely aliphatic (`CC`, `CCC`), assemble alkyl chains; if PEG (`CCOCC`), assemble ether chains; if alkynyl (`C#C`), assemble rigid alkynes; if heterocyclic, include triazole or piperazine.

---

### Audit Item 4: Chemical Assembly and Attachment Chemistry
* **Severity:** **MAJOR**
* **Affected Files:** `src/models/linker_generator.py` (lines 207–286), `src/data_curation.py`
* **Finding:**
  1. **Warhead:** (+)-JQ1 carboxylic acid derivative (`Cc1sc2c(c1C)c(-c3ccc(Cl)cc3)nc(CC(=O)O)n2`). The attachment handle is the C4-carboxamide linkage, matching the crystallographic binding mode in dBET6 (PDB 6BOY).
  2. **E3 Recruiter:** 4-amino-pomalidomide (`O=C1c2c(cccc2N)C(=O)N1C1CCC(=O)NC1=O`). The attachment handle is the phthaloyl 4-amino nitrogen, preserving the glutarimide ring critical for Cereblon tri-tryptophan pocket recognition.
  3. **Attachment Logic:** The assembly creates amide linkages at both ends: `JQ1-C(=O)NH--Linker--NH-Pomalidomide` (or `JQ1-C(=O)NH--Linker--C(=O)NH-Pomalidomide`).
* **Impact:**
  The connectivity is biologically and chemically valid. However, cross-fragment ring index numbering previously produced macrocycle collisions (resolved by indexing JQ1 as rings 1,2,3, pomalidomide as 4,5,6, and linkers as 7).
* **Remediation:**
  Verify that all 2D SMILES and 3D SDF outputs possess unambiguous isomeric chemistry, correct valency, no radical centers, and exact molecular weights corresponding to the formula.

---

### Audit Item 5: Ternary Complex Modeling & Structural Evaluation
* **Severity:** **CRITICAL**
* **Affected Files:** `src/docking/ternary_complex.py` (lines 106–236)
* **Finding:**
  The module docstrings and report text suggest that the script performs automated all-atom macromolecular docking (e.g., AutoDock Vina or Rosetta) and FreeSASA calculation on protein coordinates. In reality:
  1. It performs RDKit ETKDGv3 conformer generation on the ligand alone.
  2. It estimates the end-to-end exit-vector distance from the ligand conformer.
  3. It passes this distance to `evaluate_steric_and_interface()`, which calculates $\Delta\text{SASA}$ and steric clash penalties using a **piecewise empirical heuristic formula** parameterized around an optimal pocket span of $13.2\text{ \AA}$ (now calibrated to $14.0\text{ \AA}$ for 6BOY).
  4. It calculates thermodynamic cooperativity $\alpha$ using:
     $$\alpha = \exp\left(-\frac{\Delta\Delta G}{RT}\right), \quad \Delta\Delta G = -(0.008 \cdot \Delta\text{SASA}) + (0.25 \cdot E_{\text{strain}}) + E_{\text{clash}}$$
* **Impact:**
  Presenting an empirical distance-geometry heuristic as "coordinate-based ternary docking with protein FreeSASA" is scientifically inaccurate and would fail an academic defense.
* **Remediation:**
  Explicitly and transparently document the structural evaluation as an **in silico distance geometry and conformational strain energy evaluation with empirical analytical scoring**. This is a legitimate, defensible cheminformatics proxy for high-throughput screening, provided it is honestly described.

---

### Audit Item 6: PDB Structural Reference Mapping
* **Severity:** **CRITICAL**
* **Affected Files:** `src/docking/ternary_complex.py` (lines 4, 24–30), `run_tpd_pipeline.py` (line 130), `REPORT.md`, `README.md`
* **Finding:**
  The codebase, reports, and comments repeatedly identified **PDB 5T35** as the reference structure for "dBET6 bound to BRD4 BD1 and CRBN".
  * **Fact Check:**
    * **PDB 5T35** is the crystal structure of **BRD4 BD2 in complex with MZ1 and VHL–ElonginB–ElonginC** (Gadd et al., *Nat Chem Biol* 2017). MZ1 recruits the VHL ligase, NOT Cereblon!
    * **PDB 6BOY** is the actual crystal structure of **BRD4 BD1 in complex with dBET6 and DDB1–CRBN** (Nowak et al., *Nat Chem Biol* 2018). dBET6 recruits Cereblon via pomalidomide!
* **Impact:**
  Claiming 5T35 represents a Cereblon-dBET6 ternary complex is a major factual error that any structural biologist or thesis examiner would immediately reject.
* **Remediation:**
  1. Correct the primary reference crystal structure across all files to **PDB 6BOY** (BRD4 BD1–CRBN–DDB1–dBET6).
  2. Relegate **PDB 5T35** to its legitimate role: a cross-ligase comparator representing the VHL-mediated system (BRD4 BD2–VHL–MZ1).
  3. Update reference parameters: optimal C4-to-C4 exit-vector distance in 6BOY is $14.2 \pm 2.0\text{ \AA}$.

---

### Audit Item 7: Consistency of Reported Lead Candidates
* **Severity:** **MAJOR**
* **Affected Files:** `lead_protac_candidates.csv`, `REPORT.md`, `AML_PROTAC_Conference_Report.md`, `web_app/index.html`
* **Finding:**
  * In `REPORT.md` and `AML_PROTAC_Conference_Report.md`, the narrative highlights `PROTAC-AML-001` and `PROTAC-AML-002` as top leads.
  * In the exported `lead_protac_candidates.csv`, candidate ranking places `PROTAC-AML-015` at rank #1 ($\text{integrated score} = 0.486$, $\alpha = 8.29$, $\Delta G = -20.5\text{ kcal/mol}$), followed by `PROTAC-AML-011` ($\text{score} = 0.380$), while `PROTAC-AML-001` was not among the top-ranked leads.
* **Impact:**
  Readers checking the exported CSV against the discussion in the report encounter mismatched candidate identifiers, scaffolds, and properties.
* **Remediation:**
  Synchronize all documentation and exported CSV/SDF files. The top-ranked leads in `verified_lead_candidates.csv` and `verified_lead_structures.sdf` must match the tables and discussion in `THESIS_RESULTS_AUDIT.md`, `REPORT.md`, and `README.md`.

---

### Audit Item 8: Benchmark Ablation Study Computation
* **Severity:** **CRITICAL**
* **Affected Files:** `experiments/benchmark_experiments.py` (lines 28–88)
* **Finding:**
  The function `run_experiment_a_ablation()` does not execute any model sampling, chemical filtering, or evaluation. Instead, it directly returns a hardcoded list of dictionaries containing pre-populated numbers:
  ```python
  results = [
      {"model_variant": "Random Combinatorial Baseline", "chemical_validity_pct": 54.2, ...},
      {"model_variant": "Unconditional Autoregressive GRU", "chemical_validity_pct": 82.4, ...},
      {"model_variant": "Proposed Dual-Anchor C-VAE (Ours)", "chemical_validity_pct": 96.8, ...}
  ]
  ```
* **Impact:**
  These numbers were static fabrications rather than experimental results. If examined during a defense, this would constitute academic misconduct.
* **Remediation:**
  Implement a genuine experimental evaluation loop:
  1. Generate $N$ samples from a random token sampler (Random Baseline).
  2. Generate $N$ samples from an unconditional GRU (or untrained model).
  3. Generate $N$ samples from the trained Dual-Anchor C-VAE loaded from `checkpoints/cvae_linker_model.pt`.
  4. Compute real chemical validity via RDKit, uniqueness, novelty against `data/train_linkers.csv`, bRo5 compliance, PAINS pass rate, mean SAScore, mean QED, and mean cooperativity $\alpha$.
  5. Write the computed values to `experiments/ablation_study_results.csv`.

---

### Audit Item 9: Linker Length Trajectory Scan Computation
* **Severity:** **CRITICAL**
* **Affected Files:** `experiments/benchmark_experiments.py` (lines 90–143)
* **Finding:**
  `run_experiment_b_length_scan()` computes the end-to-end distance using an algebraic linear equation:
  ```python
  approx_dist = min(22.0, max(7.5, 6.5 + (l * 0.58)))
  ```
  It does not construct linkers of varying lengths, embed 3D conformers, or optimize force fields.
* **Impact:**
  The length-scan results were derived from an idealized mathematical curve rather than physical molecular conformations.
* **Remediation:**
  Implement actual conformer modeling for linkers spanning lengths 6 to 22 atoms:
  1. Assemble representative molecules across each length increment.
  2. Embed conformers using RDKit ETKDGv3 and minimize with MMFF94s.
  3. Measure the actual 3D inter-anchor Euclidean distance and conformer strain energy.
  4. Calculate interface $\Delta\text{SASA}$ and clash penalties based on actual coordinates.
  5. Save the computed trajectory to `experiments/linker_length_scan_results.csv`.

---

### Audit Item 10: Automated Test Suite Coverage
* **Severity:** **MODERATE**
* **Affected Files:** `tests/test_pipeline.py` (lines 1–89)
* **Finding:**
  `test_pipeline.py` contains 6 tests that pass. However:
  1. It does not test checkpoint loading from `checkpoints/cvae_linker_model.pt`.
  2. It does not verify checkpoint dictionary keys (`model_state_dict`, `vocab`, `epoch`).
  3. It does not test resilience to malformed or invalid SMILES strings.
  4. It does not verify that exported SDF files contain valid 3D conformer coordinates.
  5. It does not verify descriptor consistency between raw molecules and candidate CSVs.
* **Impact:**
  A passing test suite provided false confidence because it failed to cover the integration touchpoints where failures or discrepancies occurred.
* **Remediation:**
  Expand `tests/test_pipeline.py` to 12+ rigorous unit and integration tests covering checkpoint loading, inference tensor shapes, dynamic chemical assembly, invalid SMILES handling, descriptor calculation fidelity, and 6BOY crystallographic parameters.

---

## 3. SEVERITY MATRIX & REMEDIATION PLAN

| Item # | Inspection Target | Severity | Primary Affected File | Remediation Action |
|:---:|:---|:---:|:---|:---|
| **1** | C-VAE Checkpoint Usage | **CRITICAL** | `src/models/linker_generator.py` | Load `checkpoints/cvae_linker_model.pt` with state dict verification |
| **2** | Dataset Size & Diversity | **MAJOR** | `data/train_linkers.csv` | Document exact dataset size (20/5/5) and split methodology honestly |
| **3** | C-VAE Inference & Assembly | **MAJOR** | `src/models/linker_generator.py` | Eliminate universal `cand6` triazole override; assemble chemotypes dynamically |
| **4** | Attachment Vector Chemistry | **MAJOR** | `src/models/linker_generator.py` | Validate JQ1 C4-amide and Pomalidomide 4-amino connectivity |
| **5** | Ternary Structural Scoring | **CRITICAL** | `src/docking/ternary_complex.py` | Label transparently as distance-geometry analytical scoring |
| **6** | PDB Reference Mapping | **CRITICAL** | `src/docking/ternary_complex.py` | Correct PDB to 6BOY (BRD4 BD1-CRBN-dBET6); keep 5T35 as VHL comparator |
| **7** | Lead Compound Consistency | **MAJOR** | `lead_protac_candidates.csv` | Harmonize lead IDs, SMILES, and SDFs across all files and reports |
| **8** | Ablation Study Realism | **CRITICAL** | `experiments/benchmark_experiments.py` | Replace hardcoded returns with genuine multi-model evaluation loops |
| **9** | Length Scan Realism | **CRITICAL** | `experiments/benchmark_experiments.py` | Replace linear heuristic with RDKit 3D conformer generation |
| **10** | Automated Test Coverage | **MODERATE** | `tests/test_pipeline.py` | Expand to 12+ comprehensive unit and integration tests |

---

## 4. CONCLUSION

All 10 audit items have been systematically identified, characterized, and mapped to specific code locations. The planned remediations preserve the existing project architecture, require no ungrounded data fabrication, and will produce a fully defensible, publication-grade computational thesis repository.

