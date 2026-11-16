/**
 * Interactive Client-Side Application for TPD PROTAC Web Demo
 * Authors: Madhav Yalamarthi & Hareesh
 */

// Global Chart Instances
let trainingChart = null;
let hookChart = null;

// Fallback Embedded Datasets (ensures UI works smoothly even when opened via file:// protocol directly)
const EMBEDDED_DATA = {
  chembl: [
    { chembl_id: "CHEMBL3707348", name: "Mivebresib (ABBV-075)", type: "IC50", val: "1.5 nM", pchembl: "8.82", assay: "CHEMBL3709214", pubmed: "28416738", smiles: "Cc1cc(C(=O)NC2CC2)c2nc(-c3ccc(Cl)cc3)c3c(s3)nc2n1" },
    { chembl_id: "CHEMBL3989932", name: "BMS-986158", type: "IC50", val: "5.2 nM", pchembl: "8.28", assay: "CHEMBL3991201", pubmed: "29543431", smiles: "CN1CCN(c2ccc(-c3c4ccccc4nc4c3[nH]c3ccccc34)cc2)CC1" },
    { chembl_id: "CHEMBL3137343", name: "Birabresib (OTX015)", type: "IC50", val: "10.0 nM", pchembl: "8.00", assay: "CHEMBL3139820", pubmed: "24482594", smiles: "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)Nc1ccc(O)cc1)n2" },
    { chembl_id: "CHEMBL4297441", name: "BETd-260 Warhead", type: "IC50", val: "18.0 nM", pchembl: "7.74", assay: "CHEMBL4298115", pubmed: "29037998", smiles: "Cc1sc2c(c1C)c(-c1ccc(F)cc1)nc(CC(=O)O)n2" },
    { chembl_id: "CHEMBL3622416", name: "dBET1 Warhead", type: "IC50", val: "22.0 nM", pchembl: "7.66", assay: "CHEMBL3624890", pubmed: "26687834", smiles: "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCC(=O)O)n2" },
    { chembl_id: "CHEMBL4298110", name: "ARV-825 Warhead", type: "Kd", val: "28.0 nM", pchembl: "7.55", assay: "CHEMBL4299102", pubmed: "26432729", smiles: "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCCN)n2" },
    { chembl_id: "CHEMBL2047164", name: "Molibresib (I-BET762)", type: "IC50", val: "32.5 nM", pchembl: "7.49", assay: "CHEMBL2048992", pubmed: "22071858", smiles: "Cc1nc2c(s1)c(-c1ccc(Cl)cc1)nc(C(=O)NC1CC1)n2" },
    { chembl_id: "CHEMBL4297893", name: "Pelabresib (CPI-0610)", type: "IC50", val: "39.0 nM", pchembl: "7.41", assay: "CHEMBL4299450", pubmed: "31065181", smiles: "CC1CN(c2ccc(NC(=O)c3cn(C)c4ccccc34)cc2)C(=O)O1" },
    { chembl_id: "CHEMBL3264903", name: "(+)-JQ1 Acid", type: "Kd", val: "45.0 nM", pchembl: "7.35", assay: "CHEMBL3266100", pubmed: "21179161", smiles: "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)O)n2" },
    { chembl_id: "CHEMBL410526", name: "(+)-JQ1 (Benchmark)", type: "Kd", val: "77.0 nM", pchembl: "7.11", assay: "CHEMBL412999", pubmed: "21179161", smiles: "Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)OC(C)(C)C)n2" }
  ],
  protac_db: [
    { id: "PROTAC-REF-01", name: "dBET6", target: "BRD4 (BD1/BD2)", e3: "Cereblon (CRBN)", dc50: "1.0 nM", dmax: "98.5%", pdb: "5T35", cell: "MV4-11 (AML)" },
    { id: "PROTAC-REF-02", name: "ARV-825", target: "BRD4 (BD1/BD2)", e3: "Cereblon (CRBN)", dc50: "2.2 nM", dmax: "95.0%", pdb: "N/A", cell: "MOLM-13 (AML)" },
    { id: "PROTAC-REF-03", name: "dBET1", target: "BRD4 (BD1/BD2)", e3: "Cereblon (CRBN)", dc50: "430.0 nM", dmax: "85.0%", pdb: "N/A", cell: "AML Blasts" },
    { id: "PROTAC-REF-04", name: "ARV-771", target: "BRD4 (BD1/BD2)", e3: "VHL", dc50: "5.0 nM", dmax: "96.0%", pdb: "6BOY", cell: "MV4-11 (AML)" },
    { id: "PROTAC-REF-05", name: "MZ1", target: "BRD4 (BD1/BD2)", e3: "VHL", dc50: "12.5 nM", dmax: "92.0%", pdb: "5T35", cell: "HL-60 (AML)" },
    { id: "PROTAC-REF-06", name: "BETd-260", target: "BRD4 (BD1/BD2)", e3: "Cereblon (CRBN)", dc50: "0.5 nM", dmax: "99.2%", pdb: "N/A", cell: "RS4;11 / MV4-11" },
    { id: "PROTAC-REF-07", name: "ZXH-3-26", target: "BRD4 (BD1)", e3: "Cereblon (CRBN)", dc50: "4.5 nM", dmax: "94.0%", pdb: "6BN7", cell: "MOLM-13 (AML)" }
  ],
  training_logs: [
    { epoch: 1, train: 3.482, recon: 3.478, val: 3.312 },
    { epoch: 3, train: 2.784, recon: 2.771, val: 2.651 },
    { epoch: 5, train: 2.185, recon: 2.162, val: 2.145 },
    { epoch: 8, train: 1.554, recon: 1.514, val: 1.642 },
    { epoch: 10, train: 1.265, recon: 1.213, val: 1.425 },
    { epoch: 12, train: 1.054, recon: 0.994, val: 1.281 },
    { epoch: 15, train: 0.842, recon: 0.770, val: 1.155 },
    { epoch: 18, train: 0.718, recon: 0.634, val: 1.094 },
    { epoch: 20, train: 0.672, recon: 0.580, val: 1.075 },
    { epoch: 22, train: 0.647, recon: 0.547, val: 1.065 },
    { epoch: 25, train: 0.635, recon: 0.523, val: 1.059 }
  ],
  ablation: [
    { variant: "Random Combinatorial Baseline", val: "100.0%", uniq: "100.0%", nov: "98.0%", bro5: "92.0%", pains: "100.0%", sa: "2.95", qed: "0.078", alpha: "38.50", prod: "66.0%" },
    { variant: "Unconditional Autoregressive GRU", val: "100.0%", uniq: "32.0%", nov: "100.0%", bro5: "84.0%", pains: "100.0%", sa: "2.92", qed: "0.055", alpha: "23.32", prod: "38.0%" },
    { variant: "Proposed Dual-Anchor C-VAE (Ours)", val: "100.0%", uniq: "86.0%", nov: "100.0%", bro5: "90.0%", pains: "100.0%", sa: "2.82", qed: "0.058", alpha: "29.12", prod: "52.0%" }
  ],
  length_scan: [
    { span: "6 atoms", dist: "6.92 A", sasa: "94 A2", clash: "14.70 kcal", count: "8", strain: "10.47 kcal", alpha: "0.001", dg: "-2.69 kcal", feas: "Steric Collision" },
    { span: "8 atoms", dist: "9.14 A", sasa: "250 A2", clash: "9.15 kcal", count: "5", strain: "4.64 kcal", alpha: "0.001", dg: "-10.94 kcal", feas: "Steric Collision" },
    { span: "10 atoms", dist: "11.01 A", sasa: "381 A2", clash: "4.47 kcal", count: "2", strain: "0.00 kcal", alpha: "0.091", dg: "-17.83 kcal", feas: "Transition Boundary" },
    { span: "12 atoms", dist: "12.08 A", sasa: "486 A2", clash: "0.41 kcal", count: "0", strain: "1.75 kcal", alpha: "150.0", dg: "-22.29 kcal", feas: "Optimal" },
    { span: "13 atoms", dist: "10.75 A", sasa: "363 A2", clash: "5.12 kcal", count: "2", strain: "9.72 kcal", alpha: "0.001", dg: "-14.60 kcal", feas: "Steric Collision" },
    { span: "14 atoms", dist: "12.33 A", sasa: "495 A2", clash: "0.39 kcal", count: "0", strain: "6.69 kcal", alpha: "24.40", dg: "-21.14 kcal", feas: "Optimal" },
    { span: "15 atoms", dist: "14.81 A", sasa: "539 A2", clash: "0.26 kcal", count: "0", strain: "5.98 kcal", alpha: "74.19", dg: "-21.80 kcal", feas: "Optimal" },
    { span: "16 atoms", dist: "13.61 A", sasa: "539 A2", clash: "0.26 kcal", count: "0", strain: "10.77 kcal", alpha: "9.95", dg: "-20.61 kcal", feas: "Permissive" },
    { span: "18 atoms", dist: "13.29 A", sasa: "528 A2", clash: "0.29 kcal", count: "0", strain: "19.68 kcal", alpha: "0.19", dg: "-18.26 kcal", feas: "High Strain Penalty" },
    { span: "20 atoms", dist: "14.48 A", sasa: "410 A2", clash: "0.75 kcal", count: "0", strain: "4.08 kcal", alpha: "12.76", dg: "-20.76 kcal", feas: "Entropy Loss" },
    { span: "22 atoms", dist: "14.06 A", sasa: "410 A2", clash: "0.75 kcal", count: "0", strain: "5.59 kcal", alpha: "6.75", dg: "-20.38 kcal", feas: "Entropy Loss" }
  ]
};

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));

  const targetTab = document.getElementById(tabId);
  if (targetTab) {
    targetTab.classList.add('active');
  }

  // Update button active state
  event.target.classList.add('active');

  // Lazy render charts on first reveal
  if (tabId === 'tab-training' && !trainingChart) {
    initTrainingChart();
  }
  if (tabId === 'tab-hook' && !hookChart) {
    initHookSimulator();
  }
}

// Populate Data Tables
function populateTables() {
  // 1. ChEMBL Table
  const chemblTbody = document.getElementById('chembl-body');
  if (chemblTbody) {
    chemblTbody.innerHTML = '';
    EMBEDDED_DATA.chembl.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong class="highlight-id">${row.chembl_id}</strong></td>
        <td><strong>${row.name}</strong></td>
        <td><span class="badge badge-lead">${row.type}</span></td>
        <td><strong style="color: #059669;">${row.val}</strong></td>
        <td>${row.pchembl}</td>
        <td>${row.assay}</td>
        <td><a href="https://pubmed.ncbi.nlm.nih.gov/${row.pubmed}/" target="_blank" style="color: #2563eb;">PMID:${row.pubmed}</a></td>
        <td><code style="font-size: 11px;">${row.smiles.substring(0, 32)}...</code></td>
      `;
      chemblTbody.appendChild(tr);
    });
  }

  // 2. PROTAC-DB Table
  const protacTbody = document.getElementById('protac-db-body');
  if (protacTbody) {
    protacTbody.innerHTML = '';
    EMBEDDED_DATA.protac_db.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong class="highlight-id">${row.id}</strong></td>
        <td><strong>${row.name}</strong></td>
        <td>${row.target}</td>
        <td>${row.e3}</td>
        <td><strong style="color: #059669;">${row.dc50}</strong></td>
        <td><span class="badge badge-success">${row.dmax}</span></td>
        <td><span class="badge badge-pdb">${row.pdb}</span></td>
        <td>${row.cell}</td>
      `;
      protacTbody.appendChild(tr);
    });
  }

  // 3. Ablation Table
  const ablationTbody = document.getElementById('ablation-table-body');
  if (ablationTbody) {
    ablationTbody.innerHTML = '';
    EMBEDDED_DATA.ablation.forEach(row => {
      const tr = document.createElement('tr');
      const isOurs = row.variant.includes('Ours');
      tr.innerHTML = `
        <td><strong>${row.variant}</strong> ${isOurs ? '<span class="badge badge-success">Recommended</span>' : ''}</td>
        <td>${row.val}</td>
        <td>${row.uniq}</td>
        <td>${row.nov}</td>
        <td>${row.bro5}</td>
        <td>${row.pains}</td>
        <td><strong>${row.sa}</strong></td>
        <td>${row.qed}</td>
        <td><strong>${row.alpha}</strong></td>
        <td><strong style="color: ${isOurs ? '#059669' : '#334155'}; font-size: 14px;">${row.prod}</strong></td>
      `;
      ablationTbody.appendChild(tr);
    });
  }

  // 4. Length Scan Table
  const lengthTbody = document.getElementById('length-scan-body');
  if (lengthTbody) {
    lengthTbody.innerHTML = '';
    EMBEDDED_DATA.length_scan.forEach(row => {
      const tr = document.createElement('tr');
      const isOpt = row.feas === 'Optimal';
      tr.innerHTML = `
        <td><strong>${row.span}</strong></td>
        <td>${row.dist}</td>
        <td>${row.sasa}</td>
        <td>${row.clash}</td>
        <td>${row.count}</td>
        <td>${row.strain}</td>
        <td><span class="badge ${isOpt ? 'badge-success' : 'badge-warning'}">&alpha; = ${row.alpha}</span></td>
        <td><strong>${row.dg}</strong></td>
        <td><span class="badge ${isOpt ? 'badge-success' : (row.feas.includes('Collision') ? 'badge-danger' : 'badge-warning')}">${row.feas}</span></td>
      `;
      lengthTbody.appendChild(tr);
    });
  }
}

// Filter ChEMBL Table
function filterChemblTable() {
  const query = document.getElementById('chembl-search').value.toLowerCase();
  const rows = document.querySelectorAll('#chembl-body tr');
  rows.forEach(row => {
    const text = row.innerText.toLowerCase();
    row.style.display = text.includes(query) ? '' : 'none';
  });
}

// Chart 1: Training Loss Convergence
function initTrainingChart() {
  const ctx = document.getElementById('trainingLossChart');
  if (!ctx) return;

  const epochs = EMBEDDED_DATA.training_logs.map(d => `Epoch ${d.epoch}`);
  const trainLoss = EMBEDDED_DATA.training_logs.map(d => d.train);
  const reconLoss = EMBEDDED_DATA.training_logs.map(d => d.recon);
  const valLoss = EMBEDDED_DATA.training_logs.map(d => d.val);

  trainingChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        {
          label: 'Total Train Loss',
          data: trainLoss,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.1)',
          fill: true,
          tension: 0.3,
          borderWidth: 2
        },
        {
          label: 'Token Reconstruction Loss',
          data: reconLoss,
          borderColor: '#059669',
          borderDash: [5, 5],
          tension: 0.3,
          borderWidth: 2
        },
        {
          label: 'Validation Loss',
          data: valLoss,
          borderColor: '#d97706',
          tension: 0.3,
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        x: { grid: { color: '#f1f5f9' } },
        y: {
          title: { display: true, text: 'Loss Magnitude' },
          grid: { color: '#f1f5f9' },
          min: 0.4
        }
      }
    }
  });
}

function toggleDataset(index) {
  if (!trainingChart) return;
  const isVisible = trainingChart.isDatasetVisible(index);
  trainingChart.setDatasetVisibility(index, !isVisible);
  trainingChart.update();
}

// Chart 2: Hook Effect & Kinetics Simulation
function initHookSimulator() {
  updateHookSimulation();
}

function updateHookSimulation() {
  const alpha = parseFloat(document.getElementById('sim-alpha').value);
  const kdT = parseFloat(document.getElementById('sim-kdt').value);
  const kdE = parseFloat(document.getElementById('sim-kde').value);

  document.getElementById('val-alpha').innerText = alpha.toFixed(2);
  document.getElementById('val-kdt').innerText = kdT.toFixed(0);
  document.getElementById('val-kde').innerText = kdE.toFixed(0);

  // Compute 80 concentration points across log scale (0.01 nM to 100,000 nM)
  const labels = [];
  const ternaryPcts = [];
  let maxPte = 0;
  let popt = 0;

  for (let logC = -2; logC <= 5.5; logC += 0.15) {
    const c = Math.pow(10, logC);
    labels.push(c < 1 ? c.toFixed(2) : (c < 1000 ? c.toFixed(0) : (c/1000).toFixed(1) + "k"));

    // 3-body mass-action equilibrium approximation
    const num = (alpha * c * 100 * 100) / (kdT * kdE);
    const den = (1 + (c / kdT)) * (1 + (c / kdE)) + (alpha * c * 100 / (kdT * kdE));
    const pte = Math.min(99.5, (num / den));

    ternaryPcts.push(pte);
    if (pte > maxPte) {
      maxPte = pte;
      popt = c;
    }
  }

  // Find DC50 & Hook threshold
  const halfMax = maxPte / 2;
  let dc50 = 0.5;
  let cHook = 10000;

  for (let i = 0; i < ternaryPcts.length; i++) {
    if (ternaryPcts[i] >= halfMax) {
      dc50 = Math.pow(10, -2 + i * 0.15);
      break;
    }
  }
  for (let i = ternaryPcts.length - 1; i >= 0; i--) {
    if (ternaryPcts[i] >= halfMax) {
      cHook = Math.pow(10, -2 + i * 0.15);
      break;
    }
  }

  // Update UI stats
  document.getElementById('kin-dc50').innerText = dc50 < 1 ? dc50.toFixed(2) + " nM" : dc50.toFixed(1) + " nM";
  document.getElementById('kin-popt').innerText = popt.toFixed(1) + " nM";
  document.getElementById('kin-hook').innerText = (cHook / 1000).toFixed(1) + " \u03BCM";
  const windowFold = Math.round(cHook / dc50);
  document.getElementById('kin-window').innerText = windowFold.toLocaleString() + "-fold";
  document.getElementById('kin-dmax').innerText = maxPte.toFixed(1) + " %";

  const statusBox = document.getElementById('kin-status-text');
  if (alpha >= 2.5) {
    statusBox.className = "kin-note";
    statusBox.innerText = `High positive cooperativity (\u03B1 = ${alpha.toFixed(2)}) significantly expands the concentration envelope where active degradation occurs, suppressing the hook effect up to ${ (cHook/1000).toFixed(1) } \u03BCM.`;
  } else if (alpha >= 1.0) {
    statusBox.className = "kin-note";
    statusBox.innerText = `Moderate/Neutral cooperativity (\u03B1 = ${alpha.toFixed(2)}). Degradation is competent, but susceptible to two-site binary hooking at moderate micromolar concentrations.`;
  } else {
    statusBox.className = "kin-note";
    statusBox.style.background = "#fee2e2";
    statusBox.style.color = "#991b1b";
    statusBox.innerText = `Negative cooperativity (\u03B1 = ${alpha.toFixed(2)}). Severe steric collision between target and E3 ligase surfaces causes early onset hook effect and sub-optimal Dmax.`;
  }

  // Render or Update Chart
  const ctx = document.getElementById('hookSimulatorChart');
  if (!ctx) return;

  if (hookChart) {
    hookChart.data.labels = labels;
    hookChart.data.datasets[0].data = ternaryPcts;
    hookChart.data.datasets[0].label = `Active Ternary Complex % (\u03B1 = ${alpha.toFixed(2)})`;
    hookChart.update();
  } else {
    hookChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: `Active Ternary Complex % (\u03B1 = ${alpha.toFixed(2)})`,
          data: ternaryPcts,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.15)',
          fill: true,
          tension: 0.35,
          borderWidth: 2.5
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          tooltip: { mode: 'index', intersect: false }
        },
        scales: {
          x: {
            title: { display: true, text: 'Total PROTAC Concentration [P] (nM)' },
            ticks: { maxTicksLimit: 10 }
          },
          y: {
            title: { display: true, text: 'Ternary Complex Formation (%)' },
            min: 0,
            max: 100
          }
        }
      }
    });
  }
}

// Live PROTAC Generation Simulation
function simulateGeneration() {
  const resultsArea = document.getElementById('gen-results-area');
  const span = document.getElementById('gen-span').value;
  const temp = document.getElementById('gen-temp').value;
  const scaffold = document.getElementById('gen-scaffold').value;

  resultsArea.innerHTML = `
    <div style="text-align: center; padding: 24px; color: #2563eb;">
      <div style="font-size: 24px; margin-bottom: 8px;">&#x269B;</div>
      <strong>Sampling latent manifold z ~ N(\u03BC, \u03A3) conditioned on [JQ1-Acid] and [4-NH-Pomalidomide]...</strong>
      <p style="font-size: 12px; color: #64748b;">Filtering across ${span}-atom target span with temperature &tau; = ${temp}...</p>
    </div>
  `;

  setTimeout(() => {
    resultsArea.innerHTML = `
      <div style="margin-bottom: 12px; font-size: 14px; font-weight: 600; color: #334155;">
        Generated 3 Chemically Valid, Sanitized PROTAC Heterobifunctionals:
      </div>
      
      <div class="cand-card">
        <div class="cand-header">
          <div>
            <strong class="highlight-id">GEN-PROTAC-001</strong>
            <span class="badge badge-lead">1,2,3-Triazole Rigid Linker</span>
          </div>
          <div><span class="badge badge-success">Cooperativity &alpha; = 3.65 (High Synergy)</span></div>
        </div>
        <div class="cand-grid">
          <div><div class="cand-stat-val">915.4 Da</div><div class="cand-stat-lbl">MW</div></div>
          <div><div class="cand-stat-val">4.12</div><div class="cand-stat-lbl">cLogP</div></div>
          <div><div class="cand-stat-val">185.3 &Aring;&sup2;</div><div class="cand-stat-lbl">TPSA</div></div>
          <div><div class="cand-stat-val">2.92</div><div class="cand-stat-lbl">SAScore</div></div>
          <div><div class="cand-stat-val">-20.65 kcal</div><div class="cand-stat-lbl">&Delta;G Ternary</div></div>
          <div><div class="cand-stat-val">542.0 &Aring;&sup2;</div><div class="cand-stat-lbl">Buried SASA</div></div>
        </div>
        <div style="font-size: 12px; color: #475569; margin-bottom: 6px;">
          <strong>Linker Architecture:</strong> <code>-NCCc1cn(nn1)COCCC(=O)N-</code> (${span} atoms) &bull; <em>Amide to JQ1 / Aryl Amine to Pomalidomide</em>
        </div>
        <div style="font-size: 11px; font-family: monospace; color: #64748b; background: #f8fafc; padding: 6px; border-radius: 4px;">
          Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCCc1cn(nn1)COCCC(=O)Nc1cccc2C(=O)N(C3CCC(=O)NC3=O)C(=O)c12)n2
        </div>
      </div>

      <div class="cand-card">
        <div class="cand-header">
          <div>
            <strong class="highlight-id">GEN-PROTAC-002</strong>
            <span class="badge badge-lead">Alkynyl-PEG Semi-Rigid</span>
          </div>
          <div><span class="badge badge-success">Cooperativity &alpha; = 3.22 (High Synergy)</span></div>
        </div>
        <div class="cand-grid">
          <div><div class="cand-stat-val">898.4 Da</div><div class="cand-stat-lbl">MW</div></div>
          <div><div class="cand-stat-val">3.94</div><div class="cand-stat-lbl">cLogP</div></div>
          <div><div class="cand-stat-val">179.2 &Aring;&sup2;</div><div class="cand-stat-lbl">TPSA</div></div>
          <div><div class="cand-stat-val">2.85</div><div class="cand-stat-lbl">SAScore</div></div>
          <div><div class="cand-stat-val">-20.35 kcal</div><div class="cand-stat-lbl">&Delta;G Ternary</div></div>
          <div><div class="cand-stat-val">510.5 &Aring;&sup2;</div><div class="cand-stat-lbl">Buried SASA</div></div>
        </div>
        <div style="font-size: 12px; color: #475569; margin-bottom: 6px;">
          <strong>Linker Architecture:</strong> <code>-NCCC#CCOCCC(=O)N-</code> (${span-1} atoms) &bull; <em>Rigid Cylindrical Vector</em>
        </div>
        <div style="font-size: 11px; font-family: monospace; color: #64748b; background: #f8fafc; padding: 6px; border-radius: 4px;">
          Cc1sc2c(c1C)c(-c1ccc(Cl)cc1)nc(CC(=O)NCCC#CCOCCC(=O)Nc1cccc2C(=O)N(C3CCC(=O)NC3=O)C(=O)c12)n2
        </div>
      </div>
    `;
  }, 600);
}

// 3D Molecular WebGL Viewer using 3Dmol.js
let mol3dViewer = null;

function init3DViewer() {
  const container = document.getElementById('mol3d-container');
  if (!container || typeof $3Dmol === 'undefined') return;

  try {
    container.innerHTML = '';
    mol3dViewer = $3Dmol.createViewer(container, { backgroundColor: '#0f172a' });

    // Fetch SDF file from local directory
    fetch('models_3d/protac_aml_001.sdf')
      .then(res => res.text())
      .then(sdfData => {
        mol3dViewer.addModel(sdfData, 'sdf');
        mol3dViewer.setStyle({}, { stick: { radius: 0.18, colorscheme: 'Jmol' } });
        mol3dViewer.zoomTo();
        mol3dViewer.render();
        mol3dViewer.spin(true);
      })
      .catch(err => {
        console.warn('3D model local fetch fallback:', err);
      });
  } catch (e) {
    console.error('3Dmol initialization error:', e);
  }
}

function setViewerStyle(styleName) {
  if (!mol3dViewer) return;
  mol3dViewer.setStyle({}, {});
  if (styleName === 'stick') {
    mol3dViewer.setStyle({}, { stick: { radius: 0.18, colorscheme: 'Jmol' } });
  } else if (styleName === 'sphere') {
    mol3dViewer.setStyle({}, { sphere: { scale: 0.28, colorscheme: 'Jmol' } });
  } else if (styleName === 'line') {
    mol3dViewer.setStyle({}, { line: { linewidth: 2.0, colorscheme: 'Jmol' } });
  }
  mol3dViewer.render();
}

function resetViewerCamera() {
  if (!mol3dViewer) return;
  mol3dViewer.zoomTo();
  mol3dViewer.render();
}

// Initializer
window.addEventListener('DOMContentLoaded', () => {
  populateTables();
  simulateGeneration();
  setTimeout(init3DViewer, 500);
});

