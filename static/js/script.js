/**
 * FinGuard - Frontend JavaScript Application
 * Handles API calls, dynamic table rendering, Chart.js visualizations,
 * modal dialogs, and real-time fraud inference form processing.
 */

// Formatting Helpers
function formatINR(amount) {
  if (amount === undefined || amount === null || isNaN(amount)) return 'INR 0.00';
  return 'INR ' + Number(amount).toLocaleString('en-IN', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  });
}

function formatNumber(num) {
  if (num === undefined || num === null || isNaN(num)) return '0';
  return Number(num).toLocaleString('en-IN');
}

// Chart Palette (Warm, restrained financial colors)
const CHART_COLORS = {
  bronze: '#8d532d',
  bronzeLight: '#c99b7d',
  charcoal: '#242220',
  grey: '#8e8a84',
  greyLight: '#e6e2d8',
  green: '#215e37',
  greenLight: '#8ec59f',
  red: '#962828',
  redLight: '#e48a8a',
  blue: '#344c64',
  blueLight: '#85a4c5'
};

/* ==========================================================================
   PAGE 1: DASHBOARD LOGIC
   ========================================================================== */
function initDashboard() {
  const kpiTotal = document.getElementById('kpi-total-txns');
  if (!kpiTotal) return;

  fetch('/api/dashboard')
    .then(res => res.json())
    .then(data => {
      if (!data.success) return;
      const kpis = data.kpis;

      // Populate KPI Cards
      document.getElementById('kpi-total-txns').textContent = formatNumber(kpis.total_transactions);
      document.getElementById('kpi-legit-txns').textContent = formatNumber(kpis.legitimate_transactions);
      document.getElementById('kpi-fraud-txns').textContent = formatNumber(kpis.fraudulent_transactions);
      document.getElementById('kpi-fraud-pct').textContent = kpis.fraud_percentage + '%';
      document.getElementById('kpi-total-vol').textContent = formatINR(kpis.total_amount);
      document.getElementById('kpi-active-model').textContent = kpis.selected_model;

      // Populate Recent Transactions Table
      renderRecentTransactions(data.recent_transactions);

      // Render Dashboard Charts
      renderDashboardCharts(data.charts);
    })
    .catch(err => console.error('Error loading dashboard data:', err));
}

function renderRecentTransactions(txns) {
  const tbody = document.getElementById('recent-txns-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (!txns || txns.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:1.5rem; color:var(--text-muted);">No transactions available.</td></tr>';
    return;
  }

  txns.forEach(tx => {
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.onclick = () => openTxnModal(tx.transaction_id);

    const isFraud = tx.is_fraud === 1;
    const badgeHtml = isFraud
      ? '<span class="badge badge-fraud">Fraud</span>'
      : '<span class="badge badge-legit">Legitimate</span>';

    tr.innerHTML = `
      <td class="mono"><strong>${tx.transaction_id}</strong></td>
      <td>${tx.transaction_date} <span style="color:var(--text-muted); font-size:0.75rem;">${tx.transaction_time || ''}</span></td>
      <td class="mono">${formatINR(tx.transaction_amount)}</td>
      <td><span class="badge badge-neutral">${tx.transaction_type}</span></td>
      <td>${tx.channel}</td>
      <td>${badgeHtml}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderDashboardCharts(charts) {
  if (typeof Chart === 'undefined') return;

  // Chart 1: Fraud vs Legitimate
  const ctxFraudLegit = document.getElementById('chart-dash-fraud-legit');
  if (ctxFraudLegit && charts.fraud_vs_legit) {
    new Chart(ctxFraudLegit, {
      type: 'doughnut',
      data: {
        labels: ['Legitimate (545,127)', 'Fraudulent (4,873)'],
        datasets: [{
          data: charts.fraud_vs_legit.data,
          backgroundColor: [CHART_COLORS.green, CHART_COLORS.red],
          borderWidth: 1,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } }
        },
        cutout: '68%'
      }
    });
  }

  // Chart 2: Annual Volume
  const ctxYear = document.getElementById('chart-dash-activity');
  if (ctxYear && charts.by_year) {
    const years = charts.by_year.map(item => item.year);
    const counts = charts.by_year.map(item => item.count);
    new Chart(ctxYear, {
      type: 'bar',
      data: {
        labels: years,
        datasets: [{
          label: 'Transaction Count',
          data: counts,
          backgroundColor: CHART_COLORS.bronze,
          borderRadius: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, grid: { color: '#f0ece4' } },
          x: { grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // Chart 3: Fraud by Category
  const ctxCat = document.getElementById('chart-dash-categories');
  if (ctxCat && charts.by_category) {
    const categories = charts.by_category.map(item => item.category);
    const rates = charts.by_category.map(item => item.fraud_rate);
    new Chart(ctxCat, {
      type: 'bar',
      data: {
        labels: categories,
        datasets: [{
          label: 'Fraud Rate (%)',
          data: rates,
          backgroundColor: CHART_COLORS.blue,
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { beginAtZero: true, grid: { color: '#f0ece4' }, ticks: { callback: v => v + '%' } },
          y: { grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }
}

/* ==========================================================================
   PAGE 2: TRANSACTIONS RECORD TABLE & PAGINATION
   ========================================================================== */
let txPageState = {
  page: 1,
  perPage: 20,
  search: '',
  fraudStatus: 'all',
  txType: 'all',
  channel: 'all',
  sortBy: 'transaction_date',
  sortOrder: 'desc'
};

function initTransactions() {
  const table = document.getElementById('transactions-table');
  if (!table) return;

  // Search input with debounce
  const searchInput = document.getElementById('filter-search');
  let searchTimeout = null;
  searchInput.addEventListener('input', e => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      txPageState.search = e.target.value.trim();
      txPageState.page = 1;
      loadTransactions();
    }, 300);
  });

  // Filter dropdowns
  document.getElementById('filter-status').addEventListener('change', e => {
    txPageState.fraudStatus = e.target.value;
    txPageState.page = 1;
    loadTransactions();
  });

  document.getElementById('filter-type').addEventListener('change', e => {
    txPageState.txType = e.target.value;
    txPageState.page = 1;
    loadTransactions();
  });

  document.getElementById('filter-channel').addEventListener('change', e => {
    txPageState.channel = e.target.value;
    txPageState.page = 1;
    loadTransactions();
  });

  document.getElementById('filter-sort').addEventListener('change', e => {
    const [col, dir] = e.target.value.split(':');
    txPageState.sortBy = col;
    txPageState.sortOrder = dir;
    txPageState.page = 1;
    loadTransactions();
  });

  // Pagination buttons
  document.getElementById('btn-prev-page').addEventListener('click', () => {
    if (txPageState.page > 1) {
      txPageState.page--;
      loadTransactions();
    }
  });

  document.getElementById('btn-next-page').addEventListener('click', () => {
    txPageState.page++;
    loadTransactions();
  });

  loadTransactions();
}

function loadTransactions() {
  const tbody = document.getElementById('tx-table-body');
  tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:2rem; color:var(--text-muted);">Querying bank transaction database...</td></tr>';

  const params = new URLSearchParams({
    page: txPageState.page,
    per_page: txPageState.perPage,
    search: txPageState.search,
    fraud_status: txPageState.fraudStatus,
    transaction_type: txPageState.txType,
    channel: txPageState.channel,
    sort_by: txPageState.sortBy,
    sort_order: txPageState.sortOrder
  });

  fetch(`/api/transactions?${params.toString()}`)
    .then(res => res.json())
    .then(data => {
      if (!data.success) return;
      renderTransactionsTable(data.transactions);
      updatePaginationControls(data);
    })
    .catch(err => {
      console.error('Error querying transactions:', err);
      tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:2rem; color:var(--status-fraud-text);">Failed to load transactions.</td></tr>';
    });
}

function renderTransactionsTable(transactions) {
  const tbody = document.getElementById('tx-table-body');
  tbody.innerHTML = '';

  if (!transactions || transactions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:2rem; color:var(--text-muted);">No transactions match your search or filter criteria.</td></tr>';
    return;
  }

  transactions.forEach(tx => {
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.title = 'Click to inspect complete transaction details';
    tr.onclick = () => openTxnModal(tx.transaction_id);

    const isFraud = tx.is_fraud === 1;
    const fraudBadge = isFraud
      ? '<span class="badge badge-fraud">Fraud</span>'
      : '<span class="badge badge-legit">Legitimate</span>';

    const statusBadge = tx.transaction_status === 'Success'
      ? '<span class="badge badge-neutral">Success</span>'
      : `<span class="badge badge-info">${tx.transaction_status}</span>`;

    tr.innerHTML = `
      <td class="mono"><strong>${tx.transaction_id}</strong></td>
      <td class="mono" style="color:var(--text-secondary); font-size:0.775rem;">${tx.customer_id}</td>
      <td>${tx.transaction_date}</td>
      <td class="mono"><strong>${formatINR(tx.transaction_amount)}</strong></td>
      <td><span class="badge badge-neutral">${tx.transaction_type}</span></td>
      <td>${tx.account_type}</td>
      <td>${tx.channel}</td>
      <td>${statusBadge}</td>
      <td>${fraudBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

function updatePaginationControls(meta) {
  const pageInfo = document.getElementById('pagination-info');
  const prevBtn = document.getElementById('btn-prev-page');
  const nextBtn = document.getElementById('btn-next-page');

  const startRecord = (meta.page - 1) * meta.per_page + 1;
  const endRecord = Math.min(meta.total_records, meta.page * meta.per_page);

  pageInfo.textContent = `Showing ${formatNumber(startRecord)} - ${formatNumber(endRecord)} of ${formatNumber(meta.total_records)} records (Page ${meta.page} of ${meta.total_pages})`;

  prevBtn.disabled = meta.page <= 1;
  nextBtn.disabled = meta.page >= meta.total_pages;
}

// Transaction Detail Modal
function openTxnModal(txnId) {
  const modal = document.getElementById('txn-modal');
  if (!modal) return;

  fetch(`/api/transactions/${txnId}`)
    .then(res => res.json())
    .then(data => {
      if (!data.success) return;
      const tx = data.transaction;

      document.getElementById('modal-txn-id').textContent = tx.transaction_id;
      document.getElementById('modal-cust-id').textContent = tx.customer_id;
      document.getElementById('modal-date').textContent = `${tx.transaction_date} ${tx.transaction_time || ''}`;
      document.getElementById('modal-amount').textContent = formatINR(tx.transaction_amount);
      document.getElementById('modal-balance').textContent = formatINR(tx.account_balance);
      document.getElementById('modal-type').textContent = tx.transaction_type;
      document.getElementById('modal-direction').textContent = tx.transaction_direction;
      document.getElementById('modal-acc-type').textContent = tx.account_type;
      document.getElementById('modal-merchant').textContent = tx.merchant_category;
      document.getElementById('modal-state').textContent = tx.state;
      document.getElementById('modal-channel').textContent = tx.channel;
      document.getElementById('modal-status').textContent = tx.transaction_status;
      document.getElementById('modal-credit-score').textContent = tx.credit_score;
      document.getElementById('modal-kyc').textContent = tx.kyc_status;
      document.getElementById('modal-loan').textContent = tx.has_loan === 1 ? `Yes (${tx.loan_type})` : 'No Loan';
      document.getElementById('modal-emi').textContent = formatINR(tx.emi_amount);

      const statusTag = document.getElementById('modal-fraud-status');
      if (tx.is_fraud === 1) {
        statusTag.innerHTML = '<span class="badge badge-fraud" style="font-size:0.85rem;">POTENTIAL FRAUD</span>';
      } else {
        statusTag.innerHTML = '<span class="badge badge-legit" style="font-size:0.85rem;">LEGITIMATE</span>';
      }

      modal.classList.add('open');
    });
}

function closeTxnModal() {
  const modal = document.getElementById('txn-modal');
  if (modal) modal.classList.remove('open');
}

/* ==========================================================================
   PAGE 3: FRAUD DETECTION FORM & INFERENCE
   ========================================================================== */
function initFraudDetection() {
  const form = document.getElementById('fraud-analysis-form');
  if (!form) return;

  // Preset Auto-fill buttons (Useful for quick viva & live presentations)
  const btnFillLegit = document.getElementById('btn-preset-legit');
  const btnFillFraud = document.getElementById('btn-preset-fraud');
  const btnReset = document.getElementById('btn-preset-reset');

  if (btnFillLegit) {
    btnFillLegit.addEventListener('click', () => loadSamplePreset('legitimate'));
  }
  if (btnFillFraud) {
    btnFillFraud.addEventListener('click', () => loadSamplePreset('fraud'));
  }
  if (btnReset) {
    btnReset.addEventListener('click', () => {
      form.reset();
      document.getElementById('prediction-result-wrap').style.display = 'none';
    });
  }

  // Handle Form Submission
  form.addEventListener('submit', e => {
    e.preventDefault();
    analyzeTransaction(form);
  });
}

function loadSamplePreset(type) {
  fetch(`/api/sample-transaction?type=${type}`)
    .then(res => res.json())
    .then(data => {
      if (!data.success) return;
      const s = data.sample;

      document.getElementById('f-amount').value = s.transaction_amount;
      document.getElementById('f-balance').value = s.account_balance;
      document.getElementById('f-acc-type').value = s.account_type;
      document.getElementById('f-tx-type').value = s.transaction_type;
      document.getElementById('f-direction').value = s.transaction_direction;
      document.getElementById('f-merchant').value = s.merchant_category;
      document.getElementById('f-state').value = s.state;
      document.getElementById('f-credit-score').value = s.credit_score;
      document.getElementById('f-has-loan').value = s.has_loan;
      document.getElementById('f-loan-type').value = s.loan_type;
      document.getElementById('f-emi').value = s.emi_amount;
      document.getElementById('f-status').value = s.transaction_status;
      document.getElementById('f-channel').value = s.channel;
      document.getElementById('f-kyc').value = s.kyc_status;
      document.getElementById('f-date').value = s.transaction_date;
      document.getElementById('f-time').value = s.transaction_time || '14:30';

      // Hide previous result
      document.getElementById('prediction-result-wrap').style.display = 'none';
    })
    .catch(err => console.error('Error fetching preset sample:', err));
}

function analyzeTransaction(form) {
  const submitBtn = document.getElementById('btn-analyze-submit');
  const originalText = submitBtn.textContent;
  submitBtn.disabled = true;
  submitBtn.textContent = 'EVALUATING RISK PROFILE...';

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
    .then(res => res.json())
    .then(result => {
      submitBtn.disabled = false;
      submitBtn.textContent = originalText;

      if (!result.success) {
        alert('Validation error: ' + (result.errors ? result.errors.join(', ') : 'Evaluation failed.'));
        return;
      }

      displayPredictionResult(result);
    })
    .catch(err => {
      submitBtn.disabled = false;
      submitBtn.textContent = originalText;
      console.error('Error in risk scoring:', err);
      alert('Network or server error during fraud inference.');
    });
}

function displayPredictionResult(res) {
  const wrap = document.getElementById('prediction-result-wrap');
  const banner = document.getElementById('res-banner');
  const title = document.getElementById('res-title');
  const probScore = document.getElementById('res-prob');
  const timestamp = document.getElementById('res-timestamp');
  const factorsList = document.getElementById('res-factors');
  const recBox = document.getElementById('res-recommendation');
  const disclaimer = document.getElementById('res-disclaimer');

  wrap.style.display = 'block';

  // State styling
  if (res.is_fraud) {
    banner.className = 'result-banner is-fraud';
    title.textContent = 'POTENTIAL FRAUD DETECTED';
  } else {
    banner.className = 'result-banner is-legit';
    title.textContent = 'LEGITIMATE TRANSACTION';
  }

  probScore.textContent = res.fraud_probability_pct + '%';
  timestamp.textContent = `Evaluated at ${res.timestamp} | Model: ${res.model_name} (Decision Threshold: ${res.threshold_used})`;

  // Explanation factors
  factorsList.innerHTML = '';
  res.explanation_factors.forEach(factor => {
    const li = document.createElement('li');
    li.textContent = factor;
    factorsList.appendChild(li);
  });

  recBox.textContent = res.recommendation;
  disclaimer.textContent = res.disclaimer;

  // Smooth scroll
  wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ==========================================================================
   PAGE 4: ANALYTICS VISUALIZATIONS
   ========================================================================== */
function initAnalytics() {
  const container = document.getElementById('analytics-container');
  if (!container || typeof Chart === 'undefined') return;

  fetch('/api/analytics')
    .then(res => res.json())
    .then(data => {
      if (!data.success) return;
      const a = data.analytics;

      // 1. Class Distribution
      const ctxDist = document.getElementById('chart-an-distribution');
      if (ctxDist) {
        new Chart(ctxDist, {
          type: 'doughnut',
          data: {
            labels: ['Legitimate (545,127)', 'Fraudulent (4,873)'],
            datasets: [{
              data: [a.kpis.legitimate_transactions, a.kpis.fraudulent_transactions],
              backgroundColor: [CHART_COLORS.green, CHART_COLORS.red],
              borderWidth: 1
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            cutout: '65%'
          }
        });
      }

      // 2. Fraud by Transaction Type
      const ctxType = document.getElementById('chart-an-type');
      if (ctxType) {
        new Chart(ctxType, {
          type: 'bar',
          data: {
            labels: a.by_type.map(item => item.type),
            datasets: [{
              label: 'Fraud Rate (%)',
              data: a.by_type.map(item => item.fraud_rate),
              backgroundColor: CHART_COLORS.bronze,
              borderRadius: 3
            }]
          },
          options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { beginAtZero: true, ticks: { callback: v => v + '%' } },
              y: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }

      // 3. Fraud by Channel
      const ctxChan = document.getElementById('chart-an-channel');
      if (ctxChan) {
        new Chart(ctxChan, {
          type: 'bar',
          data: {
            labels: a.by_channel.map(item => item.channel),
            datasets: [{
              label: 'Fraud Rate (%)',
              data: a.by_channel.map(item => item.fraud_rate),
              backgroundColor: CHART_COLORS.blue,
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: true, ticks: { callback: v => v + '%' } },
              x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }

      // 4. Fraud by Merchant Category
      const ctxCat = document.getElementById('chart-an-category');
      if (ctxCat) {
        new Chart(ctxCat, {
          type: 'bar',
          data: {
            labels: a.by_category.map(item => item.category),
            datasets: [{
              label: 'Fraud Count',
              data: a.by_category.map(item => item.fraud_count),
              backgroundColor: CHART_COLORS.charcoal,
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: true },
              x: { grid: { display: false }, ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }

      // 5. Fraud by State
      const ctxState = document.getElementById('chart-an-state');
      if (ctxState) {
        new Chart(ctxState, {
          type: 'bar',
          data: {
            labels: a.by_state.map(item => item.state),
            datasets: [{
              label: 'Fraud Rate (%)',
              data: a.by_state.map(item => item.fraud_rate),
              backgroundColor: CHART_COLORS.bronzeLight,
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: true, ticks: { callback: v => v + '%' } },
              x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }

      // 6. Fraud by Hour (0-23)
      const ctxHour = document.getElementById('chart-an-hour');
      if (ctxHour) {
        new Chart(ctxHour, {
          type: 'line',
          data: {
            labels: a.by_hour.map(item => item.hour + ':00'),
            datasets: [{
              label: 'Fraud Count',
              data: a.by_hour.map(item => item.fraud_count),
              borderColor: CHART_COLORS.red,
              backgroundColor: 'rgba(150, 40, 40, 0.08)',
              fill: true,
              tension: 0.35,
              pointRadius: 2.5
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: true },
              x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }

      // 7. Fraud by Amount Range
      const ctxAmt = document.getElementById('chart-an-amount');
      if (ctxAmt) {
        new Chart(ctxAmt, {
          type: 'bar',
          data: {
            labels: a.by_amount_range.map(item => item.range),
            datasets: [{
              label: 'Fraud Rate (%)',
              data: a.by_amount_range.map(item => item.fraud_rate),
              backgroundColor: CHART_COLORS.green,
              borderRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: { beginAtZero: true, ticks: { callback: v => v + '%' } },
              x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
          }
        });
      }
    })
    .catch(err => console.error('Error loading analytics:', err));
}

/* ==========================================================================
   PAGE 5: MODEL PERFORMANCE & BENCHMARKS
   ========================================================================== */
function initModelPerformance() {
  const container = document.getElementById('model-performance-container');
  if (!container) return;

  fetch('/api/model-performance')
    .then(res => res.json())
    .then(data => {
      if (!data.success) return;
      const m = data.metrics;
      const p = data.preprocessing;

      // Overview
      document.getElementById('mp-selected-model').textContent = m.selected_algorithm;
      document.getElementById('mp-rationale').textContent = m.selection_rationale;

      // Dataset Split
      const split = m.dataset_split;
      document.getElementById('mp-train-size').textContent = `${formatNumber(split.train_size)} rows (${formatNumber(split.fraud_counts.train)} frauds)`;
      document.getElementById('mp-val-size').textContent = `${formatNumber(split.val_size)} rows (${formatNumber(split.fraud_counts.val)} frauds)`;
      document.getElementById('mp-test-size').textContent = `${formatNumber(split.test_size)} rows (${formatNumber(split.fraud_counts.test)} frauds)`;

      // Best Model Metrics
      const b = m.best_model_metrics;
      document.getElementById('mp-accuracy').textContent = (b.accuracy * 100).toFixed(2) + '%';
      document.getElementById('mp-precision').textContent = (b.precision * 100).toFixed(2) + '%';
      document.getElementById('mp-recall').textContent = (b.recall * 100).toFixed(2) + '%';
      document.getElementById('mp-f1').textContent = b.f1_score.toFixed(4);
      document.getElementById('mp-roc-auc').textContent = b.roc_auc.toFixed(4);
      document.getElementById('mp-pr-auc').textContent = b.pr_auc.toFixed(4);
      document.getElementById('mp-threshold').textContent = b.threshold_used.toFixed(4);

      // Confusion Matrix
      const cm = b.confusion_matrix;
      document.getElementById('cm-tn').textContent = formatNumber(cm[0][0]);
      document.getElementById('cm-fp').textContent = formatNumber(cm[0][1]);
      document.getElementById('cm-fn').textContent = formatNumber(cm[1][0]);
      document.getElementById('cm-tp').textContent = formatNumber(cm[1][1]);

      // Comparison Table
      renderModelComparisonTable(m.comparison, m.selected_algorithm);

      // Comparison Chart
      renderComparisonChart(m.comparison);

      // Feature Importance Table
      renderFeatureImportanceTable(m.top_features);
    })
    .catch(err => console.error('Error loading model metrics:', err));
}

function renderModelComparisonTable(comparison, bestName) {
  const tbody = document.getElementById('mp-comparison-body');
  if (!tbody) return;
  tbody.innerHTML = '';

  Object.entries(comparison).forEach(([name, metrics]) => {
    const tr = document.createElement('tr');
    const isBest = name === bestName;
    if (isBest) {
      tr.style.backgroundColor = '#faf6ef';
      tr.style.fontWeight = '600';
    }

    tr.innerHTML = `
      <td>
        <strong>${name}</strong>
        ${isBest ? ' <span class="badge badge-legit" style="font-size:0.68rem; margin-left:0.35rem;">SELECTED</span>' : ''}
      </td>
      <td class="mono">${(metrics.accuracy * 100).toFixed(2)}%</td>
      <td class="mono">${(metrics.precision * 100).toFixed(2)}%</td>
      <td class="mono">${(metrics.recall * 100).toFixed(2)}%</td>
      <td class="mono">${metrics.f1_score.toFixed(4)}</td>
      <td class="mono">${metrics.pr_auc.toFixed(4)}</td>
      <td class="mono">${metrics.roc_auc.toFixed(4)}</td>
      <td class="mono">${metrics.threshold_used.toFixed(4)}</td>
      <td class="mono">${metrics.train_time_sec}s</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderComparisonChart(comparison) {
  if (typeof Chart === 'undefined') return;
  const ctx = document.getElementById('chart-mp-comparison');
  if (!ctx) return;

  const names = Object.keys(comparison);
  const prAucs = names.map(n => comparison[n].pr_auc);
  const f1Scores = names.map(n => comparison[n].f1_score);
  const recalls = names.map(n => comparison[n].recall);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: names,
      datasets: [
        {
          label: 'PR-AUC',
          data: prAucs,
          backgroundColor: CHART_COLORS.bronze,
          borderRadius: 3
        },
        {
          label: 'F1-Score',
          data: f1Scores,
          backgroundColor: CHART_COLORS.blue,
          borderRadius: 3
        },
        {
          label: 'Fraud Recall',
          data: recalls,
          backgroundColor: CHART_COLORS.green,
          borderRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, grid: { color: '#f0ece4' } },
        x: { grid: { display: false } }
      },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12 } }
      }
    }
  });
}

function renderFeatureImportanceTable(features) {
  const tbody = document.getElementById('mp-features-body');
  if (!tbody || !features) return;
  tbody.innerHTML = '';

  features.slice(0, 15).forEach((item, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="color:var(--text-muted);">${idx + 1}</td>
      <td class="mono"><strong>${item.feature}</strong></td>
      <td class="mono">${item.importance.toFixed(5)}</td>
      <td>
        <div style="background-color:#eeebe3; height:8px; border-radius:4px; width:100%; max-width:200px; overflow:hidden;">
          <div style="background-color:var(--accent-bronze); height:100%; width:${Math.min(100, (item.importance / features[0].importance) * 100)}%;"></div>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Global DOM initialization dispatcher
document.addEventListener('DOMContentLoaded', () => {
  initDashboard();
  initTransactions();
  initFraudDetection();
  initAnalytics();
  initModelPerformance();
});
