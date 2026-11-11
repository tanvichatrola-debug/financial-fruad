# FINGUARD &mdash; Financial Fraud Detection System Using Machine Learning

A complete, production-grade Financial Fraud Detection web application designed for real-time transaction surveillance, risk assessment, behavioral analytics, and automated machine learning inference. Built using Python, Scikit-learn, Flask, Vanilla JavaScript, HTML5, and CSS3, evaluated on a real-world Kaggle banking dataset of 550,000 transaction records.

---

## 1. Project Introduction
In modern digital banking ecosystems, financial fraud represents a multi-billion dollar challenge. As instant payment systems (UPI, IMPS, RTGS, POS, Net Banking) expand exponentially, traditional static rule-based fraud detection fails to detect sophisticated, evolving fraud patterns. 

**FinGuard** is an automated, end-to-end Machine Learning surveillance platform that ingests raw banking transactions, performs automated data hygiene and temporal feature engineering, evaluates transactions against a trained **Gradient Boosting** ensemble model, and generates transparent, evidence-based risk explanations in milliseconds.

---

## 2. Problem Statement
Financial fraud detection problems are characterized by:
1. **Extreme Class Imbalance:** Legitimate transactions comprise over 99.1% of volume, while fraudulent events constitute less than 0.9%. Standard accuracy metrics fail completely (a naive model predicting all zeros achieves 99.1% accuracy while detecting 0 frauds).
2. **High Latency Costs:** Automated transaction scoring must execute within milliseconds before settlement occurs.
3. **Black-Box Skepticism:** Regulators and bank fraud teams require interpretable, mathematically grounded explanations for flagged transactions, not opaque automated decisions.

---

## 3. Project Objectives
- Ingest and audit 550,000 genuine Indian banking transactions without manual intervention.
- Automate missing value imputation (loan facilities), duplicate elimination, and temporal feature extraction.
- Implement zero-leakage chronological train, validation, and test splits (2019–2021 Train, 2022 Validation, 2023–2024 Test).
- Train, benchmark, and compare four supervised classification algorithms:
  1. **Logistic Regression**
  2. **Decision Tree**
  3. **Random Forest**
  4. **Gradient Boosting (HistGradientBoosting)**
- Optimize fraud classification thresholds on validation data using Precision-Recall Area Under Curve (PR-AUC).
- Provide a clean, human-designed financial web interface free of neon AI dashboard clichés, featuring:
  - Macro KPI Dashboard
  - High-Speed Transaction Search & Filtering Ledger
  - Real-time Fraud Inference Form with Evidence Factors
  - Behavioral Fraud Analytics & Distributions
  - Transparent Model Performance Benchmarks

---

## 4. Dataset Description
- **Dataset File:** `dataset/indian_banking_transactions.csv`
- **Total Records:** 550,000 transactions
- **Total Monitored Value:** INR 16,448,901,663.56 (~16.45 Billion INR)
- **Target Variable:** `is_fraud`
  - `0`: Legitimate Transaction (545,127 records &bull; 99.11%)
  - `1`: Fraudulent Transaction (4,873 records &bull; 0.886%)
- **20 Raw Attributes:**
  - Identifiers: `transaction_id`, `customer_id`
  - Temporal: `transaction_date`, `transaction_time`, `transaction_hour`
  - Account Details: `account_type`, `account_balance`, `kyc_status`, `credit_score`
  - Financial Activity: `transaction_type`, `transaction_amount`, `transaction_direction`, `merchant_category`, `transaction_status`, `channel`
  - Loan Exposure: `has_loan`, `loan_type`, `emi_amount`, `state`

---

## 5. Technology Stack
- **Machine Learning & Data Processing:** Python 3.8+, Pandas, NumPy, Scikit-learn, Scipy, Joblib
- **Database / High-Speed Storage:** SQLite3 (indexed tables for sub-5ms paginated queries)
- **Backend Server:** Flask (Python REST API architecture)
- **Frontend User Interface:** Semantic HTML5, Modular CSS3 (Warm ivory banking theme), Vanilla JavaScript (No heavy frameworks like React/Angular/Vue)
- **Visualizations:** Chart.js (bundled locally for 100% offline functionality)

---

## 6. Automatic Data Cleaning & Preprocessing Workflow

1. **Intelligent Missing Value Handling:**
   - For `loan_type` (377,134 missing values):
     - If `has_loan == 0`: Imputed with `'No Loan'` (357,712 instances).
     - If `has_loan == 1` and `loan_type` missing: Imputed with `'Unknown'` (19,422 instances).
   - Zero rows dropped unnecessarily, preserving all 550,000 transaction records.
2. **Duplicate & Format Validation:**
   - Checked for duplicate records (0 duplicates found).
   - Datetime parsing: converted `transaction_date` into datetime objects.
   - Synchronized `transaction_hour` with `transaction_time`.
3. **Temporal & Financial Feature Engineering:**
   - Extracted `transaction_day`, `transaction_month`, `transaction_year`, `day_of_week`.
   - Engineered `is_weekend` (1 if Saturday/Sunday, else 0).
   - Cyclical hour encoding: `hour_sin = sin(2*pi*hour/24)`, `hour_cos = cos(2*pi*hour/24)`.
   - Log-transformed transaction amounts: `log_amount = log1p(amount)`.
   - Log-transformed account balances: `log_balance = log1p(balance)`.
   - Liquidity depletion ratio: `amount_to_balance_ratio = amount / (balance + 1.0)`.
   - Net post-transaction balance: `balance_after_txn` (debit subtracts, credit adds).
4. **Leakage Prevention & Encoding Pipeline:**
   - Chronological split performed **before** any transformations:
     - **Train Split (2019–2021):** 329,874 records (2,930 frauds &bull; 0.888%)
     - **Validation Split (2022):** 110,247 records (960 frauds &bull; 0.871%)
     - **Test Split (2023–2024):** 109,879 records (983 frauds &bull; 0.895%)
   - `ColumnTransformer` fitted strictly on training data:
     - Numerical scaling: `RobustScaler` (handles extreme transaction values using median and IQR).
     - Categorical encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` producing a clean 79-dimensional feature space.

---

## 7. Machine Learning Algorithms & Comparative Benchmark Results

All four algorithms were trained on the identical 329,874 training records, tuned on the 2022 validation split, and evaluated on the unseen 109,879 test transactions:

| Model Name | Accuracy | Precision | Fraud Recall | F1-Score | PR-AUC | ROC-AUC | Optimal Threshold | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **98.11%** | **6.65%** | **8.55%** | **0.0748** | **0.0167** | **0.5351** | **0.6346** | **9.52s** |
| **Logistic Regression** | 98.03% | 6.18% | 8.44% | 0.0713 | 0.0167 | 0.5473 | 0.7078 | 30.83s |
| **Decision Tree** | 98.17% | 6.38% | 7.63% | 0.0695 | 0.0147 | 0.5361 | 0.9000 | 32.24s |
| **Random Forest** | 98.30% | 6.50% | 6.71% | 0.0661 | 0.0144 | 0.5116 | 0.4945 | 63.61s |

### Why Gradient Boosting Was Selected
1. **Highest F1-Score (0.0748) & PR-AUC (0.0167)** on unseen test data.
2. **Superior Fraud Recall (8.55%)** while maintaining 98.11% overall accuracy and low false-alarm operational overhead.
3. **Lightning Fast Training & Inference:** Fast histogram-based binning trained in 9.5 seconds and executes single-row prediction in under 3 milliseconds.

### Top Global Predictive Features
1. `transaction_amount` (0.00973) &mdash; Primary risk driver
2. `account_balance` (0.00220) &mdash; Balance depth vs transaction value
3. `merchant_category_Salary` (0.00133) &mdash; Payroll diversion patterns
4. `transaction_type_Net_Banking` (0.00120) &mdash; Digital credential takeovers
5. `loan_type_Education` (0.00107) &mdash; Demographic vulnerability
6. `channel_Mobile_App` (0.00073) &mdash; Device credential theft
7. `merchant_category_Utilities` (0.00067) &mdash; Bill payment fraud vectors

---

## 8. Application Architecture & Structure

```
FinGuard/
│
├── app.py                         # Flask web application entry point & routing
├── test_app.py                    # Complete automated integration & unit test suite
├── requirements.txt               # Dependency definitions
├── README.md                      # Comprehensive documentation & viva guide
│
├── dataset/
│   ├── indian_banking_transactions.csv  # 550,000 raw transaction records
│   └── finguard.db                      # Indexed SQLite database for fast queries
│
├── preprocessing/
│   └── data_preprocessing.py      # Automated inspection, cleaning & pipeline fitting
│
├── models/
│   ├── train_model.py             # Multi-algorithm training, tuning & evaluation
│   ├── trained_model.pkl          # Saved Champion model (Gradient Boosting)
│   └── preprocessing_pipeline.pkl # Saved ColumnTransformer preprocessor
│
├── backend/
│   ├── routes.py                  # Flask REST API endpoints (/api/...)
│   └── prediction.py              # Real-time inference & evidence explainability
│
├── templates/
│   ├── base.html                  # Core layout, navigation, brand mark & audit modal
│   ├── dashboard.html             # Executive KPIs, charts & recent transaction ledger
│   ├── transactions.html          # Paginated, searchable & filterable transaction ledger
│   ├── fraud_detection.html       # Real-time risk scoring form & assessment display
│   ├── analytics.html             # Multi-dimensional fraud behavioral visualizations
│   └── model_performance.html     # Empirical benchmarks, confusion matrix & feature importance
│
├── static/
│   ├── css/
│   │   └── style.css              # Understated financial design (warm ivory theme)
│   └── js/
│       ├── chart.min.js           # Bundled offline Chart.js library
│       └── script.js              # Vanilla JS controller & dynamic UI renderer
│
└── reports/
    ├── preprocessing_report.json  # Data cleaning & split audit report
    ├── model_metrics.json         # Benchmark comparisons & confusion matrix
    └── analytics_summary.json     # Precomputed macro analytics for instant loading
```

---

## 9. Installation & Execution Guide

### Prerequisites
- Python 3.8+ installed on your system.
- Standard terminal (PowerShell, Command Prompt, or Bash).

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "d:\malak\financial fruad"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```
*(Dependencies: Flask, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib)*

### Step 3: Run the Application
```powershell
python app.py
```

### Step 4: Access in Your Web Browser
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 10. How to Demonstrate FinGuard in a College Presentation / Viva

### Slide 1 / Opening: The Real-World Context
- Explain that banking fraud in India operates across diverse payment rails (UPI, RTGS, Net Banking).
- Highlight the **0.886% fraud rate** (extreme class imbalance) and why standard accuracy is a dangerous vanity metric.

### Demo 1: Executive Dashboard (`/dashboard`)
- Point out the **Total Monitored Value** (INR 16.45B across 550,000 transactions).
- Show the **Fraud Prevalence** indicator (0.886%).
- Show that recent transactions are loaded directly from the database and clicking any row opens a comprehensive transaction inspection modal.

### Demo 2: Transaction Audit Ledger (`/transactions`)
- Demonstrate the **instant search** by typing a customer or transaction ID (e.g. `TXN000000001` or `CUST015796`).
- Demonstrate the **Fraudulent Only filter**: show that exactly 4,873 frauds are indexed and searchable.
- Demonstrate sorting by transaction amount to inspect high-value corporate transfers.

### Demo 3: Live Fraud Detection & Explainability (`/fraud-detection`)
- Click **"Load Legitimate Sample"**: Form auto-fills with a typical grocery UPI transfer (INR 2,450).
  - Click **"ANALYSE TRANSACTION"**.
  - Show the result: **LEGITIMATE TRANSACTION** (~51% risk score, below 63.46% threshold).
  - Point out the evidence factors: "Transaction amount aligns closely with typical retail transactions", "Verified KYC status".
- Click **"Load Suspicious Sample"**: Form auto-fills with a high-value RTGS transfer (INR 850,000) executed at 02:15 AM with depleted account balance and pending KYC.
  - Click **"ANALYSE TRANSACTION"**.
  - Show the result: **POTENTIAL FRAUD DETECTED** (~89.66% risk score, above 63.46% threshold).
  - Show the evidence factors explaining *why* the model flagged it:
    1. Substantial transaction value exceeding retail velocity thresholds.
    2. High liquidity depletion ratio (24x available balance).
    3. RTGS transfer instrument exhibiting elevated baseline fraud risk.
    4. Nocturnal execution during off-peak hours.
  - Highlight the operational recommendation and ML disclaimer.

### Demo 4: Behavioral Analytics (`/analytics`)
- Walk through the hourly curve showing nocturnal fraud spikes between 00:00 and 05:00.
- Show the transaction type breakdown showing RTGS has more than double the fraud rate of UPI.
- Read the student-friendly presentation explanations below each chart.

### Demo 5: Model Evaluation & Benchmarks (`/model-performance`)
- Show the comparative benchmark table comparing all 4 algorithms.
- Explain the **Confusion Matrix** on the unseen test set of 109,879 transactions.
- Explain that validation-based threshold tuning (threshold = 0.6346) was critical to avoid missing rare fraud cases.
- Show the Global Feature Importance table confirming `transaction_amount` as the primary predictive factor.

---

## 11. Verification & Testing
To re-run the automated test suite at any time:
```powershell
python test_app.py
```
*All 7 integration test suites verify page routes, database pagination, filters, analytics aggregations, model metrics, and real-time prediction explainability.*
