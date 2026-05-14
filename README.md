# Graph-Based Adaptive Phishing Detection & Log Anomaly Detection Framework

**Group 19-10 | ITER, SOA University, Bhubaneswar**
**Supervised by:** Dr. Mohammad Maksood Akhtar

---

## Team Members

| Name | Role |
|------|------|
| Shubhrajit Malla | ML Pipeline, Graph Construction, Adaptive Retraining, Evaluation |
| Abhishek Nayak | Data Preprocessing, Feature Engineering |
| Nikhil Kumar Mahato | Log Anomaly Detection, Model Training |
| Sidhartha Ranjan Jena | Dashboard Development, Testing |
| Amiya Ranjan Mallick | Testing |

---

## Project Overview

A multi-phase cybersecurity framework combining:
- **Phishing URL Detection** using Random Forest on 16 structural URL features with graph-based enrichment
- **Log Anomaly Detection** using Isolation Forest on server access logs
- **Adaptive Retraining** that automatically monitors model performance and retrains when F1 drops below threshold
- **Real-Time Dashboard** built with Streamlit featuring a 4-layer detection system
- **CICIDS2017 Validation** on real network attack data

---

## Model Performance

### Phishing URL Detection (LegitPhish Dataset — 100,873 URLs)

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 99.96% | 0.9996 | 0.9999 |
| SVM | 99.94% | 0.9995 | — |

### Log Anomaly Detection (CICIDS2017 — 100,000 Flows)

| Method | Detection Rate | F1 Score |
|--------|---------------|----------|
| Isolation Forest (unsupervised) | 44.9% | 0.4491 |
| Random Forest (supervised) | 99.6% | 0.9815 |

### Per Attack Type (CICIDS2017)

| Attack Type | IF Detection | RF Detection |
|-------------|-------------|-------------|
| DoS Hulk (high volume) | 47.5% | 99.7% |
| DoS slowloris (slow rate) | 34.4% | 99.8% |
| DoS Slowhttptest (slow rate) | 44.8% | 99.9% |

---

## Project Structure

```
phishing-detection-framework/
├── dashboard/
│   └── app.py                          # Streamlit dashboard (4-layer detection)
├── data/
│   ├── raw/
│   │   └── url_features_cleaned.csv    # LegitPhish dataset (100,873 URLs)
│   ├── processed/
│   │   ├── feature_cols.npy            # 16 training feature names
│   │   ├── X_test_scaled.npy           # Scaled test features
│   │   ├── y_test.npy                  # Test labels
│   │   ├── log_features_scored.csv     # Isolation Forest results
│   │   └── cicids_top_anomalies.csv    # CICIDS2017 top anomalous flows
│   └── logs/
│       └── synthetic.log               # Synthetic server access logs
├── models/
│   ├── random_forest.pkl               # Trained Random Forest (16 features)
│   ├── svm_model.pkl                   # Trained SVM
│   ├── scaler.pkl                      # StandardScaler (16 features)
│   ├── isolation_forest.pkl            # Isolation Forest for log anomaly
│   ├── scaler_log.pkl                  # Log feature scaler
│   ├── phishing_graph.gml              # URL-Domain NetworkX graph
│   └── rf_archived_*.pkl               # Versioned model backups
├── notebooks/
│   ├── 01_preprocessing.ipynb          # Phase 1 - Data cleaning
│   ├── 02_preprocessing.ipynb          # Phase 2 - Feature engineering
│   ├── 03_graph_construction.ipynb     # Phase 3 - Graph building
│   ├── 04_clustering_anomaly.ipynb     # Phase 4 - Log anomaly detection
│   ├── 04b_cicids_analysis.ipynb       # Phase 4b - CICIDS2017 validation
│   ├── 05_model_training.ipynb         # Phase 5 - RF and SVM training
│   ├── 06_adaptive_learning.ipynb      # Phase 6 - Adaptive retraining
│   └── 08_testing.ipynb               # Phase 8 - Automated testing
├── reports/
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   ├── retraining_history.png
│   ├── cicids_anomaly_results.png
│   └── phase*_report.json
├── src/
│   └── adaptive_learner.py             # Adaptive retraining module
├── tests/
│   └── test_pipeline.py                # 19/19 pytest test suite
├── .github/
│   └── workflows/
│       └── weekly_retrain.yml          # GitHub Actions weekly retraining
├── requirements.txt
└── README.md
```

---

## Detection System — 4 Layers

The URL Checker uses a 4-layer hybrid detection approach:

```
Input URL
    │
    ▼
Layer 1: Dataset Lookup
    Is URL in 100,873 training URLs? → Use known label (99% confidence)
    │
    ▼
Layer 2: Whitelist
    Is domain a known safe site? (google.com, github.com etc.) → LEGITIMATE
    │
    ▼
Layer 3: Score-Based (12 signals, max 180 points, threshold = 25)
    IP address (+30), @ symbol (+25), suspicious TLD (+20),
    brand impersonation (+20), suspicious extension (+20),
    no HTTPS (+15), phishing keyword in path (+15),
    hyphen in domain (+10), many subdomains (+10),
    long URL (+5), many dots (+5), high numeric ratio (+5)
    │
    ▼
Layer 4: ML Model
    Random Forest on 16 URL structural features → Fallback decision
```

---

## Datasets

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| LegitPhish | Phishing URL classification | 100,873 URLs | Public |
| CICIDS2017 | Network intrusion detection validation | 286,832 flows | Canadian Institute for Cybersecurity |

**Label Convention:** `0.0 = Phishing` | `1.0 = Legitimate`

---

## Features Used (16 URL Structural Features)

| Feature | Description |
|---------|-------------|
| url_length | Total URL character count |
| has_ip_address | IP address instead of domain name |
| dot_count | Number of dots in URL |
| https_flag | Has HTTPS (1) or HTTP (0) |
| url_entropy | Shannon entropy of URL string |
| token_count | Number of URL tokens |
| subdomain_count | Number of subdomains |
| query_param_count | Number of query parameters |
| tld_length | Length of top-level domain |
| path_length | Length of URL path |
| has_hyphen_in_domain | Hyphen present in domain |
| number_of_digits | Count of digits in URL |
| tld_popularity | Common TLD (.com/.org/.edu) |
| suspicious_file_extension | .exe/.php/.js/.sh/.bat |
| domain_name_length | Length of domain name |
| percentage_numeric_chars | Ratio of digits to total characters |

---

## Graph Construction (Phase 3)

- **Type:** URL-to-Domain directed graph
- **Nodes:** URL nodes + Domain nodes
- **Edges:** One directed edge per URL pointing to its domain
- **Key Finding:** Legitimate domains host avg 121 URLs vs phishing domains avg 1.6 URLs
- **Graph Feature:** `domain_url_count` ranked 7th in feature importance

---

## Adaptive Retraining (Phase 6)

- **Threshold:** F1 macro average < 0.90 triggers retraining
- **Total checks run:** 17
- **Retraining events:** 3
- **Archived backups:** 3 versioned model files
- **Automation:** APScheduler runs weekly check automatically when dashboard is live
- **GitHub Actions:** Weekly automated retraining via `.github/workflows/weekly_retrain.yml`

---

## Installation

```bash
# Clone the repository
git clone https://github.com/MallaShubhrajit/Phishing_Frame.git
cd Phishing_Frame

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard pages:
- **URL Checker** — Real-time phishing detection with 4-layer system
- **Log Anomaly Viewer** — Synthetic logs + CICIDS2017 real attack results
- **Model Performance** — ROC curves, confusion matrices, retraining history

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

**Result: 19/19 tests passing**

Test coverage:
- Model loading (RF, SVM, Scaler, Isolation Forest)
- Data file integrity (16 features, correct array shapes)
- Prediction validity (binary labels, probability sums)
- F1 score above 0.90 threshold
- Log anomaly detection (3 known attackers detected)
- Adaptive retraining log validation

---

## Running Adaptive Retraining Manually

```bash
python src/adaptive_learner.py
```

---

## Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.13 |
| ML | Scikit-learn, imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Graph | NetworkX |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Scheduling | APScheduler |
| Serialization | Joblib |
| Testing | Pytest |
| CI/CD | GitHub Actions |

---

## Key Findings

1. **Random Forest achieves 99.96% accuracy** on 100,873 phishing URLs with only 8 misclassifications out of 20,175 test samples
2. **Graph features partially useful** — `domain_url_count` ranked 7th but PageRank/degree metrics were identical for all URL nodes (computed at wrong node level)
3. **Slow-rate DoS attacks evade unsupervised detection** — Isolation Forest detects only 34-47% of slowloris/Slowhttptest attacks vs 99%+ with supervised Random Forest
4. **Adaptive retraining works** — Model F1 monitored weekly, automatically retrains and archives old version when performance degrades

---

## License

MIT License — Free to use for academic and research purposes.

---

## Citation

If you use this work please cite:

```
Malla S., Nayak A., Mahato N.K., Jena S.R., Mallick A.R.
"Graph-Based Adaptive Phishing Detection and Log Anomaly Detection Framework"
ITER, SOA University, Bhubaneswar, 2026
Group 19-10
```
