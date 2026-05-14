"""
test_pipeline.py
----------------
Automated test suite for the Phishing Detection Framework.
Run with: pytest tests/test_pipeline.py -v

Tests verify every major component of the pipeline works correctly.
Group 19-10 | SOA University ITER
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import re
from urllib.parse import urlparse
from sklearn.metrics import f1_score

# ── Base directory ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(relative):
    return os.path.join(BASE, relative.replace('../', ''))


# ════════════════════════════════════════════════════════════════
# TEST 1 — Models load correctly
# ════════════════════════════════════════════════════════════════
class TestModelsLoad:

    def test_random_forest_loads(self):
        """Random Forest model file exists and loads without error."""
        model_path = path('models/random_forest.pkl')
        assert os.path.exists(model_path), \
            "random_forest.pkl not found in models/"
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, 'predict'), \
            "Loaded object does not have predict method"

    def test_svm_loads(self):
        """SVM model file exists and loads without error."""
        svm_path = path('models/svm_model.pkl')
        assert os.path.exists(svm_path), \
            "svm_model.pkl not found in models/"
        svm = joblib.load(svm_path)
        assert svm is not None
        assert hasattr(svm, 'predict')

    def test_scaler_loads(self):
        """StandardScaler loads and has correct number of features."""
        scaler_path = path('models/scaler.pkl')
        assert os.path.exists(scaler_path), \
            "scaler.pkl not found in models/"
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, 'transform')
        assert scaler.n_features_in_ == 21, \
            f"Scaler expects 21 features, got {scaler.n_features_in_}"

    def test_isolation_forest_loads(self):
        """Isolation Forest model loads correctly."""
        iso_path = path('models/isolation_forest.pkl')
        assert os.path.exists(iso_path), \
            "isolation_forest.pkl not found in models/"
        iso = joblib.load(iso_path)
        assert iso is not None
        assert hasattr(iso, 'predict')


# ════════════════════════════════════════════════════════════════
# TEST 2 — Data files exist and are correct
# ════════════════════════════════════════════════════════════════
class TestDataFiles:

    def test_feature_cols_has_21_features(self):
        """Feature column file contains exactly 21 feature names."""
        feat_path = path(
            'data/processed/feature_cols_with_graph.npy'
        )
        assert os.path.exists(feat_path)
        cols = np.load(feat_path, allow_pickle=True).tolist()
        assert len(cols) == 21, \
            f"Expected 21 features, found {len(cols)}"

    def test_graph_features_present(self):
        """Graph-derived features are present in feature list."""
        feat_path = path(
            'data/processed/feature_cols_with_graph.npy'
        )
        cols = np.load(feat_path, allow_pickle=True).tolist()
        graph_cols = [
            'pagerank', 'in_degree',
            'out_degree', 'domain_url_count'
        ]
        for gc in graph_cols:
            assert gc in cols, \
                f"Graph feature '{gc}' missing from feature list"

    def test_test_arrays_correct_shape(self):
        """Test arrays exist and have matching shapes."""
        X_test = np.load(path('data/processed/X_test_scaled.npy'))
        y_test = np.load(path('data/processed/y_test.npy'))
        assert X_test.shape[1] == 21, \
            f"X_test has {X_test.shape[1]} cols, expected 21"
        assert len(X_test) == len(y_test), \
            "X_test and y_test have different lengths"

    def test_log_features_scored_exists(self):
        """Scored log features file exists with required columns."""
        log_path = path('data/processed/log_features_scored.csv')
        assert os.path.exists(log_path), \
            "log_features_scored.csv not found — run Phase 4"
        df = pd.read_csv(log_path)
        required = ['ip', 'request_count', 'error_rate',
                    'post_ratio', 'anomaly_label']
        for col in required:
            assert col in df.columns, \
                f"Column '{col}' missing from log_features_scored.csv"

    def test_graph_file_exists(self):
        """NetworkX graph file saved by Phase 3 exists."""
        graph_path = path('models/phishing_graph.gml')
        assert os.path.exists(graph_path), \
            "phishing_graph.gml not found — run Phase 3"


# ════════════════════════════════════════════════════════════════
# TEST 3 — Prediction output format
# ════════════════════════════════════════════════════════════════
class TestPredictions:

    def setup_method(self):
        self.model  = joblib.load(path('models/random_forest.pkl'))
        self.scaler = joblib.load(path('models/scaler.pkl'))
        self.X_test = np.load(
            path('data/processed/X_test_scaled.npy')
        )
        self.y_test = np.load(
            path('data/processed/y_test.npy')
        )

    def test_prediction_returns_binary(self):
        """Model predictions are binary — only 0.0 or 1.0."""
        preds = self.model.predict(self.X_test[:50])
        unique_vals = set(preds)
        assert unique_vals.issubset({0.0, 1.0}), \
            f"Unexpected prediction values: {unique_vals}"

    def test_prediction_count_matches_input(self):
        """Number of predictions matches number of input samples."""
        n_samples = 100
        preds = self.model.predict(self.X_test[:n_samples])
        assert len(preds) == n_samples

    def test_probability_output_valid(self):
        """Probability scores sum to 1.0 for each sample."""
        probas = self.model.predict_proba(self.X_test[:20])
        assert probas.shape[1] == 2, \
            "predict_proba should return 2 columns"
        row_sums = probas.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            "Probability rows do not sum to 1.0"

    def test_probabilities_between_0_and_1(self):
        """All probability values are between 0 and 1."""
        probas = self.model.predict_proba(self.X_test[:50])
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0


# ════════════════════════════════════════════════════════════════
# TEST 4 — F1 score above threshold
# ════════════════════════════════════════════════════════════════
class TestModelPerformance:

    def setup_method(self):
        self.model  = joblib.load(path('models/random_forest.pkl'))
        self.X_test = np.load(
            path('data/processed/X_test_scaled.npy')
        )
        self.y_test = np.load(
            path('data/processed/y_test.npy')
        )

    def test_f1_above_threshold(self):
        """Random Forest F1 score on test set is above 0.90."""
        preds      = self.model.predict(self.X_test)
        f1         = f1_score(self.y_test, preds)
        assert f1 >= 0.90, \
            f"F1 score {f1:.4f} is below threshold 0.90"

    def test_recall_above_threshold(self):
        """Recall is above 0.88 — model catches most phishing URLs."""
        from sklearn.metrics import recall_score
        preds  = self.model.predict(self.X_test)
        recall = recall_score(self.y_test, preds)
        assert recall >= 0.88, \
            f"Recall {recall:.4f} is below 0.88"

    def test_precision_above_threshold(self):
        """Precision above 0.88 — few legitimate URLs wrongly flagged."""
        from sklearn.metrics import precision_score
        preds     = self.model.predict(self.X_test)
        precision = precision_score(self.y_test, preds)
        assert precision >= 0.88, \
            f"Precision {precision:.4f} is below 0.88"


# ════════════════════════════════════════════════════════════════
# TEST 5 — Isolation Forest detects known anomalies
# ════════════════════════════════════════════════════════════════
class TestLogAnomalyDetection:

    def setup_method(self):
        self.log_df = pd.read_csv(
            path('data/processed/log_features_scored.csv')
        )

    def test_anomaly_column_exists(self):
        """anomaly_label column exists in scored log file."""
        assert 'anomaly_label' in self.log_df.columns

    def test_anomaly_labels_valid(self):
        """anomaly_label contains only Normal or ANOMALY."""
        valid  = {'Normal', 'ANOMALY'}
        actual = set(self.log_df['anomaly_label'].unique())
        assert actual.issubset(valid), \
            f"Unexpected labels found: {actual - valid}"

    def test_known_attackers_detected(self):
        """All three injected attack IPs are flagged as ANOMALY."""
        known = ['10.0.0.99', '185.220.101.45', '172.16.5.200']
        for ip in known:
            rows = self.log_df[self.log_df['ip'] == ip]
            assert len(rows) > 0, \
                f"Attack IP {ip} not found in log data"
            label = rows.iloc[0]['anomaly_label']
            assert label == 'ANOMALY', \
                f"Attack IP {ip} was not detected — got '{label}'"

    def test_anomaly_scores_normalised(self):
        """Normalised anomaly scores are between 0 and 1."""
        if 'anomaly_score_norm' in self.log_df.columns:
            scores = self.log_df['anomaly_score_norm']
            assert scores.min() >= 0.0
            assert scores.max() <= 1.0


# ════════════════════════════════════════════════════════════════
# TEST 6 — Adaptive retraining log
# ════════════════════════════════════════════════════════════════
class TestAdaptiveRetraining:

    def setup_method(self):
        log_path = path('reports/retraining_log.json')
        assert os.path.exists(log_path), \
            "retraining_log.json not found — run Phase 6"
        with open(log_path) as f:
            self.log = json.load(f)

    def test_log_is_list(self):
        """Retraining log is a list of entries."""
        assert isinstance(self.log, list)

    def test_log_has_entries(self):
        """Retraining log has at least one entry."""
        assert len(self.log) >= 1, \
            "Retraining log is empty — run Phase 6 notebook"

    def test_log_entry_has_required_fields(self):
        """Each log entry contains all required fields."""
        required = [
            'timestamp', 'f1_before',
            'retrained', 'threshold'
        ]
        for i, entry in enumerate(self.log):
            for field in required:
                assert field in entry, \
                    f"Entry {i} missing field '{field}'"

    def test_f1_values_valid(self):
        """F1 values in log are between 0 and 1."""
        for entry in self.log:
            f1 = entry['f1_before']
            assert 0.0 <= f1 <= 1.0, \
                f"Invalid F1 value in log: {f1}"

    def test_threshold_consistent(self):
        """All log entries use the same threshold value."""
        thresholds = [e['threshold'] for e in self.log]
        assert len(set(thresholds)) == 1, \
            "Inconsistent thresholds found in log"