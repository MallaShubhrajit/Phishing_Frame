"""
test_pipeline.py
Automated test suite for Phishing Detection Framework.
Label convention: 0.0 = Phishing  |  1.0 = Legitimate
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def path(rel):
    return os.path.join(BASE, rel)


class TestModelsLoad:

    def test_random_forest_loads(self):
        model = joblib.load(path("models/random_forest.pkl"))
        assert model is not None
        assert hasattr(model, "predict")

    def test_svm_loads(self):
        svm = joblib.load(path("models/svm_model.pkl"))
        assert svm is not None
        assert hasattr(svm, "predict")

    def test_scaler_loads(self):
        scaler = joblib.load(path("models/scaler.pkl"))
        assert hasattr(scaler, "transform")
        assert scaler.n_features_in_ == 16, \
            f"Scaler expects 16 features, got {scaler.n_features_in_}"

    def test_isolation_forest_loads(self):
        iso = joblib.load(path("models/isolation_forest.pkl"))
        assert iso is not None
        assert hasattr(iso, "predict")


class TestDataFiles:

    def test_feature_cols_exists(self):
        feat = np.load(path("data/processed/feature_cols.npy"),
                       allow_pickle=True).tolist()
        assert len(feat) == 16, \
            f"Expected 16 features, found {len(feat)}"

    def test_test_arrays_shape(self):
        X = np.load(path("data/processed/X_test_scaled.npy"))
        y = np.load(path("data/processed/y_test.npy"))
        assert X.shape[1] == 16, \
            f"X_test has {X.shape[1]} cols, expected 16"
        assert len(X) == len(y)

    def test_log_features_scored_exists(self):
        df = pd.read_csv(
            path("data/processed/log_features_scored.csv")
        )
        assert "anomaly_label" in df.columns
        assert len(df) > 0

    def test_graph_file_exists(self):
        assert os.path.exists(path("models/phishing_graph.gml"))


class TestPredictions:

    def setup_method(self):
        self.model  = joblib.load(path("models/random_forest.pkl"))
        self.X_test = np.load(path("data/processed/X_test_scaled.npy"))
        self.y_test = np.load(path("data/processed/y_test.npy"))

    def test_prediction_returns_binary(self):
        preds = self.model.predict(self.X_test[:50])
        assert set(preds).issubset({0.0, 1.0})

    def test_prediction_count_matches(self):
        preds = self.model.predict(self.X_test[:100])
        assert len(preds) == 100

    def test_probability_valid(self):
        probas = self.model.predict_proba(self.X_test[:20])
        assert probas.shape[1] == 2
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)


class TestModelPerformance:

    def setup_method(self):
        self.model  = joblib.load(path("models/random_forest.pkl"))
        self.X_test = np.load(path("data/processed/X_test_scaled.npy"))
        self.y_test = np.load(path("data/processed/y_test.npy"))

    def test_f1_above_threshold(self):
        preds = self.model.predict(self.X_test)
        f1    = f1_score(self.y_test, preds, average="macro")
        assert f1 >= 0.90, f"F1 {f1:.4f} below 0.90"


class TestLogAnomalyDetection:

    def setup_method(self):
        self.df = pd.read_csv(
            path("data/processed/log_features_scored.csv")
        )

    def test_anomaly_column_exists(self):
        assert "anomaly_label" in self.df.columns

    def test_anomaly_labels_valid(self):
        valid = {"Normal", "ANOMALY"}
        assert set(self.df["anomaly_label"].unique()).issubset(valid)

    def test_known_attackers_detected(self):
        known = ["10.0.0.99", "185.220.101.45", "172.16.5.200"]
        for ip in known:
            rows = self.df[self.df["ip"] == ip]
            assert len(rows) > 0, f"{ip} not found"
            assert rows.iloc[0]["anomaly_label"] == "ANOMALY", \
                f"{ip} not detected"


class TestAdaptiveRetraining:

    def setup_method(self):
        log_path = path("reports/retraining_log.json")
        assert os.path.exists(log_path)
        with open(log_path) as f:
            self.log = json.load(f)

    def test_log_has_entries(self):
        assert len(self.log) >= 1

    def test_log_required_fields(self):
        required = ["timestamp", "f1_before", "retrained", "threshold"]
        for entry in self.log:
            for field in required:
                assert field in entry

    def test_f1_values_valid(self):
        for entry in self.log:
            assert 0.0 <= entry["f1_before"] <= 1.0

    def test_thresholds_valid(self):
        thresholds = [e["threshold"] for e in self.log]
        assert all(0.0 < t <= 1.0 for t in thresholds), \
            "Invalid threshold values found in log"
