"""
adaptive_learner.py
Adaptive Retraining Module — Phishing Detection Framework
Label convention: 0.0 = Phishing  |  1.0 = Legitimate
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'random_forest.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'processed',
                           'df_with_graph_features.csv')
FEAT_PATH   = os.path.join(BASE_DIR, 'data', 'processed',
                           'feature_cols.npy')
X_TEST_PATH = os.path.join(BASE_DIR, 'data', 'processed',
                           'X_test_scaled.npy')
Y_TEST_PATH = os.path.join(BASE_DIR, 'data', 'processed',
                           'y_test.npy')
LOG_PATH    = os.path.join(BASE_DIR, 'reports', 'retraining_log.json')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')

F1_THRESHOLD = 0.90


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            return json.load(f)
    return []


def save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w') as f:
        json.dump(log, f, indent=2)


def evaluate_model():
    model  = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    current_f1 = f1_score(y_test, model.predict(X_test),
                          average='macro')
    print(f"Current model F1 (macro) : {current_f1:.4f}")
    return float(current_f1)


def retrain_model():
    print("Retraining model...")
    feat_cols = np.load(FEAT_PATH, allow_pickle=True).tolist()
    print(f"Using {len(feat_cols)} features")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['ClassLabel'])

    X = df[feat_cols].fillna(0).values
    y = df['ClassLabel'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_bal, y_bal = sm.fit_resample(X_train_scaled, y_train)

    new_model = RandomForestClassifier(
        n_estimators      = 200,
        max_depth         = 15,
        min_samples_split = 5,
        min_samples_leaf  = 2,
        class_weight      = 'balanced',
        random_state      = 42,
        n_jobs            = -1
    )
    new_model.fit(X_bal, y_bal)

    new_f1 = f1_score(
        y_test, new_model.predict(X_test_scaled),
        average='macro'
    )
    print(f"New model F1 (macro) : {new_f1:.4f}")

    timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_path = os.path.join(MODELS_DIR,
                                f'rf_archived_{timestamp}.pkl')
    if os.path.exists(MODEL_PATH):
        joblib.dump(joblib.load(MODEL_PATH), archive_path)
        print(f"Old model archived : {archive_path}")

    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(scaler,    SCALER_PATH)
    print(f"New model saved : {MODEL_PATH}")

    return new_f1, archive_path


def run_adaptive_check(threshold=None):
    effective_threshold = threshold if threshold is not None \
                          else F1_THRESHOLD

    print("=" * 55)
    print("ADAPTIVE RETRAINING MODULE")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Threshold : {effective_threshold}")
    print("=" * 55)

    log        = load_log()
    current_f1 = evaluate_model()
    retrained  = False
    new_f1     = None
    archive    = None
    trigger    = None

    if current_f1 < effective_threshold:
        print(f"F1 {current_f1:.4f} BELOW threshold "
              f"{effective_threshold} — retraining...")
        new_f1, archive = retrain_model()
        retrained = True
        trigger   = 'f1_below_threshold'
        print(f"F1 before : {current_f1:.4f}")
        print(f"F1 after  : {new_f1:.4f}")
    else:
        print(f"F1 {current_f1:.4f} ABOVE threshold — no action needed.")

    entry = {
        'timestamp'  : datetime.now().isoformat(),
        'f1_before'  : round(float(current_f1), 4),
        'retrained'  : retrained,
        'f1_after'   : round(float(new_f1), 4) if new_f1 else None,
        'improvement': round(float(new_f1 - current_f1), 4)
                       if new_f1 else None,
        'trigger'    : trigger,
        'threshold'  : effective_threshold,
        'archive'    : archive,
    }

    log.append(entry)
    save_log(log)
    print(f"Log updated — total entries: {len(log)}")
    print("=" * 55)
    return entry


if __name__ == '__main__':
    result = run_adaptive_check()
    print(json.dumps(result, indent=2))