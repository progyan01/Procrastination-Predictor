import sqlite3
import pickle
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features and labels"))
# pyrefly: ignore [missing-import]
from weak_labels import build_training_dataset

DB_PATH         = "data/raw/activity.db"
MODEL_PATH      = "src/model/model.pkl"
N_SPLITS        = 5
NUDGE_THRESHOLD = 0.6
FEATURES = [
    "time_since_break",
    "switch_freq_10m",
    "productive_ratio_30m",
    "distracting_ratio_30m",
    "task_streak_seconds",
    "hour_of_day",
    "day_of_week",
]


def load_dataset(conn):
    rows = build_training_dataset(conn)
    df = pd.DataFrame(rows)
    return df[FEATURES], df["label"]


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Evaluate predictions at a specific probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def train(X, y):
    n_positive = int(y.sum())

    # need enough positive samples for stratified CV
    if n_positive < 3:
        print("  Too few positive samples for cross-validation — training without calibration.")
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X, y)
        return model

    n_splits = min(N_SPLITS, n_positive)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_f1s = []
    fold_thresh = []
    oof_true, oof_pred, oof_proba = [], [], []

    print(f"{'='*60}")
    print(f"  {n_splits}-Fold Stratified Cross-Validation")
    print(f"{'='*60}\n")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        rf.fit(X_train, y_train)

        y_pred  = rf.predict(X_test)
        y_prob  = rf.predict_proba(X_test)[:, 1]

        f1_def = f1_score(y_test, y_pred, zero_division=0)
        t_m    = evaluate_at_threshold(y_test, y_prob, NUDGE_THRESHOLD)

        fold_f1s.append(f1_def)
        fold_thresh.append(t_m)
        oof_true.extend(y_test.tolist())
        oof_pred.extend(y_pred.tolist())
        oof_proba.extend(y_prob.tolist())

        print(f"  Fold {fold}: F1={f1_def:.4f}  |"
              f"  @{NUDGE_THRESHOLD}: P={t_m['precision']:.3f}"
              f" R={t_m['recall']:.3f} F1={t_m['f1']:.3f}")

    print(f"\n  Mean F1 (default):     {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  Mean @{NUDGE_THRESHOLD} threshold:  "
          f"P={np.mean([m['precision'] for m in fold_thresh]):.3f}  "
          f"R={np.mean([m['recall'] for m in fold_thresh]):.3f}  "
          f"F1={np.mean([m['f1'] for m in fold_thresh]):.3f}")

    print("\n  Out-of-fold classification report:")
    print(classification_report(oof_true, oof_pred, zero_division=0,
                                target_names=["focused", "procrastinating"]))

    # final model: calibrated RF trained via internal CV on all data
    print(f"  Training final calibrated model on all {len(X)} samples...")
    base_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    model = CalibratedClassifierCV(base_rf, method="sigmoid", cv=n_splits)
    model.fit(X, y)

    print("  Done — CalibratedClassifierCV(RandomForest, method='sigmoid')\n")
    return model


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    X, y = load_dataset(conn)
    conn.close()

    print(f"\nDataset: {len(X)} rows, {int(y.sum())} positive ({y.mean():.1%})\n")
    model = train(X, y)
    save_model(model)
    print(f"Model saved to {MODEL_PATH}")
