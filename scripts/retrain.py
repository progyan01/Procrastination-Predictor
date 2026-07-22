import sqlite3
import pickle
import shutil
import os
import sys
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features and labels"))
# pyrefly: ignore [missing-import]
from weak_labels import build_training_dataset
# pyrefly: ignore [missing-import]
from feature_engineering import compute_features

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DB_PATH    = os.path.join(BASE_DIR, "data", "raw", "activity.db")
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "model.pkl")
PREV_PATH  = os.path.join(BASE_DIR, "src", "model", "model_prev.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "data", "retrain.log")

MIN_SAMPLES        = 50    # refuse to train below this many labeled rows
REGRESSION_TOL     = 0.05  # allow up to 5pp F1 drop before rejecting new model
N_SPLITS           = 5     # folds for cross-validation
NUDGE_THRESHOLD    = 0.6   # operating threshold for the nudge system
RANDOM_STATE       = 42
NUDGE_SAMPLE_WEIGHT = 3.0  # confirmed nudge responses count 3x vs weak labels
NUDGE_RETENTION    = 14 * 86400  # only use nudges within raw-data window (14 days)

FEATURES = [
    "time_since_break",
    "switch_freq_10m",
    "productive_ratio_30m",
    "distracting_ratio_30m",
    "task_streak_seconds",
    "hour_of_day",
    "day_of_week",
]

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def f1(model, X_test, y_test):
    return f1_score(y_test, model.predict(X_test), zero_division=0)


def load_nudge_feedback(conn):
    cutoff = time.time() - NUDGE_RETENTION
    rows = conn.execute(
        """
        SELECT ts, was_helpful FROM nudge_log
        WHERE was_helpful IS NOT NULL
          AND ts >= ?
        ORDER BY ts
        """,
        (cutoff,)
    ).fetchall()

    if not rows:
        return pd.DataFrame()

    result = []
    for ts, was_helpful in rows:
        try:
            features = compute_features(conn, ts)
            features["label"] = int(was_helpful)
            result.append(features)
        except Exception as e:
            log.warning(f"Skipping nudge at ts={ts}: {e}")

    return pd.DataFrame(result) if result else pd.DataFrame()


def log_to_db(conn, n_samples, n_positive, new_f1, old_f1, accepted, note):
    conn.execute(
        """
        INSERT INTO retrain_log (ts, sample_count, label_positive, new_f1, old_f1, accepted, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (int(time.time()), n_samples, n_positive, new_f1, old_f1, int(accepted), note),
    )
    conn.commit()


def run():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            ts              INTEGER PRIMARY KEY,
            sample_count    INTEGER,
            label_positive  INTEGER,
            new_f1          REAL,
            old_f1          REAL,
            accepted        INTEGER,
            note            TEXT
        )
    """)
    conn.commit()

    log.info("--- retrain started ---")

    # ── weak-labeled dataset ────────────────────────────────────────────────
    rows = build_training_dataset(conn)
    df_weak = pd.DataFrame(rows)
    n_samples  = len(df_weak)
    n_positive = int(df_weak["label"].sum()) if not df_weak.empty else 0

    if df_weak.empty or n_samples < MIN_SAMPLES:
        note = f"only {n_samples} samples — need {MIN_SAMPLES}"
        log.info(f"Skipping: {note}")
        log_to_db(conn, n_samples, n_positive, 0.0, 0.0, False, note)
        conn.close()
        return

    log.info(f"Weak-label dataset: {n_samples} rows, {n_positive} positive ({n_positive/n_samples:.1%})")

    # ── nudge feedback (confirmed human responses) ──────────────────────────
    df_nudge = load_nudge_feedback(conn)
    if df_nudge.empty:
        log.info("No nudge feedback available — using weak labels only.")
        df = df_weak
        weights = None
    else:
        n_nudge = len(df_nudge)
        n_nudge_pos = int(df_nudge["label"].sum())
        log.info(f"Nudge feedback: {n_nudge} responses ({n_nudge_pos} positive), weight={NUDGE_SAMPLE_WEIGHT}x")
        df = pd.concat([df_weak, df_nudge], ignore_index=True)
        weights = np.concatenate([
            np.ones(len(df_weak)),
            np.full(len(df_nudge), NUDGE_SAMPLE_WEIGHT)
        ])

    X = df[FEATURES]
    y = df["label"]

    # indices for weak-label and nudge rows
    weak_idx   = np.arange(len(df_weak))
    nudge_idx  = np.arange(len(df_weak), len(df))

    # determine CV folds (need at least 2 positive samples per fold)
    n_positive_weak = int(y.iloc[weak_idx].sum())
    n_splits = min(N_SPLITS, n_positive_weak)
    if n_splits < 2:
        n_splits = 2

    # load old model before CV loop so we can compare on each fold
    old_model = None
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as fp:
                old_model = pickle.load(fp)
        except Exception as e:
            log.warning(f"Could not load old model: {e}")

    # cross-validate: split weak labels into folds, nudge rows always in train
    new_f1_scores = []
    old_f1_scores = []
    oof_true, oof_pred = [], []

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(skf.split(weak_idx, y.iloc[weak_idx]))
    except ValueError:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(weak_idx))

    for fold, (train_pos, test_pos) in enumerate(splits, 1):
        train_weak = weak_idx[train_pos]
        test_weak  = weak_idx[test_pos]
        fold_train = np.concatenate([train_weak, nudge_idx])
        fold_test  = test_weak

        X_tr, y_tr = X.iloc[fold_train], y.iloc[fold_train]
        X_te, y_te = X.iloc[fold_test],  y.iloc[fold_test]
        w_tr       = weights[fold_train] if weights is not None else None

        fold_model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        )
        fold_model.fit(X_tr, y_tr, sample_weight=w_tr)
        new_f1_scores.append(f1(fold_model, X_te, y_te))
        oof_true.extend(y_te.tolist())
        oof_pred.extend(fold_model.predict(X_te).tolist())

        if old_model is not None:
            try:
                old_f1_scores.append(f1(old_model, X_te, y_te))
            except Exception:
                pass

    new_f1 = float(np.mean(new_f1_scores))
    old_f1 = float(np.mean(old_f1_scores)) if old_f1_scores else 0.0

    log.info(f"New model {n_splits}-fold CV F1: {new_f1:.4f} (folds: {[round(s, 4) for s in new_f1_scores]})")
    log.info("\n" + classification_report(oof_true, oof_pred, zero_division=0))
    if old_f1_scores:
        log.info(f"Old model {n_splits}-fold CV F1: {old_f1:.4f}")

    # report precision/recall at nudge operating threshold (last fold)
    y_proba = fold_model.predict_proba(X_te)[:, 1]
    y_at_thresh = (y_proba >= NUDGE_THRESHOLD).astype(int)
    log.info(f"@{NUDGE_THRESHOLD} threshold (last fold): "
             f"P={precision_score(y_te, y_at_thresh, zero_division=0):.4f} "
             f"R={recall_score(y_te, y_at_thresh, zero_division=0):.4f}")

    # regression guard: compare CV F1 scores
    if new_f1 < old_f1 - REGRESSION_TOL:
        note = f"rejected — CV F1 {new_f1:.4f} < old CV F1 {old_f1:.4f} - {REGRESSION_TOL}"
        log.warning(note)
        log_to_db(conn, n_samples, n_positive, new_f1, old_f1, False, note)
        conn.close()
        return

    # train final calibrated model on ALL data
    log.info("Training final calibrated model on all data...")
    base_rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    try:
        final_model = CalibratedClassifierCV(base_rf, method="sigmoid", cv=n_splits)
        final_model.fit(X, y, sample_weight=weights)
    except ValueError:
        log.warning("Calibration failed — saving uncalibrated model")
        final_model = RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
        )
        final_model.fit(X, y, sample_weight=weights)

    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, PREV_PATH)
        log.info(f"Previous model backed up to {PREV_PATH}")

    with open(MODEL_PATH, "wb") as fp:
        pickle.dump(final_model, fp)

    note = "accepted (calibrated)" if old_model is not None else "first model saved (calibrated)"
    log.info(f"Model saved — {note}")
    log_to_db(conn, n_samples, n_positive, new_f1, old_f1, True, note)

    conn.close()
    log.info("--- retrain complete ---")


if __name__ == "__main__":
    run()
