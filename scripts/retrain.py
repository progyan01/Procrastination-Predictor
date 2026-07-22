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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

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
TEST_SIZE          = 0.2
REGRESSION_TOL     = 0.05  # allow up to 5pp F1 drop before rejecting new model
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

    # split only the weak-label rows for eval; nudge rows go entirely to train
    weak_idx   = np.arange(len(df_weak))
    nudge_idx  = np.arange(len(df_weak), len(df))

    try:
        train_weak, test_weak = train_test_split(
            weak_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y.iloc[weak_idx]
        )
    except ValueError:
        train_weak, test_weak = train_test_split(
            weak_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    train_idx = np.concatenate([train_weak, nudge_idx])  # nudge always in train
    test_idx  = test_weak

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]
    train_weights    = weights[train_idx] if weights is not None else None

    new_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    new_model.fit(X_train, y_train, sample_weight=train_weights)
    new_f1 = f1(new_model, X_test, y_test)
    log.info(f"New model F1: {new_f1:.4f}")
    log.info("\n" + classification_report(y_test, new_model.predict(X_test), zero_division=0))

    old_model = None
    old_f1    = 0.0
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as fp:
            old_model = pickle.load(fp)
        try:
            old_f1 = f1(old_model, X_test, y_test)
            log.info(f"Old model F1 on same split: {old_f1:.4f}")
        except Exception as e:
            log.warning(f"Could not evaluate old model: {e}")

    if new_f1 < old_f1 - REGRESSION_TOL:
        note = f"rejected — new F1 {new_f1:.4f} < old F1 {old_f1:.4f} minus tolerance {REGRESSION_TOL}"
        log.warning(note)
        log_to_db(conn, n_samples, n_positive, new_f1, old_f1, False, note)
        conn.close()
        return

    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, PREV_PATH)
        log.info(f"Previous model backed up to {PREV_PATH}")

    with open(MODEL_PATH, "wb") as fp:
        pickle.dump(new_model, fp)

    note = "accepted" if old_model is not None else "first model saved"
    log.info(f"Model saved — {note}")
    log_to_db(conn, n_samples, n_positive, new_f1, old_f1, True, note)

    conn.close()
    log.info("--- retrain complete ---")


if __name__ == "__main__":
    run()
