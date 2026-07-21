import sqlite3
import pickle
import shutil
import os
import sys
import logging
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features and labels"))
# pyrefly: ignore [missing-import]
from weak_labels import build_training_dataset

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DB_PATH    = os.path.join(BASE_DIR, "data", "raw", "activity.db")
MODEL_PATH = os.path.join(BASE_DIR, "src", "model", "model.pkl")
PREV_PATH  = os.path.join(BASE_DIR, "src", "model", "model_prev.pkl")
LOG_PATH   = os.path.join(BASE_DIR, "data", "retrain.log")

MIN_SAMPLES      = 50    # refuse to train below this many labeled rows
TEST_SIZE        = 0.2
REGRESSION_TOL   = 0.05  # allow up to 5pp F1 drop before rejecting new model
RANDOM_STATE     = 42

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

    rows = build_training_dataset(conn)
    df = pd.DataFrame(rows)
    n_samples  = len(df)
    n_positive = int(df["label"].sum()) if not df.empty else 0

    if df.empty or n_samples < MIN_SAMPLES:
        note = f"only {n_samples} samples — need {MIN_SAMPLES}"
        log.info(f"Skipping: {note}")
        log_to_db(conn, n_samples, n_positive, 0.0, 0.0, False, note)
        conn.close()
        return

    log.info(f"Dataset: {n_samples} rows, {n_positive} positive ({n_positive/n_samples:.1%})")

    X = df[FEATURES]
    y = df["label"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    new_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    new_model.fit(X_train, y_train)
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
