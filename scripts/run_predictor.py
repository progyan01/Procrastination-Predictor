import sqlite3
import pickle
import time
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features and labels"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "dashboard"))
# pyrefly: ignore [missing-import]
from feature_engineering import compute_features
# pyrefly: ignore [missing-import]
from notifier import send_nudge, init_nudge_log

DB_PATH         = "data/raw/activity.db"
MODEL_PATH      = "src/model/model.pkl"
NUDGE_THRESHOLD = 0.6
POLL_INTERVAL   = 120   # seconds between prediction checks (2 min)
SWITCH_FREQ_WEIGHT   = 0.25
SWITCH_FREQ_HIGH     = 20   # switches/10min that counts as "fully active" distraction

FEATURES        = [
    "time_since_break",
    "switch_freq_10m",
    "productive_ratio_30m",
    "distracting_ratio_30m",
    "task_streak_seconds",
    "hour_of_day",
    "day_of_week",
]

def load_model():
    #try to load a trained model — fall back to heuristic rules if none exists yet
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded — running in ML mode.")
        return model, "ml"
    except FileNotFoundError:
        print("No model found — running in heuristic (cold-start) mode.")
        return None, "heuristic"

def heuristic_score(features):
    #rule-based fallback for when there isn't enough data to train on yet
    #high distracting ratio AND a long unbroken stretch without a break = likely sliding
    if (features["distracting_ratio_30m"] > 0.5
            and features["time_since_break"] > 1200):   # 20 min no break
        return 0.8
    return 0.2

def get_score(model, mode, features):
    if mode == "ml":
        X = pd.DataFrame([features])[FEATURES]
        ml_prob = model.predict_proba(X)[0][1]
        if SWITCH_FREQ_WEIGHT > 0:
            raw_switchFreq = features["switch_freq_10m"]
            score = (1 - SWITCH_FREQ_WEIGHT) * ml_prob + SWITCH_FREQ_WEIGHT * (min(raw_switchFreq / SWITCH_FREQ_HIGH, 1.0))
            return score
        return ml_prob
    return heuristic_score(features)

def main():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    init_nudge_log(conn)   #ensure nudge_log table exists before the loop starts

    model, mode = load_model()
    print(f"Predictor running. Checking every {POLL_INTERVAL // 60} min. Press Ctrl+C to stop.")

    try:
        while True:
            try:
                features = compute_features(conn)
                score = get_score(model, mode, features)
                print(f"[{time.strftime('%H:%M:%S')}] score={score:.2%}  mode={mode}")

                if score >= NUDGE_THRESHOLD:
                    fired = send_nudge(score, model_version=mode)
                    if fired:
                        print("  → nudge fired")

            except Exception as e:
                #don't let a single bad DB read kill the whole loop
                print(f"  [warn] prediction tick failed: {e}")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nPredictor stopped.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
