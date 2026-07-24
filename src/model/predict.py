import sqlite3
import pickle
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.features_and_labels.feature_engineering import compute_features, FEATURES

DB_PATH         = "data/raw/activity.db"
MODEL_PATH      = "src/model/model.pkl"
NUDGE_THRESHOLD = 0.6

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_proba(conn, model):
    features = compute_features(conn)
    X = pd.DataFrame([features])[FEATURES]
    return model.predict_proba(X)[0][1]  #probability of class 1

def should_nudge(conn, model):
    return predict_proba(conn, model) >= NUDGE_THRESHOLD

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    model = load_model()
    prob = predict_proba(conn, model)
    conn.close()

    print(f"Procrastination probability: {prob:.2%}")
    if prob >= NUDGE_THRESHOLD:
        print("above threshold — nudge should fire")
    else:
        print("below threshold — looking focused")
