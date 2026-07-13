import sqlite3
import pickle
import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features and labels"))
# pyrefly: ignore [missing-import]
from weak_labels import build_training_dataset

DB_PATH    = "data/raw/activity.db"
MODEL_PATH = "src/model/model.pkl"
TEST_SIZE  = 0.2
FEATURES   = [
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

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))
    return model

def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    X, y = load_dataset(conn)
    conn.close()

    model = train(X, y)
    save_model(model)
    print(f"Model saved to {MODEL_PATH}")
