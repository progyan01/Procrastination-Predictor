import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import compute_features

LOOKBACK        = 1800   #seconds to look back for productive activity (30 min)
LOOKAHEAD       = 900    #seconds to look forward for distracting spiral (15 min)
PRODUCTIVE_MIN  = 0.5    #fraction of lookback window that must be productive
DISTRACTING_MIN = 0.6    #fraction of lookahead window that must be distracting

def productive_before(conn, ts):
    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN category = 'productive' THEN 1 ELSE 0 END) AS productive
        FROM window_events WHERE ts > ? AND ts <= ?
        """,
        (ts - LOOKBACK, ts)
    ).fetchone()

    total, productive = row
    if not total:
        return False
    return (productive / total) >= PRODUCTIVE_MIN

def distracting_after(conn, ts):
    row = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN category = 'distracting' THEN 1 ELSE 0 END) AS distracting
        FROM window_events WHERE ts > ? AND ts <= ?
        """,
        (ts, ts + LOOKAHEAD)
    ).fetchone()

    total, distracting = row
    if not total:
        return False
    return (distracting / total) >= DISTRACTING_MIN

def label_events(conn):
    #stamp every window event with 1 if it marks the start of a procrastination spiral, 0 otherwise
    timestamps = [
        row[0] for row in conn.execute("SELECT ts FROM window_events ORDER BY ts").fetchall()
    ]

    results = []
    for ts in timestamps:
        label = 1 if (productive_before(conn, ts) and distracting_after(conn, ts)) else 0
        results.append({"ts": ts, "label": label})

    return results

def build_training_dataset(conn):
    #pair each labeled event with its feature vector
    labeled = label_events(conn)

    rows = []
    for entry in labeled:
        features = compute_features(conn, entry["ts"])
        features["label"] = entry["label"]
        rows.append(features)

    return rows
