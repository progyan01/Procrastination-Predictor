import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import compute_features

LOOKBACK        = 1800   #seconds to look back for productive activity (30 min)
LOOKAHEAD       = 900    #seconds to look forward for distracting spiral (15 min)
PRODUCTIVE_MIN  = 0.5    #fraction of lookback window that must be productive
DISTRACTING_MIN = 0.6    #fraction of lookahead window that must be distracting

def _duration_ratios(conn, start, end):
    #each event lasts from its ts until the next event's ts
    #use LEAD to compute per-row durations, then sum by category
    row = conn.execute(
        """
        WITH durations AS (
            SELECT category,
                   LEAD(ts) OVER (ORDER BY ts) - ts AS duration
            FROM window_events
            WHERE ts >= ? AND ts < ?
        )
        SELECT
            SUM(CASE WHEN category IN ('productive', 'distracting', 'neutral') THEN duration ELSE 0 END) AS known_total,
            SUM(CASE WHEN category = 'productive'  THEN duration ELSE 0 END) AS productive_time,
            SUM(CASE WHEN category = 'distracting' THEN duration ELSE 0 END) AS distracting_time
        FROM durations
        """,
        (start, end)
    ).fetchone()
    return row  #(known_total, productive_time, distracting_time)

def productive_before(conn, ts):
    known_total, productive_time, _ = _duration_ratios(conn, ts - LOOKBACK, ts)
    if not known_total:
        return False
    return (productive_time / known_total) >= PRODUCTIVE_MIN

def distracting_after(conn, ts):
    known_total, _, distracting_time = _duration_ratios(conn, ts, ts + LOOKAHEAD)
    if not known_total:
        return False
    return (distracting_time / known_total) >= DISTRACTING_MIN

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
