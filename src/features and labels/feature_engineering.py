import sqlite3
import time
from datetime import datetime

WINDOW_10MINS = 600    
WINDOW_30MINS = 1800   
STREAK = 200   #max rows to scan when computing streak 

def time_since_break(conn, at_time):
    #find the most recent moment the user went idle before at_time
    row = conn.execute(
        "SELECT ts FROM idle_events WHERE state = 'idle' AND ts <= ? ORDER BY ts DESC LIMIT 1",
        (at_time,)
    ).fetchone()

    if row is None:
        return at_time  #no idle event on record so treat the whole session as one unbroken stretch
    return at_time - row[0]

def switch_freq_10m(conn, at_time):
    #count how many window changes happened in the last 10 minutes as a high switch frequency is a known precursor to distraction 
    row = conn.execute(
        "SELECT COUNT(*) FROM window_events WHERE ts > ? AND ts <= ?",
        (at_time - WINDOW_10MINS, at_time)
    ).fetchone()
    return row[0]

def category_ratios_30m(conn, at_time):
    #compute productive and distracting ratios weighted by time spent, not event count
    #each event lasts from its ts until the next event (via LEAD); the last event extends to at_time
    row = conn.execute(
        """
        WITH durations AS (
            SELECT category,
                   COALESCE(LEAD(ts) OVER (ORDER BY ts), ?) - ts AS duration
            FROM window_events
            WHERE ts > ? AND ts <= ?
        )
        SELECT
            SUM(duration) AS total,
            SUM(CASE WHEN category = 'productive'  THEN duration ELSE 0 END) AS productive,
            SUM(CASE WHEN category = 'distracting' THEN duration ELSE 0 END) AS distracting
        FROM durations
        """,
        (at_time, at_time - WINDOW_30MINS, at_time)
    ).fetchone()

    total, productive, distracting = row
    if not total:
        return 0.0, 0.0     #no events in window — ratios are undefined, default to zero
    return productive / total, distracting / total

def task_streak_seconds(conn, at_time):
    #how long has the user been in the same category without switching
    rows = conn.execute(
        "SELECT ts, category FROM window_events WHERE ts <= ? ORDER BY ts DESC LIMIT ?",
        (at_time, STREAK)
    ).fetchall()

    if not rows:
        return 0.0

    current_category = rows[0][1]
    streak_start_ts = rows[0][0]

    for ts, category in rows[1:]:
        if category != current_category:
            break
        streak_start_ts = ts

    return at_time - streak_start_ts

def compute_features(conn, at_time=None):
    #build the full feature vector for a given moment in time
    #at_time defaults to now so the same function works for live prediction and batch training
    at_time = at_time or time.time()

    dt = datetime.fromtimestamp(at_time)
    productive_ratio, distracting_ratio = category_ratios_30m(conn, at_time)

    return {
        "time_since_break":      time_since_break(conn, at_time),
        "switch_freq_10m":       switch_freq_10m(conn, at_time),
        "productive_ratio_30m":  productive_ratio,
        "distracting_ratio_30m": distracting_ratio,
        "task_streak_seconds":   task_streak_seconds(conn, at_time),
        "hour_of_day":           dt.hour,
        "day_of_week":           dt.weekday(),  #0 = Monday
    }

def compute_features_batch(conn, timestamps):
    #generate a feature row for each timestamp in the list
    return [compute_features(conn, t) for t in timestamps]
