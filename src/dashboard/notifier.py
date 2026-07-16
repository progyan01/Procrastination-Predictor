import time
import threading
import sqlite3
# pyrefly: ignore [missing-import]
from win11toast import toast

DB_PATH       = "data/raw/activity.db"
COOLDOWN_SECS = 900   # 15 minutes between nudges

_last_nudge_ts = 0.0

def init_nudge_log(conn):
    """Create the nudge_log table if it doesn't exist yet."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nudge_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            REAL,       -- unix timestamp when nudge fired
            prob          REAL,       -- model probability at fire time
            was_helpful   INTEGER,    -- 1 = "On it" clicked, 0 = dismissed, NULL = no response
            model_version TEXT        -- "ml" or "heuristic" — matters for retraining
        )
    """)
    conn.commit()

def log_nudge(ts, prob, was_helpful, model_version):
    """Write one nudge event to nudge_log. Opens its own connection (called from a thread)."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    init_nudge_log(conn)
    conn.execute(
        "INSERT INTO nudge_log (ts, prob, was_helpful, model_version) VALUES (?, ?, ?, ?)",
        (ts, prob, was_helpful, model_version),
    )
    conn.commit()
    conn.close()



def _toast_and_log(prob, fired_ts, model_version):
    """
    Blocking toast call — runs in a daemon thread so the predictor loop isn't stalled.
    toast() returns the 'arguments' string of the button clicked, or None if dismissed.
    """
    result = toast(
        "Heads up 👀",
        f"You're {prob:.0%} likely to start a distraction spiral soon.",
        button={"activationType": "foreground", "arguments": "helpful", "content": "On it"},
        app_id="Procrastination Predictor",
    )

    # result == "helpful" if user clicked the button, None if dismissed/timed out
    was_helpful = 1 if result == "helpful" else 0
    log_nudge(fired_ts, prob, was_helpful, model_version)


def send_nudge(prob, model_version="ml"):
    global _last_nudge_ts
    now = time.time()

    if now - _last_nudge_ts < COOLDOWN_SECS:
        return False

    _last_nudge_ts = now
    fired_ts = now

    t = threading.Thread(
        target=_toast_and_log,
        args=(prob, fired_ts, model_version),
        daemon=True,
    )
    t.start()
    return True
