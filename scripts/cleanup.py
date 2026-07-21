import sqlite3
import sys
import os
import time
import logging
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features and labels"))
# pyrefly: ignore [missing-import]
from weak_labels import productive_before, distracting_after

DB_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "activity.db")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cleanup.log")

RETENTION_DAYS = 14

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def run():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            date                   TEXT PRIMARY KEY,
            productive_seconds     REAL,
            distracting_seconds    REAL,
            neutral_seconds        REAL,
            unknown_seconds        REAL,
            procrastination_events INTEGER,
            total_switches         INTEGER,
            active_seconds         REAL
        )
    """)
    conn.commit()

    cutoff_ts = time.time() - RETENTION_DAYS * 86400
    rows = conn.execute(
        """
        SELECT DISTINCT date(ts, 'unixepoch', 'localtime') AS day
        FROM window_events
        WHERE ts < ?
          AND date(ts, 'unixepoch', 'localtime') NOT IN (
              SELECT date FROM daily_summaries
          )
        ORDER BY day ASC
        """,
        (cutoff_ts,),
    ).fetchall()
    days = [row[0] for row in rows]

    if not days:
        log.info("Nothing to clean up.")
        conn.close()
        return

    log.info(f"Processing {len(days)} day(s): {days[0]} → {days[-1]}")

    succeeded = 0
    for day in days:
        day_start = datetime.strptime(day, "%Y-%m-%d").timestamp()
        day_end   = day_start + 86400

        try:
            with conn:  # rolls back the whole day if anything throws
                # aggregate window events by category
                agg = conn.execute(
                    """
                    WITH durations AS (
                        SELECT category,
                               LEAD(ts) OVER (ORDER BY ts) - ts AS duration
                        FROM window_events
                        WHERE ts >= ? AND ts < ?
                    )
                    SELECT
                        SUM(CASE WHEN category = 'productive'  THEN duration ELSE 0 END),
                        SUM(CASE WHEN category = 'distracting' THEN duration ELSE 0 END),
                        SUM(CASE WHEN category = 'neutral'     THEN duration ELSE 0 END),
                        SUM(CASE WHEN category = 'unknown'     THEN duration ELSE 0 END),
                        COUNT(*)
                    FROM durations
                    """,
                    (day_start, day_end),
                ).fetchone()

                productive_s  = agg[0] or 0.0
                distracting_s = agg[1] or 0.0
                neutral_s     = agg[2] or 0.0
                unknown_s     = agg[3] or 0.0
                switches      = agg[4]

                # aggregate active seconds from idle events
                active_row = conn.execute(
                    """
                    WITH durations AS (
                        SELECT state,
                               LEAD(ts) OVER (ORDER BY ts) - ts AS duration
                        FROM idle_events
                        WHERE ts >= ? AND ts < ?
                    )
                    SELECT SUM(CASE WHEN state = 'active' THEN duration ELSE 0 END)
                    FROM durations
                    """,
                    (day_start, day_end),
                ).fetchone()
                active_s = active_row[0] or 0.0

                # count procrastination events via weak-label rule
                timestamps = [
                    r[0] for r in conn.execute(
                        "SELECT ts FROM window_events WHERE ts >= ? AND ts < ? ORDER BY ts",
                        (day_start, day_end),
                    ).fetchall()
                ]
                proc_count = sum(
                    1 for ts in timestamps
                    if productive_before(conn, ts) and distracting_after(conn, ts)
                )

                conn.execute(
                    """
                    INSERT OR IGNORE INTO daily_summaries
                        (date, productive_seconds, distracting_seconds, neutral_seconds,
                         unknown_seconds, procrastination_events, total_switches, active_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (day, productive_s, distracting_s, neutral_s, unknown_s,
                     proc_count, switches, active_s),
                )

                conn.execute("DELETE FROM window_events WHERE ts >= ? AND ts < ?", (day_start, day_end))
                conn.execute("DELETE FROM idle_events   WHERE ts >= ? AND ts < ?", (day_start, day_end))
                conn.execute("DELETE FROM tab_events    WHERE ts >= ? AND ts < ?", (day_start, day_end))

            succeeded += 1
            log.info(
                f"{day}  productive={productive_s:.0f}s"
                f"  distracting={distracting_s:.0f}s"
                f"  proc_events={proc_count}"
                f"  switches={switches}"
            )
        except Exception as exc:
            log.error(f"{day}: failed — {exc}")

    try:
        conn.execute("VACUUM")
        log.info("VACUUM complete.")
    except Exception as exc:
        log.warning(f"VACUUM skipped: {exc}")

    log.info(f"Done. {succeeded}/{len(days)} day(s) aggregated and purged.")
    conn.close()


if __name__ == "__main__":
    run()
