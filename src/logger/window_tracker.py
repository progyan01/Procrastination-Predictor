import win32gui
import win32process
import psutil
import sqlite3
import yaml
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from redaction import redact

def DB_initilization():
    os.makedirs("data/raw", exist_ok=True)
    conn = sqlite3.connect("data/raw/activity.db", timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS window_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL,
            app_name    TEXT,
            title       TEXT,
            category    TEXT
        )
    """)
    conn.commit()
    return conn

def classification(app, title):
    with open("config/app_categories.yaml", "r") as f:
        categories = yaml.safe_load(f)

    appName = app.replace(".exe", "").lower()

    for category, items in categories.items():
        for listed_app in items.get("apps", []):
            if listed_app.lower() in appName:
                return category

    return "unknown"

def insert_event(conn, window, category):
    if category == "excluded":
        title_to_log = ""   #never store title for sensitive/excluded apps
    else:
        title_to_log = redact(window["title"])

    conn.execute(
        "INSERT INTO window_events (ts, app_name, title, category) VALUES (?, ?, ?, ?)",
        (time.time(), window["app"], title_to_log, category)
    )
    conn.commit()

def active_window():
    hwnd = win32gui.GetForegroundWindow()
    #gives the window handle of the currently open program
    title = win32gui.GetWindowText(hwnd)
    #gives the title of that window
    thread_id, pid = win32process.GetWindowThreadProcessId(hwnd)
    #thread_id is not relevant to us but pid is the process id that uniquely identifies the running program

    process = psutil.Process(pid)
    return {"app": process.name(), "title": title}

def run():
    conn = DB_initilization()
    previous = None
    #track the last logged window so we only write a row when something actually changes

    while True:
        try:
            window = active_window()
        except Exception:
            time.sleep(0.2)
            continue

        if window != previous:
            category = classification(window["app"], window["title"])
            insert_event(conn, window, category)
            previous = window

        time.sleep(0.2)
        #poll every 200ms

if __name__ == "__main__":
    run()