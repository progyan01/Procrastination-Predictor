import win32gui
import win32process
import psutil
import sqlite3
import yaml
import time

def DB_initilization():
    conn = sqlite3.connect("data/raw/activity.db")

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
    #returns the connection so other functions can use it

def classification(app, title):
    with open("config/app_categories.yaml", "r") as f:
        categories = yaml.safe_load(f)

    app_clean = app.replace(".exe", "").lower()
    #strip the .exe and lowercase so we can match against yaml entries

    for category, items in categories.items():
        for listed_app in items.get("apps", []):
            if listed_app.lower() in app_clean:
                return category
            #substring match - e.g. "cursor" matches "cursor.exe"

    return "unknown"
    #default if nothing in the yaml matches

def insert_event(conn, window, category):
    title_to_log = window["title"] if category != "excluded" else ""
    #never store the window title for sensitive/excluded apps

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
            #skip this poll if win32 throws (e.g. system window with no process)

        if window != previous:
            #window changed - classify and log it
            category = classification(window["app"], window["title"])
            insert_event(conn, window, category)
            previous = window

        time.sleep(0.2)
        #poll every 200ms - low CPU impact, still catches switches fast enough

if __name__ == "__main__":
    run()