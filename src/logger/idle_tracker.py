from pynput import keyboard, mouse
import time
import threading
import sqlite3
import os

IDLE_THRESHOLD = 300  #seconds before user is considered idle

last_activity_time = time.time()
lock = threading.Lock()

def on_activity(*args):
    global last_activity_time
    with lock:
        last_activity_time = time.time()

def DB_initialization():
    os.makedirs("data/raw", exist_ok=True)
    conn = sqlite3.connect("data/raw/activity.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS idle_events (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ts      REAL,
            state   TEXT
        )
    """)
    conn.commit()
    return conn

def insert_event(conn, state):
    conn.execute(
        "INSERT INTO idle_events (ts, state) VALUES (?, ?)",
        (time.time(), state)
    )
    conn.commit()

def run():
    conn = DB_initialization()
    current_state = "active"

    kb_listener = keyboard.Listener(on_press=on_activity)
    ms_listener = mouse.Listener(on_move=on_activity, on_click=on_activity, on_scroll=on_activity)

    kb_listener.daemon = True
    ms_listener.daemon = True
    kb_listener.start()
    ms_listener.start()

    while True:
        with lock:
            elapsed = time.time() - last_activity_time

        new_state = "idle" if elapsed > IDLE_THRESHOLD else "active"

        if new_state != current_state:
            insert_event(conn, new_state)
            current_state = new_state

        time.sleep(5)


if __name__ == "__main__":
    run()
