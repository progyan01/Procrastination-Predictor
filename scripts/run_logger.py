import threading
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.logger import window_tracker, idle_tracker, tab_server

def start_thread(target, name):
    t = threading.Thread(target=target, name=name, daemon=True)
    t.start()
    return t

def main():
    print("Starting loggers...")

    start_thread(window_tracker.run, "window_tracker")
    start_thread(idle_tracker.run, "idle_tracker")
    start_thread(tab_server.run, "tab_server")

    print("Logging active. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping loggers.")


if __name__ == "__main__":
    main()
