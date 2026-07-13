from flask import Flask, request
import sqlite3
import time
import os
import yaml

app = Flask(__name__)

def DB_initialization():
    os.makedirs("data/raw", exist_ok=True)
    conn = sqlite3.connect("data/raw/activity.db", timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tab_events (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       REAL,
            url      TEXT,
            domain   TEXT,
            title    TEXT,
            category TEXT
        )
    """)
    try:
        conn.execute("ALTER TABLE tab_events ADD COLUMN category TEXT")
    except Exception:
        pass  #column already exists
    conn.commit()
    conn.close()

def categorize_url(domain):
    with open("config/app_categories.yaml", "r") as f:
        categories = yaml.safe_load(f)
    for category, items in categories.items():
        for site in items.get("sites", []):
            if site in domain:
                return category
    return "unknown"

def extract_domain(url):
    #pull just the domain out of a full URL without any external dependencies
    try:
        stripped = url.split("://", 1)[-1]   #drop the scheme
        domain = stripped.split("/")[0]       #drop everything after the first slash
        domain = domain.split("?")[0]         #drop query params if somehow still present
        return domain.lower()
    except Exception:
        return url

def insert_event(url, title):
    domain = extract_domain(url)
    category = categorize_url(domain)
    now = time.time()
    conn = sqlite3.connect("data/raw/activity.db", timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "INSERT INTO tab_events (ts, url, domain, title, category) VALUES (?, ?, ?, ?, ?)",
        (now, url, domain, title, category)
    )
    #update the most recent chrome window_event so the feature pipeline sees the real category
    conn.execute(
        """
        UPDATE window_events SET category = ?
        WHERE id = (
            SELECT id FROM window_events
            WHERE app_name LIKE '%chrome%'
            ORDER BY ts DESC LIMIT 1
        )
        """,
        (category,)
    )
    conn.commit()
    conn.close()

@app.route("/log_tab", methods=["POST"])
def log_tab():
    data = request.get_json(silent=True)
    if not data:
        return {"error": "no json"}, 400

    url = data.get("url", "").strip()
    title = data.get("title", "").strip()

    if not url:
        return {"error": "missing url"}, 400

    insert_event(url, title)
    return {"status": "ok"}, 200

def run():
    DB_initialization()
    app.run(host="127.0.0.1", port=5001, debug=False)
    #127.0.0.1 only — never exposed outside localhost

if __name__ == "__main__":
    run()
