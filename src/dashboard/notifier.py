import time
# pyrefly: ignore [missing-import]
from win11toast import notify

FEEDBACK_URL  = "http://localhost:5001/nudge_feedback"
COOLDOWN_SECS = 900  # 15 minutes between nudges

_last_nudge_ts = 0.0

def send_nudge(prob):
    global _last_nudge_ts
    now = time.time()
    if now - _last_nudge_ts < COOLDOWN_SECS:
        return False

    notify(
        "Heads up",
        f"You're {prob:.0%} likely to start a distraction spiral soon.",
        actions=[
            {"activationType": "protocol", "arguments": f"{FEEDBACK_URL}?response=yes", "content": "On it"},
            {"activationType": "protocol", "arguments": f"{FEEDBACK_URL}?response=no",  "content": "Not helpful"},
        ],
        app_id="Procrastination Predictor",
    )
    _last_nudge_ts = now
    return True
