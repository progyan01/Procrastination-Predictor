import os
import sys
import time
import webbrowser
import threading
import subprocess

# pyrefly: ignore [missing-import]
import pystray
# pyrefly: ignore [missing-import]
from pystray import MenuItem as Item, Menu
from PIL import Image

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON         = sys.executable
ICON_PATH      = os.path.join(BASE_DIR, "assets", "tray_icon.png")
DASHBOARD_APP  = os.path.join(BASE_DIR, "src", "dashboard", "app.py")
DASHBOARD_PORT = 8501

SCRIPT_PATHS = {
    "logger":    os.path.join(BASE_DIR, "scripts", "run_logger.py"),
    "predictor": os.path.join(BASE_DIR, "scripts", "run_predictor.py"),
}

procs: dict[str, subprocess.Popen | None] = {
    "logger":    None,
    "predictor": None,
    "dashboard": None,
}


def is_running(name: str) -> bool:
    p = procs[name]
    return p is not None and p.poll() is None


def start_process(name: str) -> None:
    if is_running(name):
        return
    if name == "dashboard":
        cmd = [PYTHON, "-m", "streamlit", "run", DASHBOARD_APP,
               "--server.headless", "true",
               "--server.port", str(DASHBOARD_PORT)]
    else:
        cmd = [PYTHON, SCRIPT_PATHS[name]]
    procs[name] = subprocess.Popen(cmd, cwd=BASE_DIR)


def stop_process(name: str) -> None:
    p = procs[name]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    procs[name] = None


def toggle(name: str, icon: pystray.Icon) -> None:
    if is_running(name):
        stop_process(name)
    else:
        start_process(name)
    icon.update_menu()


def open_dashboard(icon: pystray.Icon, item: Item) -> None:
    def _launch():
        if not is_running("dashboard"):
            start_process("dashboard")
            time.sleep(3)
        webbrowser.open(f"http://localhost:{DASHBOARD_PORT}")
    threading.Thread(target=_launch, daemon=True).start()


def quit_all(icon: pystray.Icon, item: Item) -> None:
    for name in list(procs):
        stop_process(name)
    icon.stop()


def build_menu(icon_ref: list) -> Menu:
    def status_label(name: str):
        return lambda item: f"{name.capitalize()}: {'Running' if is_running(name) else 'Stopped'}"

    def toggle_label(name: str):
        return lambda item: ("Stop " if is_running(name) else "Start ") + name.capitalize()

    def make_toggle(name: str):
        return lambda icon, item: toggle(name, icon_ref[0])

    return Menu(
        Item(status_label("logger"),    None, enabled=False),
        Item(status_label("predictor"), None, enabled=False),
        Menu.SEPARATOR,
        Item(toggle_label("logger"),    make_toggle("logger")),
        Item(toggle_label("predictor"), make_toggle("predictor")),
        Item("Open Dashboard",          open_dashboard),
        Menu.SEPARATOR,
        Item("Quit All",                quit_all),
    )


def main() -> None:
    if not os.path.exists(ICON_PATH):
        print(f"Icon not found at {ICON_PATH}. Place icon.png in the assets/ folder.")
        sys.exit(1)

    icon_image = Image.open(ICON_PATH)

    icon_ref: list[pystray.Icon] = []
    menu = build_menu(icon_ref)

    icon = pystray.Icon(
        name="procrastination_predictor",
        icon=icon_image,
        title="Procrastination Predictor",
        menu=menu,
    )
    icon_ref.append(icon)

    start_process("logger")
    start_process("predictor")

    icon.run()


if __name__ == "__main__":
    main()
