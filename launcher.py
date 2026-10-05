"""
Launcher for the packaged .exe version of the Eye Anomaly Detection app.

Streamlit apps normally run via `streamlit run app.py` in a terminal and
open in a browser tab. This launcher does the same thing programmatically,
with no visible terminal, so a double-clicked .exe can start the server
and open the browser automatically - the packaged app experience.

This file itself is NOT the app - it's a wrapper PyInstaller turns into
the .exe. Your actual app code (app/streamlit_app.py) is unchanged.
"""

import os
import sys
import time
import threading
import webbrowser

from streamlit.web import cli as stcli


def open_browser_when_ready():
    """Give the Streamlit server a moment to start, then open the browser."""
    time.sleep(2.5)
    webbrowser.open("http://localhost:8501")


def main():
    # When PyInstaller bundles this, sys._MEIPASS points at the temp
    # extraction folder containing all bundled files (including app/).
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    app_path = os.path.join(base_path, "app", "streamlit_app.py")

    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.headless=true",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()