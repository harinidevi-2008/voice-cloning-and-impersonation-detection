"""
run_dev.py
==========
Convenience script for local development and demo day: starts the FastAPI
backend and the Streamlit dashboard together with one command, and shuts
both down cleanly on Ctrl+C.

This is purely a convenience wrapper — it does not change how either
service works, and both can still be started manually and independently
exactly as documented in README.md.

Usage:
    python run_dev.py

Requires both requirements.txt and dashboard/requirements.txt to already
be installed.
"""

import subprocess
import sys
import time

BACKEND_CMD = [
    sys.executable, "-m", "uvicorn", "app.main:app",
    "--reload", "--port", "8000",
]
DASHBOARD_CMD = [
    sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_app.py",
    "--server.port", "8501",
]


def main() -> int:
    print("=" * 60)
    print("Voice Integrity Security Layer — starting backend + dashboard")
    print("=" * 60)

    print("\n[1/2] Starting backend  -> http://127.0.0.1:8000  (docs at /docs)")
    backend = subprocess.Popen(BACKEND_CMD)

    # Give the backend a moment to bind its port before the dashboard's
    # first health check runs, so the sidebar doesn't briefly show "offline".
    time.sleep(2)

    print("[2/2] Starting dashboard -> http://127.0.0.1:8501")
    dashboard = subprocess.Popen(DASHBOARD_CMD)

    print("\nBoth services are starting. Press Ctrl+C to stop both.\n")

    try:
        while True:
            time.sleep(1)
            if backend.poll() is not None:
                print("\n[!] Backend process exited unexpectedly. Shutting down dashboard too.")
                break
            if dashboard.poll() is not None:
                print("\n[!] Dashboard process exited unexpectedly. Shutting down backend too.")
                break
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for name, proc in (("dashboard", dashboard), ("backend", backend)):
            if proc.poll() is None:
                proc.terminate()
        for name, proc in (("dashboard", dashboard), ("backend", backend)):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[!] {name} did not exit in time, killing it.")
                proc.kill()
        print("Stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
