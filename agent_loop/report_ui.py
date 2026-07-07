"""
agent_loop.report_ui — launcher for the research-report Streamlit app (ADR 0003).

Mirrors the `rdagent ui --port …` ergonomics but for the agent-driven loop's file state:

    python -m agent_loop.report_ui --port 19900 [--root git_ignore_folder/agent_loop]

It shells `streamlit run agent_loop/report_app.py`, passing `--root` through to the app. No qlib, no
Docker, no `.env` — the app is a pure host-side reader.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fire

APP = Path(__file__).with_name("report_app.py")


def run(port: int = 19900, root: str = "git_ignore_folder/agent_loop", address: str = "127.0.0.1") -> int:
    """Launch the Streamlit report UI. Ctrl-C to stop."""
    if not APP.exists():
        raise FileNotFoundError(f"report app not found: {APP}")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        "--server.address", address,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--", "--root", root,
    ]
    print(f"[report-ui] http://{address}:{port}   (root={root})")
    print(f"[report-ui] {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    fire.Fire(run)
