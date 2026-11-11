"""The demo script imports the installed package and does not need PYTHONPATH."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lemonade_flow.mock_server import MockLemonadeServer

ROOT = Path(__file__).resolve().parents[1]


def test_help_does_not_need_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "demo_app.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--audio" in completed.stdout
    assert "--mic" in completed.stdout


def test_demo_text_path_against_mock():
    server = MockLemonadeServer().start()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("LEMONADE_API_KEY", None)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "examples" / "demo_app.py"),
                "--endpoint",
                server.base_url,
                "--text",
                "a cosmic warrior under red suns",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            cwd=ROOT,
            timeout=30,
        )
    finally:
        server.close()
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "stale render cancelled" in completed.stdout
    assert len(server.chat_requests) == 1
    assert server.chat_requests[0]["model"] == "Qwen2.5-0.5B-Instruct-Hybrid"
    assert len(server.image_requests) == 1
    assert server.image_requests[0]["model"] == "SD-Turbo"
    assert "stale frame" not in server.image_requests[0]["prompt"]
