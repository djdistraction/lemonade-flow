"""The benchmark harness runs against the mock and refuses to fill RESULTS.md."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mock_benchmark_is_labeled(tmp_path: Path):
    output = tmp_path / "mock-run.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "bench_latency.py"),
            "--mock",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["label"] == "mock"
    assert payload["hardware_status"] == "mock"
    assert "to be recorded on AMD hardware" in payload["note"]
    assert payload["ttft_seconds"] is not None
    assert payload["total_seconds"] >= payload["ttft_seconds"]
    assert payload["completion_tokens"] == 2
    assert payload["output_text"] == "Hello"
    assert payload["tokens_per_second"] is not None
    template = (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    assert "to be recorded on AMD hardware" in template
    assert "Hello" not in template
    assert str(payload["ttft_seconds"]) not in template


def test_benchmark_refuses_to_overwrite_the_template(tmp_path: Path):
    target = tmp_path / "RESULTS.md"
    target.write_text("keep\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "benchmarks" / "bench_latency.py"), "--mock", "--output", str(target)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert completed.returncode == 2
    assert target.read_text(encoding="utf-8") == "keep\n"
