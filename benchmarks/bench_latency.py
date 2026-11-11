"""Measure one streaming chat completion against Lemonade Server.

Client metrics
--------------
* ``ttft_seconds``: ``time.perf_counter`` from just before the POST until the
  first server-sent event whose ``choices[].delta.content`` is non-empty.
* ``total_seconds``: the same clock until ``data: [DONE]`` or the body ends.
* ``tokens_per_second``: ``usage.completion_tokens / (total_seconds - ttft_seconds)``
  when the stream reports usage (``stream_options.include_usage``) and TTFT is
  known. Otherwise null. This script does not invent a token count.

``GET /v1/stats`` is stored separately as ``server_stats``. It is not copied
into the client fields.

``--mock`` talks to ``lemonade_flow.mock_server`` on 127.0.0.1. That output is
a harness check, not a benchmark result. This script does not write
``benchmarks/RESULTS.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp

from lemonade_flow import DEFAULT_TEXT_MODEL
from lemonade_flow.mock_server import MockLemonadeServer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "Reply with one short sentence about a local model on a PC."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Lemonade chat latency")
    parser.add_argument("--endpoint", default="http://localhost:13305")
    parser.add_argument("--model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--mock", action="store_true", help="Run against the in-process mock, not AMD hardware")
    parser.add_argument("--output", type=Path, default=None, help="Write the JSON result here")
    return parser.parse_args(argv)


async def measure_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    ttft: Optional[float] = None
    text_parts: list[str] = []
    usage: Optional[dict[str, Any]] = None
    async with session.post(url, json=payload) as response:
        body_preview = ""
        if response.status != 200:
            body_preview = await response.text()
            raise RuntimeError(f"HTTP {response.status} {url} {body_preview}")
        buffer = ""
        async for raw in response.content:
            buffer += raw.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                event = _parse_sse_line(line.strip())
                if event is None:
                    continue
                if event == "[DONE]":
                    buffer = ""
                    break
                if not isinstance(event, dict):
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content:
                        if ttft is None:
                            ttft = time.perf_counter() - started
                        text_parts.append(content)
        if buffer.strip():
            event = _parse_sse_line(buffer.strip())
            if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                usage = event["usage"]
    total = time.perf_counter() - started
    completion_tokens = None
    if usage is not None and isinstance(usage.get("completion_tokens"), int):
        completion_tokens = usage["completion_tokens"]
    tokens_per_second = None
    if completion_tokens is not None and ttft is not None and total > ttft:
        tokens_per_second = completion_tokens / (total - ttft)
    return {
        "model": model,
        "prompt": prompt,
        "ttft_seconds": ttft,
        "total_seconds": total,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
        "tokens_per_second_method": (
            "usage.completion_tokens / (total_seconds - ttft_seconds)"
            if tokens_per_second is not None
            else "unavailable (no usage.completion_tokens or no content token)"
        ),
        "output_text": "".join(text_parts),
        "usage": usage,
    }


def _parse_sse_line(line: str) -> Any:
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return "[DONE]"
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


async def _read_stats(session: aiohttp.ClientSession, endpoint: str) -> Optional[dict[str, Any]]:
    url = endpoint.rstrip("/") + "/v1/stats"
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            payload = await response.json()
    except aiohttp.ClientError:
        return None
    return payload if isinstance(payload, dict) else None


async def _read_system(session: aiohttp.ClientSession, endpoint: str) -> Optional[dict[str, Any]]:
    url = endpoint.rstrip("/") + "/v1/system-info"
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            payload = await response.json()
    except aiohttp.ClientError:
        return None
    if not isinstance(payload, dict):
        return None
    devices = payload.get("devices") if isinstance(payload.get("devices"), dict) else {}
    npu = devices.get("amd_npu") if isinstance(devices.get("amd_npu"), dict) else {}
    gpus = []
    for gpu in devices.get("amd_gpu") or []:
        if isinstance(gpu, dict) and gpu.get("name"):
            gpus.append(str(gpu["name"]))
    return {
        "processor": payload.get("Processor"),
        "amd_npu_available": npu.get("available"),
        "amd_npu_name": npu.get("name"),
        "amd_gpu": gpus,
    }


async def run_measurement(endpoint: str, model: str, prompt: str, max_tokens: int, mock: bool) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        measured = await measure_chat(session, endpoint, model, prompt, max_tokens)
        measured["server_stats"] = await _read_stats(session, endpoint)
        measured["server_system"] = await _read_system(session, endpoint)
    measured["endpoint"] = endpoint
    measured["label"] = "mock" if mock else "live"
    if mock:
        measured["hardware_status"] = "mock"
        measured["note"] = (
            "MOCK RUN against lemonade_flow.mock_server. Harness check only. "
            "Not a benchmark result."
        )
    else:
        measured["hardware_status"] = "local"
        measured["note"] = (
            "Client timings from this process. They are not in benchmarks/RESULTS.md "
            "until a pull request adds a row. Mock output is not a result."
        )
    return measured


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output
    if output is not None and output.name == "RESULTS.md":
        print(
            "Refusing to write RESULTS.md. Add a measured row in a pull request.",
            file=sys.stderr,
        )
        return 2
    server: Optional[MockLemonadeServer] = None
    endpoint = args.endpoint
    if args.mock:
        server = MockLemonadeServer().start()
        endpoint = server.base_url or endpoint
        if output is None:
            output = REPO_ROOT / "benchmarks" / "results" / "mock-run.json"
    try:
        result = asyncio.run(
            run_measurement(endpoint, args.model, args.prompt, args.max_tokens, args.mock)
        )
    except Exception as exc:
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.close()
    text = json.dumps(result, indent=2)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
