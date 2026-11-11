"""In-process stand-in for the Lemonade routes this project calls.

It follows the documented JSON shapes. It is not Lemonade Server and it does
not produce AMD hardware measurements.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Optional

from aiohttp import web

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

DEFAULT_MODELS: list[dict[str, Any]] = [
    {
        "id": "Qwen2.5-0.5B-Instruct-Hybrid",
        "object": "model",
        "owned_by": "lemonade",
        "checkpoint": "amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-1.7-hybrid",
        "recipe": "ryzenai-llm",
        "labels": ["chat"],
    },
    {
        "id": "Qwen3-0.6B-GGUF",
        "object": "model",
        "owned_by": "lemonade",
        "recipe": "llamacpp",
        "labels": ["chat"],
    },
    {
        "id": "Llama-3.2-1B-Instruct-NPU",
        "object": "model",
        "owned_by": "lemonade",
        "checkpoint": "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-npu",
        "recipe": "ryzenai-llm",
        "labels": ["chat"],
    },
    {
        "id": "legacy-oga-npu",
        "object": "model",
        "owned_by": "lemonade",
        "recipe": "oga-npu",
        "labels": ["chat"],
    },
    {
        "id": "SD-Turbo",
        "object": "model",
        "owned_by": "lemonade",
        "recipe": "sd-cpp",
        "labels": ["image"],
    },
    {
        "id": "Whisper-Tiny",
        "object": "model",
        "owned_by": "lemonade",
        "recipe": "whispercpp",
        "labels": ["transcription", "realtime-transcription"],
    },
    {
        "id": "odd-backend-only",
        "object": "model",
        "owned_by": "lemonade",
        "backend": "fastflow-lm",
        "labels": ["chat"],
    },
]

DEFAULT_HEALTH = {
    "status": "ok",
    "version": "mock",
    "all_models_loaded": [
        {
            "model_name": "Qwen2.5-0.5B-Instruct-Hybrid",
            "type": "llm",
            "device": "gpu|npu",
            "recipe": "ryzenai-llm",
        }
    ],
}

DEFAULT_SYSTEM_INFO = {
    "OS Version": "test",
    "Processor": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
    "devices": {
        "amd_npu": {"available": True, "name": "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", "family": "XDNA2"},
        "amd_gpu": [{"available": True, "name": "Radeon 8060S Graphics"}],
    },
}

DEFAULT_CHAT = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "glowing crystal cavern"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
}


class MockLemonadeServer:
    """Loopback HTTP server. Bind address is 127.0.0.1 only."""

    def __init__(self, prefixes: tuple[str, ...] = ("/v1",)):
        self.prefixes = prefixes
        self.chat_requests: list[dict[str, Any]] = []
        self.image_requests: list[dict[str, Any]] = []
        self.transcription_requests: list[dict[str, Any]] = []
        self.chat_status = 200
        self.chat_payload: Any = json.loads(json.dumps(DEFAULT_CHAT))
        self.omit_choices = False
        self.image_delay_seconds = 0.0
        self.image_b64 = TINY_PNG_B64
        self.transcript_text = "a cosmic warrior looking out"
        self.models = json.loads(json.dumps(DEFAULT_MODELS))
        self.health = json.loads(json.dumps(DEFAULT_HEALTH))
        self.system_info = json.loads(json.dumps(DEFAULT_SYSTEM_INFO))
        self.stats = {
            "time_to_first_token": 0.01,
            "tokens_per_second": 10.0,
            "input_tokens": 3,
            "output_tokens": 2,
        }
        self.image_started = threading.Event()
        self.base_url: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "MockLemonadeServer":
        ready = threading.Event()
        error: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(self._start_site(ready))
                loop.run_forever()
            except BaseException as exc:  # pragma: no cover - surfaced to the caller
                error.append(exc)
                ready.set()

        self._thread = threading.Thread(target=run, name="mock-lemonade", daemon=True)
        self._thread.start()
        if not ready.wait(5):
            raise RuntimeError("mock Lemonade server did not start")
        if error:
            raise error[0]
        return self

    async def _start_site(self, ready: threading.Event) -> None:
        app = web.Application()
        for prefix in self.prefixes:
            app.router.add_post(f"{prefix}/chat/completions", self._chat)
            app.router.add_post(f"{prefix}/images/generations", self._images)
            app.router.add_post(f"{prefix}/audio/transcriptions", self._transcribe)
            app.router.add_get(f"{prefix}/models", self._models)
            app.router.add_get(f"{prefix}/health", self._health)
            app.router.add_get(f"{prefix}/system-info", self._system_info)
            app.router.add_get(f"{prefix}/stats", self._stats)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.base_url = f"http://127.0.0.1:{site.port}"
        self._runner = runner
        ready.set()

    def close(self) -> None:
        if self._loop is None or self._runner is None:
            return
        async def _cleanup() -> None:
            assert self._runner is not None
            await self._runner.cleanup()

        future = asyncio.run_coroutine_threadsafe(_cleanup(), self._loop)
        future.result(5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(5)
        self._loop = None
        self._runner = None

    async def _chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.chat_requests.append(body)
        if self.chat_status != 200:
            return web.json_response(
                {"error": {"message": "model not found", "type": "model_not_found"}},
                status=self.chat_status,
            )
        if self.omit_choices:
            return web.json_response({"id": "chatcmpl-mock", "object": "chat.completion", "choices": []})
        if body.get("stream"):
            response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            for piece in ("Hel", "lo"):
                chunk = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(0.01)
            usage = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
            await response.write(f"data: {json.dumps(usage)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response(self.chat_payload)

    async def _images(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.image_requests.append(body)
        self.image_started.set()
        if "model" not in body:
            return web.json_response(
                {"error": {"message": "missing model", "type": "missing_argument"}},
                status=400,
            )
        if self.image_delay_seconds:
            await asyncio.sleep(self.image_delay_seconds)
        return web.json_response({"created": 0, "data": [{"b64_json": self.image_b64}]})

    async def _transcribe(self, request: web.Request) -> web.StreamResponse:
        reader = await request.multipart()
        fields: dict[str, Any] = {}
        while True:
            part = await reader.next()
            if part is None:
                break
            payload = await part.read()
            if part.name == "file":
                fields["file_name"] = part.filename
                fields["file_size"] = len(payload)
            else:
                fields[part.name] = payload.decode()
        self.transcription_requests.append(fields)
        return web.json_response({"text": self.transcript_text})

    async def _models(self, _request: web.Request) -> web.StreamResponse:
        return web.json_response({"object": "list", "data": self.models})

    async def _health(self, _request: web.Request) -> web.StreamResponse:
        return web.json_response(self.health)

    async def _system_info(self, _request: web.Request) -> web.StreamResponse:
        return web.json_response(self.system_info)

    async def _stats(self, _request: web.Request) -> web.StreamResponse:
        return web.json_response(self.stats)
