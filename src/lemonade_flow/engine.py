"""Lemonade Server client with a one-slot render queue and a token budget.

Model ids default to entries in lemonade-sdk/lemonade ``src/cpp/resources/server_models.json``
(main, commit 0ca5189874c2 when this default was chosen):

* ``Qwen2.5-0.5B-Instruct-Hybrid`` — recipe ``ryzenai-llm``, checkpoint
  ``amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-1.7-hybrid``, suggested chat model,
  about 0.83 GB. Hybrid Ryzen AI 1.7 checkpoints use the XDNA NPU for prefill
  and the Radeon GPU for decode. It is the smallest suggested instruct hybrid
  above the 135M SmolLM entries. Machines without a Ryzen AI NPU can set
  ``LEMONADE_TEXT_MODEL=Qwen3-0.6B-GGUF`` (recipe ``llamacpp``).
  ``Llama-3.2-1B-Instruct-NPU`` is the NPU-only registry id.
* ``SD-Turbo`` — recipe ``sd-cpp``, the image id used in the Lemonade
  ``POST /v1/images/generations`` example.
* ``Whisper-Tiny`` — recipe ``whispercpp``, the id used in the
  ``POST /v1/audio/transcriptions`` example.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import tiktoken

DEFAULT_ENDPOINT = "http://localhost:13305"
DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-Hybrid"
DEFAULT_IMAGE_MODEL = "SD-Turbo"
DEFAULT_TRANSCRIPTION_MODEL = "Whisper-Tiny"

# Documented recipe -> device hint. These are not measurements.
# ``/v1/models`` returns ``recipe``. A loaded model's ``device`` comes from
# ``/v1/health`` (``cpu``, ``gpu``, ``npu``, ``gpu|npu``, or ``none``).
_RECIPE_HINTS = {
    "oga-npu": "AMD XDNA NPU (legacy oga-npu recipe; current registry uses ryzenai-llm)",
    "oga-hybrid": "AMD XDNA NPU + GPU (legacy oga-hybrid recipe; current registry uses ryzenai-llm)",
    "oga-cpu": "CPU (legacy oga-cpu recipe; current registry uses ryzenai-llm)",
    "flm": "AMD XDNA NPU (FastFlowLM flm recipe, per Lemonade backend docs)",
    "llamacpp": "llama.cpp; the recipe does not name CPU vs GPU (see system-info llamacpp backends)",
    "llamacpp-hrx": "llama.cpp HRX; the recipe does not name the device",
    "whispercpp": "whisper.cpp speech-to-text; CPU, or NPU when that backend is installed",
    "moonshine": "Moonshine speech-to-text; the recipe does not name the device",
    "sd-cpp": "stable-diffusion.cpp; CPU unless a GPU backend is installed (not implied by the model id)",
    "kokoro": "Kokoro text-to-speech",
    "vllm": "vLLM; typically a GPU, but the recipe field does not name it",
    "onnxruntime": "ONNX Runtime; the recipe field does not name the device",
}


class LemonadeAPIError(RuntimeError):
    """A Lemonade HTTP call failed or the body did not match the documented shape."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ModelAllocation:
    model_id: str
    recipe: Optional[str]
    hint: str
    loaded_device: Optional[str]
    labels: list[str]
    ignored_backend_field: Optional[str] = None


@dataclass
class SystemSummary:
    source_path: Optional[str]
    processor: Optional[str]
    npu_available: Optional[bool]
    npu_name: Optional[str]
    gpus: list[str] = field(default_factory=list)


@dataclass
class HardwareReport:
    """What the server actually returned, plus documented recipe hints."""

    models: list[ModelAllocation] = field(default_factory=list)
    system: Optional[SystemSummary] = None
    health_path: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [
            "[lemonade-flow] hardware report",
            "recipe hints are documentation mappings, not a live device measurement.",
            "loaded_device is copied from /v1/health when that model is loaded.",
        ]
        if not self.models:
            lines.append("no models returned")
        for model in self.models:
            loaded = model.loaded_device if model.loaded_device is not None else "not loaded"
            lines.append(
                f"- id={model.model_id} recipe={model.recipe} "
                f"hint={model.hint} loaded_device={loaded} labels={','.join(model.labels) or '-'}"
            )
            if model.ignored_backend_field is not None:
                lines.append(
                    "  ignored undocumented field backend="
                    f"{model.ignored_backend_field!r}; the documented field is recipe"
                )
        if self.system is None:
            lines.append("system-info: not returned (tried /v1/system-info and /api/v1/system-info)")
        else:
            gpu = ", ".join(self.system.gpus) if self.system.gpus else "none reported"
            lines.append(
                f"system-info ({self.system.source_path}): processor={self.system.processor} "
                f"amd_npu.available={self.system.npu_available} amd_npu.name={self.system.npu_name} "
                f"amd_gpu={gpu}"
            )
        for note in self.notes:
            lines.append(f"note: {note}")
        return "\n".join(lines)


def describe_recipe(model: dict) -> str:
    """Map a ``/v1/models`` object to a device hint.

    Prefers ``recipe``. Does not treat a ``backend`` key as the device: the
    documented model object uses ``recipe`` (for example ``llamacpp``,
    ``ryzenai-llm``, ``sd-cpp``, ``whispercpp``, and the legacy ``oga-*`` names).
    """

    recipe = model.get("recipe")
    if not isinstance(recipe, str) or not recipe:
        return "no recipe field on this model object; device was not reported"
    if recipe == "ryzenai-llm":
        return _ryzenai_hint(model)
    return _RECIPE_HINTS.get(recipe, f"recipe {recipe!r} has no device mapping; reporting the recipe only")


def _ryzenai_hint(model: dict) -> str:
    blob = f"{model.get('id', '')} {model.get('checkpoint', '')}".lower()
    if "hybrid" in blob:
        return (
            "Ryzen AI hybrid via ryzenai-llm "
            "(NPU prefill + GPU decode for *-hybrid checkpoints; not measured here)"
        )
    if "npu" in blob:
        return "AMD XDNA NPU via ryzenai-llm (*-npu checkpoint; not measured here)"
    if "cpu" in blob:
        return "CPU via ryzenai-llm (*-cpu checkpoint; not measured here)"
    return "ryzenai-llm; the server selects NPU, hybrid, or CPU for this checkpoint"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("LEMONADE_API_KEY")
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _record_wav(seconds: float, sample_rate: int) -> str:
    """Record mono PCM16 from the default microphone. Requires the audio extra."""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Microphone capture is optional. Install it with: pip install -e \".[audio]\" "
            "(sounddevice / PortAudio). Transcribing a WAV file does not need a microphone."
        ) from exc
    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    handle, name = tempfile.mkstemp(suffix=".wav")
    os.close(handle)
    with wave.open(name, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return name


class LemonadeFlowEngine:
    """Queue image renders in a single slot and call Lemonade's OpenAI-compatible API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        text_model: Optional[str] = None,
        image_model: Optional[str] = None,
        transcription_model: Optional[str] = None,
        image_size: str = "512x512",
    ):
        self.endpoint = (endpoint or _env_str("LEMONADE_ENDPOINT", DEFAULT_ENDPOINT)).rstrip("/")
        if max_context_tokens is None:
            max_context_tokens = int(_env_str("LEMONADE_MAX_CONTEXT_TOKENS", "2048"))
        self.max_context_tokens = max_context_tokens
        self.text_model = text_model or _env_str("LEMONADE_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        self.image_model = image_model or _env_str("LEMONADE_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
        self.transcription_model = transcription_model or _env_str(
            "LEMONADE_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL
        )
        self.image_size = image_size
        self.session: Optional[aiohttp.ClientSession] = None
        self.speech_callbacks: list[dict[str, Any]] = []
        self._next_render_task: Optional[dict[str, Any]] = None
        self._render_signal = asyncio.Event()
        self._slot_lock = asyncio.Lock()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.system_instruction_cache = ""
        self.rolling_context_buffer: list[dict[str, str]] = []
        self.worker_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.last_hardware_report: Optional[HardwareReport] = None

    async def boot_engine(self) -> HardwareReport:
        """Open the HTTP session, start the render worker, and probe hardware."""

        await self._ensure_session()
        self.is_running = True
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._backpressure_orchestration_worker())
        print("[lemonade-flow] engine active")
        report = await self.probe_hardware()
        print(report.text())
        return report

    def register_voice_pipeline(self, word_boundary: int = 12):
        """Register a callback invoked with each transcript chunk.

        ``word_boundary`` is the maximum number of words passed to one call.
        Shorter transcripts are delivered whole. The callback may be sync or async.
        """

        def decorator(func: Callable):
            self.speech_callbacks.append({"func": func, "boundary": word_boundary})
            return func

        return decorator

    async def enqueue_gpu_render(self, prompt: str, steps: int = 4) -> asyncio.Future:
        """Queue one image generation. A newer pending task cancels the previous future."""

        futures = await self.enqueue_render_batch([(prompt, steps)])
        return futures[0]

    async def enqueue_render_batch(self, items: list[tuple[str, int]]) -> list[asyncio.Future]:
        """Queue several renders while holding the slot lock.

        Only the last item stays pending. Earlier futures in the batch, and any
        future already waiting in the slot, are cancelled before the worker can
        observe them.
        """

        if not items:
            return []
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future] = []
        async with self._slot_lock:
            for prompt, steps in items:
                future: asyncio.Future = loop.create_future()
                previous = self._next_render_task
                if previous is not None and not previous["future"].done():
                    previous["future"].cancel()
                self._next_render_task = {"prompt": prompt, "steps": steps, "future": future}
                futures.append(future)
            self._render_signal.set()
        return futures

    async def _backpressure_orchestration_worker(self) -> None:
        while self.is_running:
            await self._render_signal.wait()
            if not self.is_running:
                break
            async with self._slot_lock:
                self._render_signal.clear()
                task = self._next_render_task
                self._next_render_task = None
            if not task:
                continue
            await self._run_image_task(task)

    async def _run_image_task(self, task: dict[str, Any]) -> None:
        future: asyncio.Future = task["future"]
        if future.done():
            return
        url = f"{self.endpoint}/v1/images/generations"
        payload = {
            "model": self.image_model,
            "prompt": task["prompt"],
            "n": 1,
            "size": self.image_size,
            "steps": task["steps"],
            "response_format": "b64_json",
        }
        try:
            await self._ensure_session()
            assert self.session is not None
            async with self.session.post(url, json=payload) as resp:
                data = await _read_json(resp)
            image = _image_b64(data)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
            return
        if not future.done():
            future.set_result(image)

    def _context_token_count(self) -> int:
        total = self._calculate_token_weight(self.system_instruction_cache)
        for message in self.rolling_context_buffer:
            total += self._calculate_token_weight(message["content"])
        return total

    def _remember_user_turn(self, user_prompt: str) -> None:
        """Append the user turn once and drop oldest turns until the budget fits.

        The user prompt is only counted as the buffer entry just appended.
        A single over-budget turn is kept so the request still has a user message.
        """

        self.rolling_context_buffer.append({"role": "user", "content": user_prompt})
        while self._context_token_count() > self.max_context_tokens and len(self.rolling_context_buffer) > 1:
            self.rolling_context_buffer.pop(0)

    def _rollback_user_turn(self, user_prompt: str) -> None:
        if (
            self.rolling_context_buffer
            and self.rolling_context_buffer[-1].get("role") == "user"
            and self.rolling_context_buffer[-1].get("content") == user_prompt
        ):
            self.rolling_context_buffer.pop()

    async def dispatch_text_to_npu(self, user_prompt: str, context_rules: str) -> str:
        """Send a chat completion. Raises ``LemonadeAPIError`` when the call fails."""

        if context_rules != self.system_instruction_cache:
            self.system_instruction_cache = context_rules
        self._remember_user_turn(user_prompt)
        url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.text_model,
            "messages": [{"role": "system", "content": self.system_instruction_cache}]
            + self.rolling_context_buffer,
            "temperature": 0.2,
        }
        try:
            await self._ensure_session()
            assert self.session is not None
            async with self.session.post(url, json=payload) as resp:
                data = await _read_json(resp)
            assistant_response = _chat_content(data)
        except Exception:
            self._rollback_user_turn(user_prompt)
            raise
        self.rolling_context_buffer.append({"role": "assistant", "content": assistant_response})
        return assistant_response

    async def transcribe_file(self, audio_path: str | os.PathLike[str], language: Optional[str] = None) -> str:
        """Transcribe a WAV file with ``POST /v1/audio/transcriptions`` and run voice callbacks."""

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        url = f"{self.endpoint}/v1/audio/transcriptions"
        form = aiohttp.FormData()
        form.add_field("model", self.transcription_model)
        form.add_field("response_format", "json")
        if language:
            form.add_field("language", language)
        with path.open("rb") as handle:
            form.add_field("file", handle, filename=path.name, content_type="audio/wav")
            await self._ensure_session()
            assert self.session is not None
            async with self.session.post(url, data=form) as resp:
                data = await _read_json(resp)
        text = data.get("text")
        if not isinstance(text, str):
            raise LemonadeAPIError("transcription response missing string field text", body=json.dumps(data))
        await self.submit_transcript(text)
        return text

    async def transcribe_microphone(self, seconds: float = 5.0, sample_rate: int = 16000) -> str:
        """Record from the microphone, then transcribe the WAV. Needs the audio extra."""

        if seconds <= 0:
            raise ValueError("seconds must be positive")
        wav_path = await asyncio.to_thread(_record_wav, seconds, sample_rate)
        try:
            return await self.transcribe_file(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    async def submit_transcript(self, text: str) -> str:
        """Invoke registered voice callbacks with a transcript (typed, file, or mic)."""

        cleaned = text.strip()
        if not cleaned:
            return ""
        await self._emit_transcript(cleaned)
        return cleaned

    async def _emit_transcript(self, text: str) -> None:
        words = text.split()
        for callback in self.speech_callbacks:
            boundary = int(callback["boundary"])
            chunks = _word_chunks(words, boundary)
            for chunk in chunks:
                result = callback["func"](chunk)
                if inspect.isawaitable(result):
                    await result

    async def probe_hardware(self) -> HardwareReport:
        """Build a report from ``/v1/models``, ``/v1/health``, and ``/v1/system-info``."""

        await self._ensure_session()
        report = HardwareReport()
        models_payload, models_error = await self._get_first(["/v1/models", "/api/v1/models"])
        health_payload, health_error = await self._get_first(["/v1/health", "/api/v1/health"])
        system_payload, system_error = await self._get_first(["/v1/system-info", "/api/v1/system-info"])
        loaded_devices: dict[str, str] = {}
        if health_payload is None:
            report.notes.append(health_error or "health endpoint was not returned")
        else:
            report.health_path = health_payload[0]
            loaded_devices = _loaded_devices(health_payload[1])
        if models_payload is None:
            report.notes.append(models_error or "models endpoint was not returned")
        else:
            data = models_payload[1].get("data", []) if isinstance(models_payload[1], dict) else []
            if not isinstance(data, list):
                report.notes.append("models response data was not a list")
                data = []
            for model in data:
                if not isinstance(model, dict):
                    continue
                model_id = str(model.get("id", ""))
                recipe = model.get("recipe") if isinstance(model.get("recipe"), str) else None
                labels = [str(label) for label in model.get("labels", [])] if isinstance(model.get("labels"), list) else []
                backend_field = model.get("backend") if isinstance(model.get("backend"), str) else None
                report.models.append(
                    ModelAllocation(
                        model_id=model_id,
                        recipe=recipe,
                        hint=describe_recipe(model),
                        loaded_device=loaded_devices.get(model_id),
                        labels=labels,
                        ignored_backend_field=backend_field,
                    )
                )
        if system_payload is None:
            report.system = None
            if system_error:
                report.notes.append(system_error)
        else:
            report.system = _summarize_system(system_payload[0], system_payload[1])
        self.last_hardware_report = report
        return report

    async def _get_first(self, paths: list[str]) -> tuple[Optional[tuple[str, Any]], Optional[str]]:
        assert self.session is not None
        last_error: Optional[str] = None
        for path in paths:
            try:
                async with self.session.get(f"{self.endpoint}{path}") as resp:
                    if resp.status == 404:
                        continue
                    text = await resp.text()
                    if resp.status != 200:
                        last_error = f"{path} HTTP {resp.status}"
                        continue
                    try:
                        return (path, json.loads(text) if text else {}), None
                    except json.JSONDecodeError:
                        last_error = f"{path} returned non-JSON"
            except aiohttp.ClientError as exc:
                last_error = f"{path} {exc.__class__.__name__}"
        return None, last_error

    async def teardown(self) -> None:
        self.is_running = False
        async with self._slot_lock:
            pending = self._next_render_task
            self._next_render_task = None
            if pending is not None and not pending["future"].done():
                pending["future"].cancel()
            self._render_signal.set()
        if self.worker_task is not None:
            try:
                await asyncio.wait_for(self.worker_task, timeout=5)
            except asyncio.TimeoutError:
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass
            self.worker_task = None
        if self.session is not None and not self.session.closed:
            await self.session.close()
        self.session = None

    async def _ensure_session(self) -> None:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(headers=_auth_headers(), timeout=timeout)

    def _calculate_token_weight(self, text: str) -> int:
        return len(self.tokenizer.encode(text or ""))


def _word_chunks(words: list[str], boundary: int) -> list[str]:
    if not words:
        return []
    if boundary <= 0:
        return [" ".join(words)]
    return [" ".join(words[index : index + boundary]) for index in range(0, len(words), boundary)]


def _loaded_devices(health: Any) -> dict[str, str]:
    devices: dict[str, str] = {}
    if not isinstance(health, dict):
        return devices
    loaded = health.get("all_models_loaded")
    if not isinstance(loaded, list):
        return devices
    for entry in loaded:
        if not isinstance(entry, dict):
            continue
        name = entry.get("model_name")
        device = entry.get("device")
        if isinstance(name, str) and isinstance(device, str):
            devices[name] = device
    return devices


def _summarize_system(path: str, payload: Any) -> SystemSummary:
    if not isinstance(payload, dict):
        return SystemSummary(source_path=path, processor=None, npu_available=None, npu_name=None)
    devices = payload.get("devices") if isinstance(payload.get("devices"), dict) else {}
    npu = devices.get("amd_npu") if isinstance(devices.get("amd_npu"), dict) else {}
    gpus: list[str] = []
    amd_gpu = devices.get("amd_gpu")
    if isinstance(amd_gpu, list):
        for gpu in amd_gpu:
            if isinstance(gpu, dict) and gpu.get("name"):
                gpus.append(str(gpu["name"]))
    npu_available = npu.get("available") if isinstance(npu.get("available"), bool) else None
    npu_name = str(npu["name"]) if isinstance(npu.get("name"), str) else None
    processor = payload.get("Processor")
    return SystemSummary(
        source_path=path,
        processor=str(processor) if isinstance(processor, str) else None,
        npu_available=npu_available,
        npu_name=npu_name,
        gpus=gpus,
    )


async def _read_json(resp: aiohttp.ClientResponse) -> Any:
    text = await resp.text()
    if resp.status != 200:
        raise LemonadeAPIError(
            f"Lemonade request failed: HTTP {resp.status} {resp.url} {text}",
            status=resp.status,
            body=text,
        )
    try:
        return json.loads(text) if text else {}
    except json.JSONDecodeError as exc:
        raise LemonadeAPIError(
            f"Lemonade returned a non-JSON body: {text}",
            status=resp.status,
            body=text,
        ) from exc


def _chat_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise LemonadeAPIError(f"chat completion response was not an object: {data!r}")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LemonadeAPIError(
            f"chat completion response missing choices: {json.dumps(data)}",
            body=json.dumps(data),
        )
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LemonadeAPIError(
            f"chat completion response missing message content: {json.dumps(data)}",
            body=json.dumps(data),
        )
    return message["content"]


def _image_b64(data: Any) -> str:
    if not isinstance(data, dict) or not isinstance(data.get("data"), list) or not data["data"]:
        raise LemonadeAPIError(f"image response missing data list: {data!r}")
    first = data["data"][0]
    if not isinstance(first, dict) or not isinstance(first.get("b64_json"), str):
        raise LemonadeAPIError(f"image response missing b64_json: {data!r}")
    return first["b64_json"]
