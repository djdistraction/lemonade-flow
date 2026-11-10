"""Engine tests against a loopback mock that speaks the documented Lemonade shapes."""

from __future__ import annotations

import asyncio
import wave

import pytest

from lemonade_flow import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    DEFAULT_TRANSCRIPTION_MODEL,
    LemonadeAPIError,
    LemonadeFlowEngine,
    describe_recipe,
)
from lemonade_flow.engine import _record_wav
from lemonade_flow.mock_server import MockLemonadeServer


@pytest.fixture
def server():
    mock = MockLemonadeServer().start()
    try:
        yield mock
    finally:
        mock.close()


def _engine(server: MockLemonadeServer, **kwargs) -> LemonadeFlowEngine:
    return LemonadeFlowEngine(endpoint=server.base_url, **kwargs)


def test_recipe_hints_follow_the_documented_field():
    assert "hybrid" in describe_recipe(
        {
            "id": "Qwen2.5-0.5B-Instruct-Hybrid",
            "recipe": "ryzenai-llm",
            "checkpoint": "amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-1.7-hybrid",
        }
    )
    assert "NPU" in describe_recipe(
        {
            "id": "Llama-3.2-1B-Instruct-NPU",
            "recipe": "ryzenai-llm",
            "checkpoint": "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-npu",
        }
    )
    assert describe_recipe({"id": "Qwen2.5-0.5B-Instruct-CPU", "recipe": "ryzenai-llm"}).startswith("CPU")
    assert "does not name CPU vs GPU" in describe_recipe({"id": "Qwen3-0.6B-GGUF", "recipe": "llamacpp"})
    assert "NPU" not in describe_recipe({"id": "Qwen3-0.6B-GGUF", "recipe": "llamacpp"})
    assert "legacy oga-npu" in describe_recipe({"id": "x", "recipe": "oga-npu"})
    assert "legacy oga-hybrid" in describe_recipe({"id": "x", "recipe": "oga-hybrid"})
    sd_hint = describe_recipe({"id": "SD-Turbo", "recipe": "sd-cpp"})
    assert "stable-diffusion.cpp" in sd_hint
    assert "Radeon" not in sd_hint
    assert "whisper.cpp" in describe_recipe({"id": "Whisper-Tiny", "recipe": "whispercpp"})
    assert "FastFlowLM" in describe_recipe({"id": "flm-model", "recipe": "flm"})
    assert "no recipe" in describe_recipe({"id": "odd", "backend": "fastflow-lm"})


def test_image_request_includes_the_registry_model(server: MockLemonadeServer):
    async def scenario() -> None:
        engine = _engine(server)
        await engine._ensure_session()
        pending, current = await engine.enqueue_render_batch([("stale prompt", 2), ("latest prompt", 4)])
        assert pending.cancelled()
        with pytest.raises(asyncio.CancelledError):
            await pending
        engine.is_running = True
        engine.worker_task = asyncio.create_task(engine._backpressure_orchestration_worker())
        image = await current
        await engine.teardown()
        assert image == server.image_b64
        assert len(server.image_requests) == 1
        body = server.image_requests[0]
        assert body["model"] == DEFAULT_IMAGE_MODEL == "SD-Turbo"
        assert body["prompt"] == "latest prompt"
        assert body["steps"] == 4
        assert body["response_format"] == "b64_json"
        assert body["size"] == "512x512"

    asyncio.run(scenario())


def test_image_model_is_configurable(server: MockLemonadeServer):
    async def scenario() -> None:
        engine = _engine(server, image_model="SDXL-Turbo")
        await engine.boot_engine()
        image = await engine.enqueue_gpu_render("a mountain")
        assert await image
        await engine.teardown()
        assert server.image_requests[0]["model"] == "SDXL-Turbo"

    asyncio.run(scenario())


def test_text_model_default_and_errors(server: MockLemonadeServer):
    async def scenario() -> None:
        engine = _engine(server, max_context_tokens=2048)
        await engine._ensure_session()
        reply = await engine.dispatch_text_to_npu("hello", "be brief")
        assert reply == "glowing crystal cavern"
        assert server.chat_requests[0]["model"] == DEFAULT_TEXT_MODEL
        assert server.chat_requests[0]["model"] != "fastflow-lm"
        assert server.chat_requests[0]["messages"][0]["role"] == "system"
        user_messages = [message for message in server.chat_requests[0]["messages"] if message["role"] == "user"]
        assert user_messages == [{"role": "user", "content": "hello"}]

        server.chat_status = 404
        with pytest.raises(LemonadeAPIError, match="HTTP 404"):
            await engine.dispatch_text_to_npu("again", "be brief")
        assert "NPU response received." not in str(engine.rolling_context_buffer)
        assert [message["content"] for message in engine.rolling_context_buffer if message["role"] == "user"] == ["hello"]

        server.chat_status = 200
        server.omit_choices = True
        with pytest.raises(LemonadeAPIError, match="missing choices"):
            await engine.dispatch_text_to_npu("third", "be brief")
        await engine.teardown()

    asyncio.run(scenario())


def test_text_model_override(server: MockLemonadeServer):
    async def scenario() -> None:
        engine = _engine(server, text_model="Llama-3.2-1B-Instruct-NPU")
        await engine._ensure_session()
        await engine.dispatch_text_to_npu("hello", "rules")
        await engine.teardown()
        assert server.chat_requests[0]["model"] == "Llama-3.2-1B-Instruct-NPU"

    asyncio.run(scenario())


def test_token_budget_counts_the_user_prompt_once():
    class WordTokenizer:
        def encode(self, text: str) -> list[str]:
            return text.split() or ["empty"]

    engine = LemonadeFlowEngine(endpoint="http://127.0.0.1:9", max_context_tokens=6)
    engine.tokenizer = WordTokenizer()
    engine.system_instruction_cache = "sys one"
    engine._remember_user_turn("alpha beta")
    engine.rolling_context_buffer.append({"role": "assistant", "content": "gamma"})
    engine._remember_user_turn("delta")
    engine._remember_user_turn("epsilon zeta")
    assert [message["content"] for message in engine.rolling_context_buffer] == [
        "gamma",
        "delta",
        "epsilon zeta",
    ]
    engine.max_context_tokens = 1
    engine.rolling_context_buffer.clear()
    engine._remember_user_turn("one two three four")
    assert engine.rolling_context_buffer == [{"role": "user", "content": "one two three four"}]


def test_hardware_report_uses_recipe_health_and_system_info(server: MockLemonadeServer):
    async def scenario() -> None:
        engine = _engine(server)
        report = await engine.boot_engine()
        await engine.teardown()
        text = report.text()
        by_id = {model.model_id: model for model in report.models}
        hybrid = by_id["Qwen2.5-0.5B-Instruct-Hybrid"]
        assert hybrid.recipe == "ryzenai-llm"
        assert hybrid.loaded_device == "gpu|npu"
        assert "hybrid" in hybrid.hint
        assert "Radeon" not in by_id["SD-Turbo"].hint
        assert by_id["odd-backend-only"].recipe is None
        assert by_id["odd-backend-only"].ignored_backend_field == "fastflow-lm"
        assert "ignored undocumented field backend" in text
        assert "Mapped to AMD XDNA NPU via" not in text
        assert report.system is not None
        assert report.system.source_path == "/v1/system-info"
        assert report.system.npu_available is True
        assert report.system.gpus == ["Radeon 8060S Graphics"]
        assert report.health_path == "/v1/health"

    asyncio.run(scenario())


def test_system_info_falls_back_to_legacy_prefix():
    mock = MockLemonadeServer(prefixes=("/api/v1",)).start()

    async def scenario() -> None:
        engine = LemonadeFlowEngine(endpoint=mock.base_url)
        report = await engine.probe_hardware()
        await engine.teardown()
        assert report.system is not None
        assert report.system.source_path == "/api/v1/system-info"
        assert report.health_path == "/api/v1/health"
        assert any(model.model_id == "SD-Turbo" for model in report.models)

    try:
        asyncio.run(scenario())
    finally:
        mock.close()


def test_npu_available_false_is_reported():
    mock = MockLemonadeServer().start()
    mock.system_info["devices"]["amd_npu"]["available"] = False

    async def scenario() -> None:
        engine = LemonadeFlowEngine(endpoint=mock.base_url)
        report = await engine.probe_hardware()
        await engine.teardown()
        assert report.system is not None
        assert report.system.npu_available is False

    try:
        asyncio.run(scenario())
    finally:
        mock.close()


def test_transcription_file_invokes_callbacks(server: MockLemonadeServer, tmp_path):
    wav_path = tmp_path / "speech.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 1600)

    async def scenario() -> None:
        engine = _engine(server)
        seen: list[str] = []

        @engine.register_voice_pipeline(word_boundary=3)
        async def on_speech(chunk: str) -> None:
            seen.append(chunk)

        text = await engine.transcribe_file(wav_path, language="en")
        await engine.teardown()
        assert text == "a cosmic warrior looking out"
        assert seen == ["a cosmic warrior", "looking out"]
        posted = server.transcription_requests[0]
        assert posted["model"] == DEFAULT_TRANSCRIPTION_MODEL == "Whisper-Tiny"
        assert posted["response_format"] == "json"
        assert posted["language"] == "en"
        assert posted["file_name"] == "speech.wav"
        assert posted["file_size"] > 0

    asyncio.run(scenario())


def test_microphone_helper_explains_the_optional_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sounddevice":
            raise ImportError("no module named sounddevice")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="optional"):
        _record_wav(0.1, 16000)


def test_microphone_path_posts_the_recorded_wav(server: MockLemonadeServer, tmp_path, monkeypatch):
    recorded = tmp_path / "mic.wav"
    with wave.open(str(recorded), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)

    def fake_record(seconds: float, sample_rate: int) -> str:
        assert seconds == 1.5
        assert sample_rate == 16000
        return str(recorded)

    monkeypatch.setattr("lemonade_flow.engine._record_wav", fake_record)

    async def scenario() -> None:
        engine = _engine(server)
        text = await engine.transcribe_microphone(seconds=1.5)
        await engine.teardown()
        assert text == server.transcript_text
        assert server.transcription_requests[0]["file_name"] == "mic.wav"

    asyncio.run(scenario())
    assert not recorded.exists()


def test_in_flight_render_is_not_cancelled(server: MockLemonadeServer):
    server.image_delay_seconds = 0.2

    async def scenario() -> None:
        engine = _engine(server)
        await engine.boot_engine()
        first = await engine.enqueue_gpu_render("first", steps=1)
        for _ in range(100):
            if server.image_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert server.image_started.is_set()
        second = await engine.enqueue_gpu_render("second", steps=2)
        assert first.cancelled() is False
        assert await first == server.image_b64
        assert await second == server.image_b64
        await engine.teardown()
        assert [body["prompt"] for body in server.image_requests] == ["first", "second"]

    asyncio.run(scenario())


def test_requirements_do_not_depend_on_pyaudio():
    requirement_lines = []
    for line in open("requirements.txt", encoding="utf-8"):
        stripped = line.split("#", 1)[0].strip().lower()
        if stripped:
            requirement_lines.append(stripped)
    assert all("pyaudio" not in line for line in requirement_lines)
    project = open("pyproject.toml", encoding="utf-8").read().lower()
    assert "pyaudio" not in project
    assert "sounddevice" in project
