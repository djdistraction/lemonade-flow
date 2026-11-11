"""Dashboard submits one prompt once, and a cancelled render is a dropped frame."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

from dashboard import execute_pipeline
from lemonade_flow.mock_server import MockLemonadeServer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def server(monkeypatch):
    mock = MockLemonadeServer().start()
    monkeypatch.setenv("LEMONADE_ENDPOINT", mock.base_url)
    monkeypatch.delenv("LEMONADE_API_KEY", raising=False)
    try:
        yield mock
    finally:
        mock.close()


def test_cancelled_render_is_a_dropped_frame(server: MockLemonadeServer):
    from lemonade_flow import LemonadeFlowEngine

    async def scenario() -> None:
        flow = LemonadeFlowEngine(endpoint=server.base_url)
        real = flow.enqueue_gpu_render

        async def cancel_immediately(prompt: str, steps: int = 4):
            future = await real(prompt, steps)
            future.cancel()
            return future

        flow.enqueue_gpu_render = cancel_immediately  # type: ignore[method-assign]
        result = await execute_pipeline(flow, "a lonely spaceman")
        await flow.teardown()
        assert result["dropped"] is True
        assert result["image_b64"] is None
        assert len(server.chat_requests) == 1

    asyncio.run(scenario())


def test_one_submit_sends_one_chat_and_one_image(server: MockLemonadeServer):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "dashboard.py"), default_timeout=30)
    app.run()
    assert server.chat_requests == []
    assert server.image_requests == []

    app.text_input[0].input("a lonely spaceman")
    app.button[0].click().run()
    assert len(server.chat_requests) == 1
    assert len(server.image_requests) == 1
    assert server.chat_requests[0]["messages"][-1]["content"] == "a lonely spaceman"
    assert server.image_requests[0]["model"] == "SD-Turbo"

    app.run()
    assert len(server.chat_requests) == 1
    assert len(server.image_requests) == 1
    app.session_state["loop"].run_until_complete(app.session_state["flow"].teardown())
    app.session_state["loop"].close()
