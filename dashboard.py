"""Streamlit diagnostic UI for one Lemonade text call and one image call per submit."""

from __future__ import annotations

import asyncio
import base64
import os
from io import BytesIO

import streamlit as st
from PIL import Image

from lemonade_flow import LemonadeFlowEngine

DIRECTIVE = (
    "Convert this narration into a visual design prompt for an image model. "
    "Reply with the prompt only."
)


async def execute_pipeline(flow: LemonadeFlowEngine, prompt: str) -> dict:
    """Run one chat completion and wait for the render future it enqueues.

    ``CancelledError`` means a newer task replaced this one in the single slot.
    The caller records that as a dropped frame instead of treating it as a crash.
    """

    if not flow.is_running:
        await flow.boot_engine()
    art_prompt = await flow.dispatch_text_to_npu(prompt, DIRECTIVE)
    styled = f"Digital concept art, masterwork illustration, deep color grade, {art_prompt}"
    render_future = await flow.enqueue_gpu_render(styled, steps=4)
    try:
        image_b64 = await render_future
    except asyncio.CancelledError:
        return {"dropped": True, "image_b64": None, "art_prompt": art_prompt}
    return {"dropped": False, "image_b64": image_b64, "art_prompt": art_prompt}


def render_b64_image(b64_string: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_string)))


def main() -> None:
    st.set_page_config(page_title="Lemonade-Flow", layout="wide")
    if "flow" not in st.session_state:
        st.session_state.flow = LemonadeFlowEngine(
            max_context_tokens=int(os.environ.get("LEMONADE_MAX_CONTEXT_TOKENS", "1024"))
        )
        st.session_state.loop = asyncio.new_event_loop()
        st.session_state.frame_drop_count = 0
        st.session_state.last_image_b64 = None
        st.session_state.last_error = None

    st.title("Lemonade-Flow")
    st.caption(
        "One submit sends one chat completion and one image generation to Lemonade Server. "
        "Device lines come from /v1/models recipe, /v1/health, and /v1/system-info."
    )

    with st.form("pipeline_form", clear_on_submit=True):
        user_speech_input = st.text_input(
            "Prompt",
            placeholder="A lonely spaceman exploring a bioluminescent crystal cavern",
        )
        submitted = st.form_submit_button("Run pipeline")

    # Work happens before the widgets that display it, so this run does not call
    # st.rerun() while the input still holds the submitted value.
    if submitted and user_speech_input.strip():
        try:
            result = st.session_state.loop.run_until_complete(
                execute_pipeline(st.session_state.flow, user_speech_input.strip())
            )
        except Exception as exc:
            st.session_state.last_error = f"{exc.__class__.__name__}: {exc}"
        else:
            st.session_state.last_error = None
            if result["dropped"]:
                st.session_state.frame_drop_count += 1
                st.toast("Backpressure guard: this frame was cancelled because a newer one replaced it.")
            elif result["image_b64"]:
                st.session_state.last_image_b64 = result["image_b64"]

    flow: LemonadeFlowEngine = st.session_state.flow
    token_count = sum(len(flow.tokenizer.encode(message["content"])) for message in flow.rolling_context_buffer)
    metric_context, metric_drops, metric_state = st.columns(3)
    metric_context.metric(
        "Context tokens (tiktoken estimate)",
        f"{token_count} / {flow.max_context_tokens}",
    )
    metric_drops.metric("Cancelled pending frames", st.session_state.frame_drop_count)
    metric_state.metric("Render worker", "Online" if flow.is_running else "Offline")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    left, right = st.columns(2)
    with left:
        st.subheader("Conversation")
        if not flow.rolling_context_buffer:
            st.info("No turns yet.")
        for message in flow.rolling_context_buffer:
            role = "User" if message["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {message['content']}")
    with right:
        st.subheader("Image")
        if st.session_state.last_image_b64:
            st.image(render_b64_image(st.session_state.last_image_b64), width="stretch")
        else:
            st.info("No image yet. A successful render from Lemonade shows up here.")

    report = flow.last_hardware_report
    if report is not None:
        with st.expander("Hardware report"):
            st.code(report.text())


if __name__ == "__main__":
    main()
