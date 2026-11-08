import asyncio
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
from framework import LemonadeFlowEngine

st.set_page_config(page_title="Lemonade-Flow Orchestration Hub", layout="wide")

if "flow" not in st.session_state:
    st.session_state.flow = LemonadeFlowEngine(endpoint="http://localhost:13305", max_context_tokens=1024)
    st.session_state.loop = asyncio.new_event_loop()
    st.session_state.frame_drop_count = 0
    st.session_state.processed_logs = []

def render_b64_image(b64_string):
    img_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_data))

st.title("🌐 Lemonade-Flow: Multi-Modal Heterogeneous Orchestrator")
st.caption("Production Diagnostics | Real-Time Hardware Balancer Engine")

col_metric1, col_metric2, col_metric3 = st.columns(3)
with col_metric1:
    token_count = sum(len(st.session_state.flow.tokenizer.encode(m["content"])) for m in st.session_state.flow.rolling_context_buffer)
    st.metric(label="🧠 Active NPU Context Footprint", value=f"{token_count} / {st.session_state.flow.max_context_tokens} Tokens")
with col_metric2:
    st.metric(label="⏸️ Backpressure Dropped Frames", value=st.session_state.frame_drop_count)
with col_metric3:
    status_label = "Online" if st.session_state.flow.is_running else "Offline"
    st.metric(label="⚙️ System Routing State", value=status_label)

st.divider()

col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("🎙️ Live Stream Capture & NPU Translation")
    
    user_speech_input = st.text_input(
        "Simulate Spoken Input (Press Enter to trigger cross-modal thread compilation):",
        placeholder="e.g., A lonely spaceman exploring a bioluminescent crystal cavern..."
    )
    
    st.markdown("### 📝 Active Conversation Memory Cache")
    log_container = st.container(height=350)
    with log_container:
        if not st.session_state.flow.rolling_context_buffer:
            st.info("System waiting for speech chunks. Memory array empty.")
        for msg in st.session_state.flow.rolling_context_buffer:
            role_label = "👤 User Speech" if msg["role"] == "user" else "🤖 NPU Director"
            st.markdown(f"**{role_label}:** {msg['content']}")

with col_right:
    st.subheader("🎨 Hardware Accelerated Canvas (Radeon GPU)")
    canvas_placeholder = st.empty()
    canvas_placeholder.image("https://placehold.co/512x512?text=Lemonade-Flow+Canvas", use_container_width=True)

if user_speech_input:
    async def process_pipeline():
        if not st.session_state.flow.is_running:
            await st.session_state.flow.boot_engine()
            
        with st.spinner("NPU parsing intent... GPU scheduling priority matrix..."):
            npu_directive = "Convert this spoken narration into an ultra-descriptive, highly visual design prompt for a graphics engine."
            art_prompt = await st.session_state.flow.dispatch_text_to_npu(user_speech_input, npu_directive)
            
            style_wrapped_prompt = f"Digital concept art, masterwork illustration, deep color grade, {art_prompt}"
            render_future = await st.session_state.flow.enqueue_gpu_render(style_wrapped_prompt, steps=4)
            
            try:
                img_b64 = await render_future
                pil_img = render_b64_image(img_b64)
                canvas_placeholder.image(pil_img, use_container_width=True)
            except asyncio.CancelledError:
                st.session_state.frame_drop_count += 1
                st.toast("⏸️ Backpressure Guard triggered: Stale frame dropped to preserve VRAM pipeline!")
                
    st.session_state.loop.run_until_complete(process_pipeline())
    st.rerun()
