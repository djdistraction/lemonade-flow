import asyncio
from framework import LemonadeFlowEngine

flow = LemonadeFlowEngine(endpoint="http://localhost:13305")

@flow.register_voice_pipeline(word_boundary=15)
async def handle_live_dictation(raw_text: str):
    print(f"\n📢 [Captured Live Audio Stream]: '{raw_text}'")
    
    npu_rules = "You are an expert design director. Convert the raw speech input into an artistic prompt for a digital graphics engine."
    optimized_art_prompt = await flow.dispatch_text_to_npu(raw_text, npu_rules)
    print(f"🧠 [NPU Task - FastFlowLM]: Evaluated Prompt -> '{optimized_art_prompt}'")
    
    curated_prompt = f"Futuristic cyberpunk concept art, hyper-detailed digital painting, neon accents, {optimized_art_prompt}"
    
    print("🎨 [GPU Task - Enqueueing Render]: Dispatching 4-step image pipeline task...")
    render_future = await flow.enqueue_gpu_render(curated_prompt, steps=4)
    
    image_base64 = await render_future
    print(f"✨ [GPU Success Frame Received]: Buffer Loaded. Length: {len(image_base64)} characters.")

async def runtime_loop():
    await flow.boot_engine()
    try:
        simulated_voice_chunks = [
            "A cosmic warrior looking out over a destroyed obsidian city skyline under dark red binary suns.",
            "A giant mechanical spider weaving a network of fiber optic cables down a rainy alleyway."
        ]
        
        for spoken_chunk in simulated_voice_chunks:
            await handle_live_dictation(spoken_chunk)
            await asyncio.sleep(1)
            
    finally:
        await flow.teardown()

if __name__ == "__main__":
    asyncio.run(runtime_loop())
