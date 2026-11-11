"""Run a text or WAV prompt through Lemonade-Flow.

The default path sends two sample lines as text. It does not open a microphone.
Pass ``--audio file.wav`` to call ``POST /v1/audio/transcriptions``.
Pass ``--mic`` only after ``pip install -e '.[audio]'``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from lemonade_flow import LemonadeFlowEngine

SAMPLE_LINES = [
    "A cosmic warrior looking out over an obsidian city under two red suns.",
    "A mechanical spider weaving fiber optic cables down a rainy alley.",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lemonade-Flow demo")
    parser.add_argument("--endpoint", default=None, help="Lemonade base URL. Default http://localhost:13305")
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--image-model", default=None)
    parser.add_argument("--transcription-model", default=None)
    parser.add_argument(
        "--text",
        action="append",
        help="Use this transcript instead of a microphone. Repeat to send several turns.",
    )
    parser.add_argument("--audio", help="WAV file to transcribe with Lemonade. No microphone required.")
    parser.add_argument("--mic", action="store_true", help="Record the microphone, then transcribe. Optional extra.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Microphone recording length.")
    return parser.parse_args(argv)


async def runtime_loop(args: argparse.Namespace) -> None:
    flow = LemonadeFlowEngine(
        endpoint=args.endpoint,
        text_model=args.text_model,
        image_model=args.image_model,
        transcription_model=args.transcription_model,
    )

    @flow.register_voice_pipeline(word_boundary=15)
    async def handle_transcript(raw_text: str) -> None:
        print(f"[lemonade-flow] transcript: {raw_text}")
        art_prompt = await flow.dispatch_text_to_npu(
            raw_text,
            "You are a design director. Turn the transcript into one image prompt.",
        )
        print(f"[lemonade-flow] text model {flow.text_model} replied: {art_prompt}")
        styled = f"Futuristic concept art, neon accents, {art_prompt}"
        # One lock hold: the stale future is cancelled before the worker runs.
        stale, fresh = await flow.enqueue_render_batch(
            [("stale frame " + styled, 4), (styled, 4)]
        )
        if stale.cancelled():
            print("[lemonade-flow] stale render cancelled")
        image_base64 = await fresh
        print(f"[lemonade-flow] image base64 length: {len(image_base64)}")

    await flow.boot_engine()
    try:
        if args.mic:
            await flow.transcribe_microphone(seconds=args.seconds)
        elif args.audio:
            await flow.transcribe_file(args.audio)
        else:
            lines = args.text or SAMPLE_LINES
            for line in lines:
                await flow.submit_transcript(line)
    finally:
        await flow.teardown()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        asyncio.run(runtime_loop(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
