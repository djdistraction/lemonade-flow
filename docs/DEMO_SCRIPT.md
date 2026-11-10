# Demo script (about 2 minutes 40 seconds)

Record this on the AMD machine that will run Lemonade. Do not use the mock harness, and do not reuse anyone else's footage. The repository does not contain a video.

Have these ready before you hit record:

- Lemonade installed and running at `http://localhost:13305`
- Models pulled (commands below)
- This repo installed with `pip install -e .`
- A short WAV of your own voice, 16 kHz mono if you can, saved as `samples/line.wav`
- Task Manager open to Performance, with the NPU graph available (Windows 11)
- A terminal and a browser

Pull the registry ids this project calls:

```bash
lemonade pull Qwen2.5-0.5B-Instruct-Hybrid
lemonade pull SD-Turbo
lemonade pull Whisper-Tiny
```

If the shell only has the older executable name, the model ids stay the same:

```bash
lemonade-server pull Qwen2.5-0.5B-Instruct-Hybrid
lemonade-server pull SD-Turbo
lemonade-server pull Whisper-Tiny
```

## Shot list

| Time | On screen | Do this |
| --- | --- | --- |
| 0:00–0:15 | Repo README, then the terminal prompt | Show the title and the MIT license file. |
| 0:15–0:35 | Terminal | `lemonade status`, then the three `lemonade pull` commands if the models are not already present. Leave the command output visible. |
| 0:35–0:55 | Task Manager, Performance tab, NPU graph, plus the terminal | Start the text model with a real request (next shot). Point at the NPU graph while it moves. If the graph stays flat, say so. Hybrid checkpoints use the NPU for prefill and the GPU for token generation, so a short prompt can flash the NPU and then sit on the GPU. |
| 0:55–1:25 | Browser, `streamlit run dashboard.py` | Submit one sentence: "A lonely spaceman exploring a bioluminescent crystal cavern." Wait until one image appears. Open the Hardware report expander. |
| 1:25–1:55 | Terminal | `python examples/demo_app.py --audio samples/line.wav`. Keep the "stale render cancelled" line and the transcript on screen. |
| 1:55–2:20 | Task Manager GPU and NPU, then the generated image in the browser or the base64 length in the terminal | SD-Turbo is `sd-cpp`. Say whether a GPU backend is installed. Do not call it a Radeon render unless Task Manager or `/v1/health` shows the GPU doing the work. |
| 2:20–2:40 | `benchmarks/RESULTS.md` and this machine's Task Manager CPU name | Show the empty table. Say the numbers are still to be recorded on this AMD machine. Optional: run `python benchmarks/bench_latency.py` once, live, and point at the JSON. Do not show `mock-run.json` as a result. |

## Voiceover

Read this, or stay close to it. Cut a line before you claim something the screen does not show.

1. "This is Lemonade-Flow. It is a small open-source orchestrator for Lemonade Server on a local PC. Text, speech, and images go through Lemonade's OpenAI-compatible API on port 13305."
2. "The text model is Qwen2.5-0.5B-Instruct-Hybrid, recipe ryzenai-llm. On a Ryzen AI PC that checkpoint is the hybrid path: NPU prefill, GPU decode. The image model is SD-Turbo. Speech is Whisper-Tiny."
3. "I am pulling those models with the Lemonade CLI, then checking the server."
4. "This is the NPU graph in Task Manager while the chat request runs. I will only say the NPU is busy if this graph moves."
5. "The dashboard sends one chat completion and one image generation per submit. The hardware panel prints the recipe from /v1/models, the loaded device from /v1/health, and whether system-info sees an NPU. Those are separate facts."
6. "This WAV goes to POST /v1/audio/transcriptions. No microphone library is required for a file. The render slot keeps one pending image; the extra future is cancelled before it is sent."
7. "The latency table is empty on purpose. I will record TTFT, total time, and tokens per second on this machine and fill it in. The mock file is only a test of the script."

If you also show the microphone path, install the extra first (`pip install -e ".[audio]"`) and say that it records a WAV and then calls the same transcription endpoint. Skip it if PortAudio is not installed. The file path is the one that counts.

## Submission checklist

- [ ] You are a member of the [AMD AI Developer Program](https://www.amd.com/en/developer/resources/technical-articles/2026/join-the-lemonade-developer-challenge.html)
- [ ] You joined the Lemonade Discord and posted or followed the work in `#AMDDevChallenge`
- [ ] The GitHub repo is public and the license is the MIT text in `LICENSE` (Copyright (c) 2026 Randall Gene)
- [ ] You filled out the submission form on the AMD AI Developer Program member site
- [ ] The form links this repo and, if you recorded one, the demo video
- [ ] The video was recorded on your AMD machine. It shows Lemonade, the models, and the hardware you actually used
- [ ] `benchmarks/RESULTS.md` either stays marked "to be recorded on AMD hardware" or contains numbers you measured yourself
- [ ] You did not claim the NPU, GPU, or a tokens/s figure unless that shot shows it

Challenge rules: <https://www.amd.com/en/developer/resources/technical-articles/2026/join-the-lemonade-developer-challenge.html>
