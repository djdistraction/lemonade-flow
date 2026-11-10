# Lemonade-Flow

Lemonade-Flow is a Python client for [Lemonade Server](https://lemonade-server.ai/). It sends text, speech, and image requests to one local OpenAI-compatible server and keeps two guards around those calls:

- The image queue holds one pending render. A newer pending request cancels the previous future before the worker sends it. A request the server has already accepted is left to finish.
- Chat context is trimmed to a token budget. The system prompt is kept. The user prompt is counted once.

It is built for the [Lemonade Developer Challenge](https://www.amd.com/en/developer/resources/technical-articles/2026/join-the-lemonade-developer-challenge.html). The license is MIT.

## Architecture

```
typed text, WAV file, or optional microphone
        |
        v
voice callbacks (register_voice_pipeline)
        |
        +--> POST /v1/chat/completions          text model
        |
        +--> single pending image slot
                 |
                 v
             POST /v1/images/generations        image model

startup probe:
  GET /v1/models          recipe, id, labels
  GET /v1/health          loaded device (cpu, gpu, npu, gpu|npu, none)
  GET /v1/system-info     processor, amd_npu.available, amd_gpu names
```

Speech uses `POST /v1/audio/transcriptions` when you pass a WAV file. The microphone path records a WAV with `sounddevice` and then calls that same endpoint. The default demo does not open a microphone. Lemonade's realtime WebSocket (`/v1/realtime`) is not used.

Code map:

| Piece | Role |
| --- | --- |
| `src/lemonade_flow/engine.py` | HTTP calls, token budget, render slot, hardware report |
| `examples/demo_app.py` | CLI: sample text, `--audio`, or `--mic` |
| `dashboard.py` | Streamlit page, one submit, one pipeline |
| `benchmarks/bench_latency.py` | One streaming chat timing |
| `src/lemonade_flow/mock_server.py` | Loopback stand-in for tests. Not Lemonade |

`framework.py` re-exports `LemonadeFlowEngine` for older imports.

## Install Lemonade

Official installers:

- [Install Lemonade](https://lemonade-server.ai/docs/guide/install/)
- [Windows](https://lemonade-server.ai/docs/guide/install/windows/) (latest MSI: <https://github.com/lemonade-sdk/lemonade/releases/latest/download/lemonade.msi>)
- [Linux](https://lemonade-server.ai/docs/guide/install/linux/), including [Ubuntu](https://lemonade-server.ai/docs/guide/install/ubuntu/) (`sudo apt install lemonade-server` from the Lemonade PPA, or the Snap)

The server listens on `http://localhost:13305` unless you change the port. Check it with `lemonade status`. API reference: [OpenAI-compatible routes](https://lemonade-server.ai/docs/api/openai/) and [server routes](https://lemonade-server.ai/docs/api/lemonade/) (`/v1/health`, `/v1/system-info`). CLI reference: <https://lemonade-server.ai/docs/guide/cli/>.

The current CLI binary is `lemonade`. The Linux package is still named `lemonade-server`. If your install only put `lemonade-server` on `PATH`, use that name with the same arguments.

## Models

Ids were taken from `src/cpp/resources/server_models.json` on [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade) `main` at `0ca5189874c2`.

| Role | Id | Why this id |
| --- | --- | --- |
| Text | `Qwen2.5-0.5B-Instruct-Hybrid` | `recipe: ryzenai-llm`, checkpoint `amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-1.7-hybrid`, `suggested: true`, `labels: ["chat"]`, about 0.83 GB. It is the smallest suggested instruct hybrid above the 135M SmolLM entries. Lemonade's hybrid Ryzen AI path uses the XDNA NPU for prefill and the Radeon GPU for decode. A machine without a Ryzen AI NPU should switch to `Qwen3-0.6B-GGUF` (`llamacpp`, about 0.38 GB). `Llama-3.2-1B-Instruct-NPU` is the NPU-only registry id if you want that instead. |
| Image | `SD-Turbo` | `recipe: sd-cpp`, `suggested: true`, the id in the `POST /v1/images/generations` example. Four steps, 512x512. CPU unless you install a GPU backend for `sd-cpp`. The model id does not mean the render is on a Radeon. |
| Speech | `Whisper-Tiny` | `recipe: whispercpp`, `suggested: true`, the id in the `POST /v1/audio/transcriptions` example. The registry also lists an optional NPU encoder cache. This client does not assume the NPU is in use. |

`fastflow-lm` is not a registry id. FastFlowLM models use the `flm` recipe and are not hard-coded here.

Pull the three defaults:

```bash
lemonade pull Qwen2.5-0.5B-Instruct-Hybrid
lemonade pull SD-Turbo
lemonade pull Whisper-Tiny
```

Same ids if the executable is `lemonade-server`:

```bash
lemonade-server pull Qwen2.5-0.5B-Instruct-Hybrid
lemonade-server pull SD-Turbo
lemonade-server pull Whisper-Tiny
```

CPU/GPU fallback when Ryzen AI hybrid models are filtered out (`404 model_not_supported`):

```bash
lemonade pull Qwen3-0.6B-GGUF
```

Then set `LEMONADE_TEXT_MODEL=Qwen3-0.6B-GGUF`.

## Run this project

Python 3.10 or newer. From this repository:

```bash
pip install -e .
python examples/demo_app.py
streamlit run dashboard.py
```

`pip install -e .` puts the package on the import path, so `python examples/demo_app.py` does not need `PYTHONPATH`. `pip install -r requirements.txt` installs the same runtime libraries (aiohttp, streamlit, tiktoken, pillow) and is the dependency list for Linux and Windows. It does not install PyAudio. Microphone capture is optional:

```bash
pip install -e ".[audio]"
python examples/demo_app.py --mic --seconds 5
```

That extra installs `sounddevice` and needs PortAudio. Skip it if the install fails. A WAV file does not need it:

```bash
python examples/demo_app.py --audio path/to/speech.wav
python examples/demo_app.py --text "A red bicycle on a wet street"
```

The default `python examples/demo_app.py` sends two built-in sentences as text. It registers a voice callback and calls it. It also queues two image prompts in one step so the first future is cancelled and only the second is posted. That is the dropped-frame guard, not two GPU jobs.

The dashboard uses a form. One click runs one chat completion and one image generation. Submitting again is a new click. The page does not call `st.rerun()` while the prompt is still in the box.

## Configuration

Environment variables are the defaults. Constructor arguments on `LemonadeFlowEngine` override them.

| Variable | Default | Used for |
| --- | --- | --- |
| `LEMONADE_ENDPOINT` | `http://localhost:13305` | Server origin |
| `LEMONADE_TEXT_MODEL` | `Qwen2.5-0.5B-Instruct-Hybrid` | `POST /v1/chat/completions` `model` |
| `LEMONADE_IMAGE_MODEL` | `SD-Turbo` | `POST /v1/images/generations` `model` |
| `LEMONADE_TRANSCRIPTION_MODEL` | `Whisper-Tiny` | `POST /v1/audio/transcriptions` `model` |
| `LEMONADE_MAX_CONTEXT_TOKENS` | `2048` (dashboard uses `1024` unless you set this) | Token budget |
| `LEMONADE_API_KEY` | unset | Sent as `Authorization: Bearer` only when the variable is set. The value is not logged. |

The token count uses tiktoken `cl100k_base`. That is an estimate, not the tokenizer inside the Ryzen AI or GGUF model.

Chat and image errors raise `LemonadeAPIError` with the HTTP status and body. A missing `choices` entry is an error. The old placeholder `NPU response received.` is not returned.

## Hardware notes

The startup report has three sources, and they are not interchangeable:

- **Recipe hint.** `/v1/models` returns `recipe`, not `backend`. The client maps known recipe names (`ryzenai-llm`, `llamacpp`, `sd-cpp`, `whispercpp`, `flm`, and the legacy `oga-npu` / `oga-hybrid` / `oga-cpu` names) to a sentence about where that recipe usually runs. For `ryzenai-llm` the hint looks at the model id and checkpoint (`hybrid`, `npu`, or `cpu`). A `backend` field, if a server still sends one, is printed as an ignored undocumented field.
- **Loaded device.** `/v1/health` field `all_models_loaded[].device` is the device string the server reports for a model that is actually loaded (`cpu`, `gpu`, `npu`, `gpu|npu`, or `none`).
- **Host devices.** `/v1/system-info` supplies the processor name, `devices.amd_npu.available`, and AMD GPU names. If `/v1/...` returns 404, the probe tries `/api/v1/...`.

A hybrid recipe hint does not mean the NPU graph moved. `amd_npu.available: false` means this host did not report an NPU. `sd-cpp` does not imply a Radeon. Read the loaded device and Task Manager before you say which chip did the work.

The pending image slot is one task. That bounds how many renders this process will queue. It does not measure VRAM.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests start a loopback mock of the documented routes. They do not call a network Lemonade server. GitHub Actions runs that suite on Ubuntu and Windows (`.github/workflows/tests.yml`).

## Benchmark

Method: [benchmarks/METHOD.md](benchmarks/METHOD.md).

Empty table, **to be recorded on AMD hardware**: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

```bash
python benchmarks/bench_latency.py --mock
python benchmarks/bench_latency.py --endpoint http://localhost:13305 --model Qwen2.5-0.5B-Instruct-Hybrid
```

`--mock` writes `benchmarks/results/mock-run.json` and labels it `mock`. That file is gitignored. Do not copy those numbers into `RESULTS.md`. The script will not write `RESULTS.md` for you.

## Recording a demo

[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) is a shot list and voiceover for a recording on your own AMD PC, plus the submission checklist (AMD AI Developer Program, Lemonade Discord `#AMDDevChallenge`, member-site form).

## License

MIT. Copyright (c) 2026 Randall Gene. See [LICENSE](LICENSE).
