# Lemonade-Flow

Lemonade-Flow is a Python client for a local [Lemonade Server](https://lemonade-server.ai/). It calls the server's OpenAI-compatible API on port 13305 (`/v1`) for text, speech, and images, and adds two controls around those calls:

- The image queue holds one pending render. A newer pending request cancels the previous future before the worker sends it. A request the server has already accepted runs to completion.
- Chat context is trimmed to a token budget. The system prompt stays. Each user turn is counted once.

The license is MIT.

## Requirements

- Python 3.10 or newer
- A Lemonade Server listening on `http://localhost:13305` (or another origin set with `LEMONADE_ENDPOINT`)
- The default models below, pulled into that server

Runtime libraries are aiohttp, Streamlit, tiktoken, and Pillow. They have wheels on Linux and Windows. PyAudio is not a dependency.

## Install Lemonade

Use the official installers:

- [Install Lemonade](https://lemonade-server.ai/docs/guide/install/)
- [Windows](https://lemonade-server.ai/docs/guide/install/windows/) (latest MSI: <https://github.com/lemonade-sdk/lemonade/releases/latest/download/lemonade.msi>)
- [Linux](https://lemonade-server.ai/docs/guide/install/linux/), including [Ubuntu](https://lemonade-server.ai/docs/guide/install/ubuntu/)

Confirm the server with `lemonade status`. The CLI binary in current docs is `lemonade`. The Linux package name is `lemonade-server`; if that is the executable on `PATH`, pass it the same arguments. API reference: [OpenAI-compatible routes](https://lemonade-server.ai/docs/api/openai/) and [server routes](https://lemonade-server.ai/docs/api/lemonade/). CLI reference: <https://lemonade-server.ai/docs/guide/cli/>.

## Install this package

From a clone of this repository:

```bash
pip install -e .
```

`pip install -e .` installs the runtime libraries and puts `lemonade_flow` on the import path, so `python examples/demo_app.py` does not need `PYTHONPATH`. `pip install -r requirements.txt` installs the same libraries without installing the package itself.

## Quick start

Pull the default registry ids, then run the example against `http://localhost:13305/v1`:

```bash
lemonade pull Qwen2.5-0.5B-Instruct-Hybrid
lemonade pull SD-Turbo
lemonade pull Whisper-Tiny
pip install -e .
python examples/demo_app.py
```

If the executable is `lemonade-server`:

```bash
lemonade-server pull Qwen2.5-0.5B-Instruct-Hybrid
lemonade-server pull SD-Turbo
lemonade-server pull Whisper-Tiny
```

The example sends two built-in text lines. It does not open a microphone. Each line is one chat completion. Two image prompts are queued together so the first future is cancelled and only the second is posted.

Other entry points:

```bash
python examples/demo_app.py --text "A red bicycle on a wet street"
python examples/demo_app.py --audio path/to/speech.wav
streamlit run dashboard.py
```

`--audio` posts the WAV to `POST /v1/audio/transcriptions`. The file path does not need a microphone library.

## Default models

Ids come from `src/cpp/resources/server_models.json` on [lemonade-sdk/lemonade](https://github.com/lemonade-sdk/lemonade) `main` at `0ca5189874c2`.

| Role | Id | Registry facts |
| --- | --- | --- |
| Text | `Qwen2.5-0.5B-Instruct-Hybrid` | `recipe: ryzenai-llm`, checkpoint `amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-1.7-hybrid`, `suggested: true`, `labels: ["chat"]`, about 0.83 GB. Lemonade documents hybrid Ryzen AI checkpoints as NPU prefill and GPU decode. This is the smallest suggested instruct hybrid above the 135M SmolLM entries. |
| Image | `SD-Turbo` | `recipe: sd-cpp`, `suggested: true`, the id in Lemonade's `POST /v1/images/generations` example. Default generation is 4 steps at 512x512. The recipe runs on CPU unless a GPU backend for `sd-cpp` is installed. The id does not mean the render used a Radeon. |
| Speech | `Whisper-Tiny` | `recipe: whispercpp`, `suggested: true`, the id in Lemonade's `POST /v1/audio/transcriptions` example. The registry also lists an optional NPU encoder cache. This client does not assume that cache is in use. |

`fastflow-lm` is not a registry id. FastFlowLM models use the `flm` recipe and are selected by setting `LEMONADE_TEXT_MODEL`.

On a machine where the hybrid model is filtered out (`404 model_not_supported`), pull a llama.cpp model and point the client at it:

```bash
lemonade pull Qwen3-0.6B-GGUF
```

Windows:

```bat
set LEMONADE_TEXT_MODEL=Qwen3-0.6B-GGUF
```

Linux and macOS:

```bash
export LEMONADE_TEXT_MODEL=Qwen3-0.6B-GGUF
```

`Qwen3-0.6B-GGUF` is `llamacpp`, about 0.38 GB. `Llama-3.2-1B-Instruct-NPU` is the NPU-only registry id.

## Configuration

Environment variables supply defaults. Arguments on `LemonadeFlowEngine` override them.

| Variable | Default | Request |
| --- | --- | --- |
| `LEMONADE_ENDPOINT` | `http://localhost:13305` | Server origin. Routes are under `/v1`. |
| `LEMONADE_TEXT_MODEL` | `Qwen2.5-0.5B-Instruct-Hybrid` | `POST /v1/chat/completions` field `model` |
| `LEMONADE_IMAGE_MODEL` | `SD-Turbo` | `POST /v1/images/generations` field `model` |
| `LEMONADE_TRANSCRIPTION_MODEL` | `Whisper-Tiny` | `POST /v1/audio/transcriptions` field `model` |
| `LEMONADE_MAX_CONTEXT_TOKENS` | `2048` | Token budget. The dashboard uses `1024` when this variable is unset. |
| `LEMONADE_API_KEY` | unset | Sent as `Authorization: Bearer` only when the variable is set. The value is not logged. |

Token counts use tiktoken `cl100k_base`. That is an estimate, not the tokenizer inside the loaded model.

A failed chat or image call raises `LemonadeAPIError` with the HTTP status and body. A chat body with no `choices` is an error.

## Optional audio extra

Microphone capture is optional. It records a WAV with `sounddevice` (PortAudio) and posts that file to the same transcription route:

```bash
pip install -e ".[audio]"
python examples/demo_app.py --mic --seconds 5
```

Skip the extra when PortAudio is not available. `--audio` and `--text` do not need it.

## Dashboard and examples

`streamlit run dashboard.py` opens a form. One submit runs one chat completion and one image generation. The page shows the conversation, the image, estimated context tokens, and the hardware report from the last startup probe.

`examples/demo_app.py` is the CLI. `framework.py` re-exports `LemonadeFlowEngine` for older imports. New code imports `lemonade_flow`.

| Piece | Role |
| --- | --- |
| `src/lemonade_flow/engine.py` | HTTP calls, token budget, render slot, hardware report |
| `examples/demo_app.py` | CLI for text, `--audio`, or `--mic` |
| `dashboard.py` | Streamlit page |
| `benchmarks/bench_latency.py` | One streaming chat timing |
| `src/lemonade_flow/mock_server.py` | Loopback stand-in for tests. Not Lemonade Server. |

## Hardware report

Startup reads three different sources. They are not substitutes for each other.

- **Recipe hint.** `GET /v1/models` returns `recipe`, not `backend`. The client maps known names (`ryzenai-llm`, `llamacpp`, `sd-cpp`, `whispercpp`, `flm`, and legacy `oga-npu`, `oga-hybrid`, `oga-cpu`) to a sentence about where that recipe usually runs. For `ryzenai-llm` the sentence uses the model id and checkpoint (`hybrid`, `npu`, or `cpu`). A `backend` field, if a server still sends one, is printed as an ignored undocumented field.
- **Loaded device.** `GET /v1/health` field `all_models_loaded[].device` is the device string for a model that is loaded (`cpu`, `gpu`, `npu`, `gpu|npu`, or `none`).
- **Host devices.** `GET /v1/system-info` supplies the processor name, `devices.amd_npu.available`, and AMD GPU names. A 404 on `/v1/...` is retried at `/api/v1/...`.

A hybrid recipe hint is not a measurement that the NPU ran. `amd_npu.available: false` means this response did not report an NPU. `sd-cpp` does not name a GPU. The loaded `device` field is the server's report for a model that is actually loaded.

The pending image slot holds one task. That limits how many renders this process queues. It is not a VRAM measurement.

## Tests

Tests talk to `lemonade_flow.mock_server` on 127.0.0.1. They do not open a network connection to Lemonade.

```bash
pip install -e ".[dev]"
python -m pytest
```

GitHub Actions runs that suite on Ubuntu and Windows (`.github/workflows/tests.yml`). The audio extra is not installed in CI.

## Benchmarks

Method: [benchmarks/METHOD.md](benchmarks/METHOD.md).

Community table: [benchmarks/RESULTS.md](benchmarks/RESULTS.md). No results have been submitted yet. A pull request can add one row per machine and run. Output from `--mock` is not a result.

```bash
python benchmarks/bench_latency.py --mock
python benchmarks/bench_latency.py --endpoint http://localhost:13305 --model Qwen2.5-0.5B-Instruct-Hybrid
```

`--mock` writes `benchmarks/results/mock-run.json` and labels it `mock`. That file is gitignored. The script does not write `benchmarks/RESULTS.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Copyright (c) 2026 Randall Gene. See [LICENSE](LICENSE).
