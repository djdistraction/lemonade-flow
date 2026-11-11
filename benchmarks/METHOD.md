# Latency method

`benchmarks/bench_latency.py` times one streaming chat completion. It does not time image generation.

Checked-in numbers belong in [RESULTS.md](RESULTS.md). That table starts empty. Add a row by pull request after a live run. `--mock` output is a harness check on 127.0.0.1 and is not a result.

## What is timed

1. Record `time.perf_counter` immediately before `POST /v1/chat/completions`.
2. The body sets `stream: true` and `stream_options.include_usage: true`.
3. **TTFT** is the elapsed time until the first server-sent event whose `choices[].delta.content` is a non-empty string. Role-only chunks do not count.
4. **Total time** is the elapsed time until `data: [DONE]` or the response body ends.
5. **Tokens/s** is `usage.completion_tokens / (total_seconds - ttft_seconds)` when the stream reports usage and a content token arrived. The field stays null when either piece is missing. The script does not invent a token count.
6. After the stream, `GET /v1/stats` is stored as `server_stats` (`time_to_first_token` and `tokens_per_second` from Lemonade). Those values stay in their own columns.
7. `GET /v1/system-info` is stored as the processor name, `amd_npu.available`, and AMD GPU names. Paths are not recorded.

Temperature is 0. `max_tokens` defaults to 32. The default prompt is:

`Reply with one short sentence about a local model on a PC.`

Rows are easier to compare when they use that prompt. A different prompt belongs in the Prompt column.

## Live run

Install Lemonade, pull the text model, and confirm the server (`lemonade status`). From this repository, with the package installed:

```bash
python benchmarks/bench_latency.py --endpoint http://localhost:13305 --model Qwen2.5-0.5B-Instruct-Hybrid --output benchmarks/results/live-run.json
```

`live-run.json` is gitignored. `hardware_status` in that file is `local`. The script does not write `RESULTS.md`.

A run that loads the model is cold. A later run, after the model is already loaded, is warm. Put that in the Warm/cold column.

Copy client TTFT, total time, completion tokens, and tokens/s from the JSON. Copy server TTFT and server tokens/s from `server_stats`. Copy CPU, GPU, and NPU from `server_system` or from the machine. Copy recipe and loaded device from `GET /v1/models` and `GET /v1/health`.

## Mock run

```bash
python benchmarks/bench_latency.py --mock
```

This starts the in-process mock and writes `benchmarks/results/mock-run.json` with `label` set to `mock`. The file only shows that the harness parses a streaming response.
