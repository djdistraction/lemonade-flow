# Latency method

This measures one streaming chat completion. It does not measure image generation.
No result in this repository is an AMD hardware number. Live results are **to be recorded on AMD hardware**.

## What is timed

The client is `benchmarks/bench_latency.py`.

1. Record `time.perf_counter` immediately before `POST /v1/chat/completions`.
2. The body is a streaming chat completion: `stream: true` and `stream_options.include_usage: true`.
3. **TTFT** is the elapsed time until the first server-sent event whose `choices[].delta.content` is a non-empty string. Role-only chunks do not count.
4. **Total time** is the elapsed time until `data: [DONE]` or the response body ends.
5. **Tokens/s** is `usage.completion_tokens / (total_seconds - ttft_seconds)` when both the usage object and a content token arrived. The script leaves the field null rather than guessing a token count.
6. After the stream, the script reads `GET /v1/stats` and stores it as `server_stats` (Lemonade's own `time_to_first_token` and `tokens_per_second`). Those numbers stay in their own fields. They are not copied over the client timings.
7. `GET /v1/system-info` is stored as processor name, `amd_npu.available`, and AMD GPU names. Paths are not recorded.

Temperature is 0. `max_tokens` defaults to 32. The default prompt is:

`Reply with one short sentence about a local model on a PC.`

Use the same prompt when you fill in `RESULTS.md`.

## Live run

Install Lemonade, pull the text model, and confirm the server is up (`lemonade status`). Then, from this repository, with the package installed:

```bash
python benchmarks/bench_latency.py --endpoint http://localhost:13305 --model Qwen2.5-0.5B-Instruct-Hybrid --output benchmarks/results/live-run.json
```

The JSON field `hardware_status` is `to be recorded on AMD hardware` until you copy the numbers into `benchmarks/RESULTS.md` and fill in the machine description yourself. The script will not write `RESULTS.md`.

Run it once after the model is loaded, and note whether that call paid for model load. A second run is the warm number. Record which one you are reporting.

## Mock run

```bash
python benchmarks/bench_latency.py --mock
```

This starts the in-process mock on 127.0.0.1 and writes `benchmarks/results/mock-run.json`. The file is labeled `mock`. It only proves the harness parses a streaming response. Do not put those numbers in `RESULTS.md`.
