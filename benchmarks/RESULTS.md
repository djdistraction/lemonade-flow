# Latency results

Status: **to be recorded on AMD hardware**

Nothing below has been measured. Do not paste the mock harness output here. `benchmarks/results/mock-run.json`, when present, is a fake server on loopback and is not a hardware result.

| Field | Value |
| --- | --- |
| Date | |
| Machine (CPU) | |
| GPU | |
| NPU | |
| Lemonade version (`lemonade status` or `/v1/health`) | |
| Model id | |
| Recipe | |
| Loaded device from `/v1/health` (`device`) | |
| Warm or cold (did this call load the model?) | |
| Prompt | Reply with one short sentence about a local model on a PC. |
| Client TTFT (s) | |
| Client total time (s) | |
| Completion tokens | |
| Client tokens/s | |
| Server-reported TTFT from `/v1/stats` (s) | |
| Server-reported tokens/s | |
| Command | `python benchmarks/bench_latency.py --endpoint http://localhost:13305 --model Qwen2.5-0.5B-Instruct-Hybrid` |

Method: [METHOD.md](METHOD.md).
