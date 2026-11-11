# Latency results

No results have been submitted yet.

Add one row per machine and run in a pull request. Use a live run of `benchmarks/bench_latency.py` against Lemonade Server, plus the recipe and loaded device from `GET /v1/models` and `GET /v1/health`. The method is [METHOD.md](METHOD.md).

Mock harness output is not a result. Do not paste `python benchmarks/bench_latency.py --mock` or `benchmarks/results/mock-run.json` into this table.

| Date | CPU | GPU | NPU | Lemonade version | Model id | Recipe | Loaded device | Warm/cold | Prompt | Client TTFT (s) | Total time (s) | Completion tokens | Tokens/s | Server TTFT (s) | Server tokens/s | Command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
