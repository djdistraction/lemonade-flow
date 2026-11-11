JSON files in this directory are local runs and are gitignored.

`python benchmarks/bench_latency.py --mock` writes `mock-run.json`. That file is a loopback harness check. It is not a result and does not belong in `../RESULTS.md`.

A live `--output` file is also local. Publish the measured fields by adding one row to `../RESULTS.md` in a pull request. No results have been submitted yet.
