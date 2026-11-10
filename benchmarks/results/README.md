JSON files in this directory are local runs and are gitignored.

`python benchmarks/bench_latency.py --mock` writes `mock-run.json`. That file is a loopback harness check. It is not an AMD result.

Live output belongs in a file you choose with `--output`. Copy the fields you measured into `../RESULTS.md`. Until you do that, the table stays **to be recorded on AMD hardware**.
