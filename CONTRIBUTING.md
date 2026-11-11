# Contributing

Issues and pull requests are welcome.

## Develop

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests use the in-process mock server. They do not need a running Lemonade Server. The `[audio]` extra is optional and is not required for the suite.

## Benchmark rows

`benchmarks/RESULTS.md` is a community table. No results are checked in until a pull request adds them.

- Run `benchmarks/bench_latency.py` against a real Lemonade Server.
- Add one row per machine and run, with the columns already in the table.
- Say whether the run was cold (the call loaded the model) or warm.
- Leave `--mock` output and `benchmarks/results/mock-run.json` out of the table. Those files are a harness check.
- The script refuses to write `RESULTS.md`. Edit the table in the pull request so the numbers can be reviewed.

## License

Contributions are under the MIT license in `LICENSE`.
