# Manuscript-reported values

`reported_metrics.csv` is a machine-readable transcription of the submitted
manuscript tables. It is not a substitute for per-seed logs or checkpoints.

Recomputed results should be written to `results/generated/` by
`scripts/aggregate_runs.py` and compared with this file. A tagged archival
release should include the generated mean, sample standard deviation, seeds,
checkpoint hashes, and Git commit.
