#!/usr/bin/env bash
set -e
set -x

# Use Python wrapper to run no-baseline sweep with real-time logs
cd "$(dirname "$0")"/..

python3 scripts/eval_sweep_no_baseline.py
