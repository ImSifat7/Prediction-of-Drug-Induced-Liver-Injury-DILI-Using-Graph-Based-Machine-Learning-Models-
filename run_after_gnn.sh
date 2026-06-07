#!/bin/bash
# Orchestrate the remaining work after GNN retrain finishes.
# Runs ChemBERTa (5 seeds, per-seed save) -> stack_v2 -> save final summary.
set -e
cd "$(dirname "$0")"

echo "[$(date '+%H:%M:%S')] Starting ChemBERTa retrain (5 seeds, 30 epochs)..."
venv/Scripts/python.exe -m src.improved.chemberta --seeds 42 7 13 21 100 --epochs 30 > results/chemberta_v2_log.txt 2>&1
echo "[$(date '+%H:%M:%S')] ChemBERTa done."

echo "[$(date '+%H:%M:%S')] Running stack_v2..."
venv/Scripts/python.exe -m src.improved.stack_v2 > results/stack_v2_final_log.txt 2>&1
echo "[$(date '+%H:%M:%S')] Stack done."

echo "[$(date '+%H:%M:%S')] All done."
