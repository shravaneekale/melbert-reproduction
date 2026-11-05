#!/usr/bin/env bash
# Produce per-token predictions for every evaluation set with one checkpoint.
# Usage (from repo root):
#   bash scripts/run_predictions.sh <checkpoint_dir> <tag>
# e.g.
#   bash scripts/run_predictions.sh external/MelBERT/saves/roberta-base/0_20261106-1012 ours
#   bash scripts/run_predictions.sh checkpoints/official official
set -euo pipefail
CKPT=$1
TAG=${2:-ours}
mkdir -p results/preds

run () {  # $1 = data dir, $2 = name
  if [ -f "$1/test.tsv" ]; then
    python src/predict.py --checkpoint "$CKPT" --data_dir "$1" --out "results/preds/${TAG}_$2.csv"
  else
    echo "[skip] $1/test.tsv not found"
  fi
}

# Original benchmarks (replication)
run data/VUA18 vua18
run data/VUAverb vuaverb
# MOH-X / TroFi are test-only transfer sets in the paper; we score the whole set
run data/MOH-X_all mohx
run data/TroFi_all trofi
# New data (generalisation)
run data/NEW/reviews reviews
run data/NEW/arxiv arxiv
run data/NEW/all new_all
