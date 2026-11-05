#!/usr/bin/env bash
# Reduced replication for Apple Silicon (MPS) or CPU-only machines.
# Trains on a 10,000-sentence subsample of VUA-18 train (all target words in
# those sentences) and evaluates on the FULL VUA-18 test set.
# The 0.5-epoch warmup mirrors the pilot reported in the project proposal.
#
# Usage (from repo root):  bash scripts/run_replication_mac.sh [n_sentences] [epochs]
set -euo pipefail
N=${1:-10000}
EPOCHS=${2:-4}

python src/make_subsample.py --src data/VUA18 --dst "data/VUA18_sub${N}" --n_sentences "$N"
cd external/MelBERT
# MPS has no float64 support for some ops; this env var lets PyTorch fall back to CPU for them.
export PYTORCH_ENABLE_MPS_FALLBACK=1
python main.py \
  --data_dir "data/VUA18_sub${N}" \
  --task_name vua \
  --model_type MELBERT \
  --bert_model roberta-base \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --warmup_epoch 0.5 \
  --num_train_epoch "$EPOCHS" 2>&1 | tee "../../results/train_vua18_sub${N}.log"
