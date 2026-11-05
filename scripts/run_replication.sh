#!/usr/bin/env bash
# Full replication of MelBERT on VUA-18 with the paper's hyper-parameters
# (RoBERTa-base, 3 epochs, batch 32, lr 3e-5, warmup 2 epochs, class weight 3,
# max_seq_length 150, seed 42 -- the defaults in the official main_config.cfg).
#
# Recommended on a CUDA GPU (Google Colab T4/L4/A100). On a Mac it also runs
# (MPS backend), but a full run takes a day or more; use run_replication_mac.sh
# for a reduced run instead.
#
# Usage (from repo root):  bash scripts/run_replication.sh [seed]
set -euo pipefail
SEED=${1:-42}
cd external/MelBERT
export PYTORCH_ENABLE_MPS_FALLBACK=1

python main.py \
  --data_dir data/VUA18 \
  --task_name vua \
  --model_type MELBERT \
  --bert_model roberta-base \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --warmup_epoch 2 \
  --num_train_epoch 3 \
  --eval_batch_size 64 \
  --seed "$SEED" 2>&1 | tee "../../results/train_vua18_seed${SEED}.log"

echo "Checkpoint saved under external/MelBERT/saves/roberta-base/ (newest folder)."
