#!/usr/bin/env bash
set -euo pipefail
pip install -q gdown
mkdir -p /tmp/melbert_dl checkpoints/official
if [ ! -d data/VUA18 ]; then
  gdown -q "18hemekvUuOw-qkQhWv4F6qFnis6lkaog" -O /tmp/melbert_dl/data.zip
  unzip -q -o /tmp/melbert_dl/data.zip -d /tmp/melbert_dl/data
  for name in VUA18 VUA20 VUAverb MOH-X TroFi; do
    src=$(find /tmp/melbert_dl/data -type d -name "$name" | head -1)
    if [ -n "$src" ] && [ ! -e "data/$name" ]; then cp -R "$src" "data/$name"; echo "data/$name ok"; fi
  done
fi
python src/prepare_transfer_sets.py --data_root data
if [ ! -f checkpoints/official/melbert_ckpt/pytorch_model.bin ]; then
  gdown -q "1RKyP-_kje6xWlNuXeoSIHZnYxUyXnupZ" -O /tmp/melbert_dl/ckpt.zip
  unzip -q -o /tmp/melbert_dl/ckpt.zip -d checkpoints/official
fi
wc -l data/VUA18/*.tsv
ls checkpoints/official/*
