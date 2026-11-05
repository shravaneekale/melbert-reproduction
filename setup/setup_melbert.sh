#!/usr/bin/env bash
# Clone the official MelBERT implementation at a pinned commit and apply the
# compatibility patches. Run from the repository root:
#     bash setup/setup_melbert.sh
set -euo pipefail

MELBERT_COMMIT=11b0f78dae93c71c56b805dc28d810f6050c86c4
DEST=external/MelBERT

mkdir -p external
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/jin530/MelBERT.git "$DEST"
fi
git -C "$DEST" checkout -q "$MELBERT_COMMIT"
python setup/patch_melbert.py "$DEST"

# MelBERT expects its data under <repo>/data. We keep all data in our own
# data/ folder and symlink it in so paths like data/VUA18 work in both places.
if [ ! -e "$DEST/data" ]; then
  ln -s "$(pwd)/data" "$DEST/data"
fi
echo "MelBERT ready at $DEST (commit $MELBERT_COMMIT)"
