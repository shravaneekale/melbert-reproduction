"""Create a sentence-level subsample of a MelBERT training set.

Useful for running a reduced replication on a laptop (e.g. Apple Silicon)
when a GPU is not available. Sampling is done by *sentence* (all target rows
of a sampled sentence are kept), so the token-level class balance and
sentence context are preserved. test.tsv is copied unchanged so evaluation is
always on the full official test set.

Usage:
    python src/make_subsample.py --src data/VUA18 --dst data/VUA18_sub10k --n_sentences 10000
"""
import argparse
import csv
import random
import shutil
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n_sentences", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)

    with open(src / "train.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        header = next(reader)
        by_sent = OrderedDict()
        for row in reader:
            # VUA rows of the same sentence share index + sentence text
            by_sent.setdefault((row[0], row[2]), []).append(row)

    keys = list(by_sent)
    random.Random(a.seed).shuffle(keys)
    keep = set(keys[: a.n_sentences])
    n_rows = 0
    with open(dst / "train.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quotechar=None, quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(header)
        for k in by_sent:  # keep original order
            if k in keep:
                for row in by_sent[k]:
                    w.writerow(row)
                    n_rows += 1
    for name in ("test.tsv", "dev.tsv"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    print(f"{len(keys)} sentences in source; kept {len(keep)} sentences = {n_rows} target rows -> {dst}")


if __name__ == "__main__":
    main()
