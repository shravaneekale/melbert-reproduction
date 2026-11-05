"""Terminal annotation tool for MIPVU-style word-level metaphor labelling.

Each annotator works independently and never sees MelBERT's predictions, the
LLM's labels or the other annotator's labels. Progress is saved after every
decision, so you can quit (q) and resume at any time.

Keys
  m  metaphorical (MRW)          l  literal
  m? / l?  same, but low confidence (flagged for discussion)
  b  go back one item            q  save and quit
  n  add a free-text note to the previous decision

Usage
  python src/annotate.py --annotator A1                 # all items
  python src/annotate.py --annotator A2 --double_only   # agreement subset only

Read annotation/guidelines.md before starting.
"""
import argparse
import csv
import datetime as dt
import os
import random
from pathlib import Path

BOLD, YEL, DIM, END = "\033[1m", "\033[93m", "\033[2m", "\033[0m"


def load_items(path, double_only, seed):
    with open(path, encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    if double_only:
        items = [i for i in items if i["double"] == "1"]
    # shuffle sentence order (fixed seed) but keep words of a sentence together, in order
    sents = sorted({i["sent_id"] for i in items})
    random.Random(seed).shuffle(sents)
    rank = {s: k for k, s in enumerate(sents)}
    items.sort(key=lambda i: (rank[i["sent_id"]], int(i["w_index"])))
    return items


def load_labels(path):
    labels = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                labels[r["item_id"]] = r
    return labels


def save_labels(path, labels):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "label", "low_conf", "note", "time"])
        w.writeheader()
        for r in labels.values():
            w.writerow(r)
    os.replace(tmp, path)


def show(item, k, n):
    toks = item["sentence"].split()
    wi = int(item["w_index"])
    toks[wi] = f"{BOLD}{YEL}[{toks[wi]}]{END}"
    print("\n" + "-" * 78)
    print(f"{DIM}{k + 1}/{n}  {item['item_id']}  domain={item['domain']}{END}")
    print("  " + " ".join(toks))
    print(f"  target: {BOLD}{item['word']}{END}  (lemma: {item['lemma']}, POS: {item['POS']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True, help="e.g. A1 or A2")
    ap.add_argument("--items", default="data/annotation/items.csv")
    ap.add_argument("--double_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    out = Path(f"data/annotation/labels_{a.annotator}.csv")
    items = load_items(a.items, a.double_only, a.seed)
    labels = load_labels(out)
    n = len(items)
    k = next((j for j, it in enumerate(items) if it["item_id"] not in labels), n)
    print(f"{len(labels)} labels already saved in {out}; {n - k} remaining.")

    while k < n:
        item = items[k]
        show(item, k, n)
        prev = labels.get(item["item_id"])
        if prev:
            print(f"  {DIM}(current label: {prev['label']}{'?' if prev['low_conf'] == '1' else ''}){END}")
        ans = input("  [m]etaphor / [l]iteral / m? l? / [b]ack / [n]ote / [q]uit > ").strip().lower()
        if ans == "q":
            break
        if ans == "b":
            k = max(0, k - 1)
            continue
        if ans == "n":
            target = items[max(0, k - 1)]["item_id"]
            if target in labels:
                labels[target]["note"] = input("  note: ").strip()
                save_labels(out, labels)
            continue
        if ans.rstrip("?") not in ("m", "l"):
            print("  ?? type m, l, m?, l?, b, n or q")
            continue
        labels[item["item_id"]] = {
            "item_id": item["item_id"],
            "label": 1 if ans.startswith("m") else 0,
            "low_conf": int(ans.endswith("?")),
            "note": prev["note"] if prev else "",
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        }
        save_labels(out, labels)
        k += 1

    done = sum(1 for it in items if it["item_id"] in labels)
    print(f"\nSaved {out}: {done}/{n} items labelled.")


if __name__ == "__main__":
    main()
