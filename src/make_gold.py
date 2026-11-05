"""Build the final gold-standard labels and MelBERT-format test files.

Gold label for each item:
  * the adjudicated label, if the item was double-annotated and A1/A2 disagreed;
  * otherwise the primary annotator's (A1) label.

Outputs
  data/NEW/gold.csv                   all items + gold label + provenance
  data/NEW/reviews/test.tsv           MelBERT input, one row per target word
  data/NEW/arxiv/test.tsv
  data/NEW/all/test.tsv               both domains

Usage:
    python src/make_gold.py --primary data/annotation/labels_A1.csv \
        --adjudication data/annotation/adjudication.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from agreement import read_items, read_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/annotation/items.csv")
    ap.add_argument("--primary", default="data/annotation/labels_A1.csv")
    ap.add_argument("--adjudication", default=None)
    ap.add_argument("--out_dir", default="data/NEW")
    a = ap.parse_args()

    items = read_items(a.items)
    prim = read_labels(a.primary)
    adj = {}
    if a.adjudication and Path(a.adjudication).exists():
        with open(a.adjudication, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["adjudicated_label"].strip() in ("0", "1"):
                    adj[r["item_id"]] = int(r["adjudicated_label"])
                else:
                    raise SystemExit(f"Item {r['item_id']} in {a.adjudication} has no adjudicated_label yet.")

    missing = [i for i in items if i not in prim]
    if missing:
        raise SystemExit(f"{len(missing)} items have no label from the primary annotator, e.g. {missing[:5]}")

    out = Path(a.out_dir)
    gold_rows = []
    for iid, it in items.items():
        lab, src = (adj[iid], "adjudicated") if iid in adj else (prim[iid], "A1")
        gold_rows.append({**it, "label": lab, "label_source": src})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gold.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gold_rows[0].keys()))
        w.writeheader()
        w.writerows(gold_rows)

    with open(out / "gold_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "label", "low_conf", "note", "time"])
        for r in gold_rows:
            w.writerow([r["item_id"], r["label"], 0, r["label_source"], ""])

    def write_tsv(rows, d):
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "test.tsv", "w", encoding="utf-8") as f:
            f.write("index\tlabel\tsentence\tPOS\tw_index\n")
            for r in rows:
                assert "\t" not in r["sentence"]
                f.write(f"{r['sent_id']}\t{r['label']}\t{r['sentence']}\t{r['POS']}\t{r['w_index']}\n")

    for dom in sorted({r["domain"] for r in gold_rows}):
        write_tsv([r for r in gold_rows if r["domain"] == dom], out / dom)
    write_tsv(gold_rows, out / "all")

    # LaTeX statistics table for the report
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Domain & Sent. & Targets & \%M & \%M V & \%M N & \%M Adj & \%M Adv & Double \\", r"\midrule"]
    def pct(rows):
        return f"{100 * sum(r['label'] for r in rows) / len(rows):.1f}" if rows else "--"
    groups = [(d.capitalize() if d != "arxiv" else "arXiv", [r for r in gold_rows if r["domain"] == d])
              for d in sorted({r["domain"] for r in gold_rows}, reverse=True)] + [("All", gold_rows)]
    for name, rows in groups:
        if name == "All":
            lines.append(r"\midrule")
        cells = [name, str(len({r["sent_id"] for r in rows})), str(len(rows)), pct(rows)]
        cells += [pct([r for r in rows if r["POS"] == p]) for p in ("VERB", "NOUN", "ADJ", "ADV")]
        cells.append(str(sum(int(r["double"]) for r in rows)))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/new_stats.tex").write_text("\n".join(lines) + "\n")

    print(f"gold: {len(gold_rows)} items ({len(adj)} adjudicated) -> {out/'gold.csv'}")
    for dom in sorted({r["domain"] for r in gold_rows}):
        d = [r for r in gold_rows if r["domain"] == dom]
        m = sum(r["label"] for r in d)
        print(f"  {dom}: {len(d)} items, {len({r['sent_id'] for r in d})} sentences, "
              f"{m} metaphorical ({100*m/len(d):.1f}%)  by POS: "
              f"{dict(Counter(r['POS'] for r in d if r['label'] == 1))}")


if __name__ == "__main__":
    main()
