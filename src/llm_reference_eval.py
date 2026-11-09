"""LLM-as-judge, reference-free evaluation of MelBERT on the new data.

Question: if we had NO human labels, and used an LLM's MIPVU judgements as
the reference, how close would the estimated precision / recall / F1 of
MelBERT be to the scores against the human gold standard?

Each LLM judged every target blind (labels_LLM.csv = LLM1, labels_LLM2.csv
= LLM2). A target counts as correctly predicted by MelBERT, in the judge's
view, when MelBERT's label equals the judge's label.

Caveat built into the output: LLM1 also suggested analyses during
adjudication, so the gold standard is not independent of LLM1 and its
estimate is optimistic (circular). LLM2 took no part in building the gold.

Usage:
    python src/llm_reference_eval.py --preds results/preds/ours_new_all.csv
"""
import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


def labels(path):
    with open(path, encoding="utf-8") as f:
        return {r["item_id"]: int(r["label"]) for r in csv.DictReader(f) if r["label"] in ("0", "1")}


def prf(pred, ref, ids):
    tp = sum(1 for i in ids if pred[i] and ref[i])
    fp = sum(1 for i in ids if pred[i] and not ref[i])
    fn = sum(1 for i in ids if not pred[i] and ref[i])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": round(100 * p, 1), "R": round(100 * r, 1), "F1": round(100 * f, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds/ours_new_all.csv")
    ap.add_argument("--gold", default="data/NEW/gold.csv")
    ap.add_argument("--judges", nargs="+", default=["LLM1=data/annotation/labels_LLM.csv",
                                                    "LLM2=data/annotation/labels_LLM2.csv"])
    a = ap.parse_args()

    with open(a.gold, encoding="utf-8") as f:
        g = list(csv.DictReader(f))
    gold = {r["item_id"]: int(r["label"]) for r in g}
    dom = {r["item_id"]: r["domain"] for r in g}
    with open(a.preds, encoding="utf-8") as f:
        pred = {f'{r["index"]}_{int(r["w_index"]):02d}': int(r["pred"]) for r in csv.DictReader(f)}
    ids_all = sorted(i for i in gold if i in pred)

    judges = {}
    for spec in a.judges:
        name, path = spec.split("=", 1)
        if Path(path).exists():
            judges[name] = labels(path)

    out = {}
    for d in ("reviews", "arxiv", "all"):
        ids = [i for i in ids_all if d == "all" or dom[i] == d]
        row = {"n": len(ids), "human_gold": prf(pred, gold, ids)}
        for name, lab in judges.items():
            row[name] = prf(pred, lab, ids)
            row[name]["kappa_vs_gold"] = round(float(cohen_kappa_score([gold[i] for i in ids], [lab[i] for i in ids])), 3)
            row[name]["pctM"] = round(100 * sum(lab[i] for i in ids) / len(ids), 1)
        out[d] = row
    Path("results").mkdir(exist_ok=True)
    Path("results/llm_reference_eval.json").write_text(json.dumps(out, indent=2))

    names = list(judges)
    lines = [r"\begin{tabular}{l" + "r" * (1 + 2 * len(names)) + "}", r"\toprule",
             "Reference & " + " & ".join([r"\multicolumn{2}{c}{" + n + "}" for n in names]) + r" & Human \\",
             "Test set & " + " & ".join(["\\%M & F1"] * len(names)) + r" & F1 \\", r"\midrule"]
    for d, lab in (("reviews", "Reviews"), ("arxiv", "arXiv"), ("all", "Both")):
        r = out[d]
        cells = [lab] + [f"{r[n]['pctM']:.1f} & {r[n]['F1']:.1f}" for n in names] + [f"{r['human_gold']['F1']:.1f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/llm_reference_eval.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
