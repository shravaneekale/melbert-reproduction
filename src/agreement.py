"""Inter-annotator agreement and adjudication sheet.

Compares two label files on the items both have labelled and reports
observed agreement, Cohen's kappa (overall / per domain / per POS) and the
prevalence of metaphor for each rater. Disagreements are written to a CSV for
adjudication (discussion between annotators, or a third rater).

The same script is used to compare the human gold labels with the LLM
annotator (src/llm_judge.py annotate) - pass any two label CSVs.

Usage:
    python src/agreement.py --a data/annotation/labels_A1.csv --b data/annotation/labels_A2.csv \
        --name human --disagreements data/annotation/adjudication.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


def read_labels(path):
    with open(path, encoding="utf-8") as f:
        return {r["item_id"]: int(r["label"]) for r in csv.DictReader(f) if r["label"] in ("0", "1")}


def read_items(path):
    with open(path, encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def stats(pairs):
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    obs = sum(x == y for x, y in pairs) / n
    kappa = cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else float("nan")
    return {
        "n": n,
        "observed_agreement": round(obs, 4),
        "cohen_kappa": round(float(kappa), 4),
        "pct_metaphor_a": round(100 * sum(a) / n, 1),
        "pct_metaphor_b": round(100 * sum(b) / n, 1),
        "both_M": sum(1 for x, y in pairs if x == y == 1),
        "both_L": sum(1 for x, y in pairs if x == y == 0),
        "a_M_b_L": sum(1 for x, y in pairs if x == 1 and y == 0),
        "a_L_b_M": sum(1 for x, y in pairs if x == 0 and y == 1),
    }


def landis_koch(k):
    if k != k:
        return "n/a"
    for lim, name in [(0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if k <= lim:
            return name
    return "almost perfect"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--items", default="data/annotation/items.csv")
    ap.add_argument("--name", default="human", help="tag used for output files")
    ap.add_argument("--disagreements", default=None, help="write adjudication CSV here")
    a = ap.parse_args()

    A, B, items = read_labels(a.a), read_labels(a.b), read_items(a.items)
    common = sorted(set(A) & set(B) & set(items))
    groups = defaultdict(list)
    for iid in common:
        pair = (A[iid], B[iid])
        groups["all"].append(pair)
        groups[f"domain={items[iid]['domain']}"].append(pair)
        groups[f"POS={items[iid]['POS']}"].append(pair)

    order = lambda g: (0 if g == "all" else 1 if g.startswith("domain") else 2, g)
    res = {g: stats(groups[g]) for g in sorted(groups, key=order)}
    for g, s in res.items():
        if s.get("n"):
            s["interpretation"] = landis_koch(s["cohen_kappa"])
            print(f"{g:16s} n={s['n']:4d}  agree={s['observed_agreement']:.3f}  "
                  f"kappa={s['cohen_kappa']:.3f} ({s['interpretation']})  "
                  f"%M a={s['pct_metaphor_a']} b={s['pct_metaphor_b']}")

    Path("results").mkdir(exist_ok=True)
    with open(f"results/agreement_{a.name}.json", "w") as f:
        json.dump(res, f, indent=2)

    # LaTeX table
    rows = [g for g in res if res[g].get("n")]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Subset & $n$ & Agree. & $\kappa$ & \%M (A/B) \\", r"\midrule"]
    for g in rows:
        s = res[g]
        label = g.replace("domain=", "").replace("POS=", "").replace("_", r"\_")
        kap = "--" if s["cohen_kappa"] != s["cohen_kappa"] else f"{s['cohen_kappa']:.2f}"
        lines.append(f"{label} & {s['n']} & {s['observed_agreement']:.2f} & {kap} & "
                     f"{s['pct_metaphor_a']:.0f}/{s['pct_metaphor_b']:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path(f"results/tables/agreement_{a.name}.tex").write_text("\n".join(lines) + "\n")

    if a.disagreements and Path(a.disagreements).exists():
        with open(a.disagreements, encoding="utf-8") as f:
            if any(r.get("adjudicated_label", "").strip() for r in csv.DictReader(f)):
                a.disagreements = a.disagreements.replace(".csv", "_new.csv")
                print(f"existing adjudication file already has decisions; writing to {a.disagreements} instead")
    if a.disagreements:
        with open(a.disagreements, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "domain", "POS", "word", "sentence", "label_a", "label_b",
                        "adjudicated_label", "reason"])
            n = 0
            for iid in common:
                if A[iid] != B[iid]:
                    it = items[iid]
                    toks = it["sentence"].split()
                    toks[int(it["w_index"])] = f"[{toks[int(it['w_index'])]}]"
                    w.writerow([iid, it["domain"], it["POS"], it["word"], " ".join(toks), A[iid], B[iid], "", ""])
                    n += 1
        print(f"{n} disagreements written to {a.disagreements} - fill in adjudicated_label (0/1) and reason.")


if __name__ == "__main__":
    main()
