"""Compute all evaluation numbers, LaTeX tables and figures for the report.

Reads the per-token prediction files written by src/predict.py
(results/preds/<tag>_<set>.csv) and, if present, the LLM annotator labels.

Produces
  results/metrics.json                 every number below
  results/tables/replication.tex       paper vs our replication (VUA-18, VUA-Verb, MOH-X, TroFi)
  results/tables/new_data.tex          VUA-18 (same POS set) vs new domains, with 95% CIs
  results/tables/per_pos.tex           F1 by POS, paper / ours-VUA / new domains
  results/figures/f1_by_pos.pdf
  results/errors_<tag>_<set>.csv       false positives / negatives for error analysis

Usage:
    python src/evaluate.py --tag ours [--tag2 official] [--llm_labels data/annotation/labels_LLM.csv]
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

CONTENT_POS = ["VERB", "NOUN", "ADJ", "ADV"]

# Choi et al. (2021): Table 2 (VUA-18, VUA-Verb), Table 6 (zero-shot, trained on VUA-20),
# Table 5 (VUA-18 F1 by POS). MelBERT rows, single model (no bagging), mean of 5 seeds.
PAPER = {
    "vua18": {"precision": 80.1, "recall": 76.9, "f1": 78.5},
    "vuaverb": {"precision": 78.7, "recall": 72.9, "f1": 75.7},
    "mohx": {"precision": 79.3, "recall": 79.7, "f1": 79.2},
    "trofi": {"precision": 53.4, "recall": 74.1, "f1": 62.0},
}
PAPER_POS_F1 = {"VERB": 75.1, "ADJ": 64.4, "ADV": 74.6, "NOUN": 70.7}
SET_NAMES = {"vua18": "VUA-18", "vuaverb": "VUA-Verb", "mohx": "MOH-X$^\\dagger$", "trofi": "TroFi$^\\dagger$",
             "reviews": "Reviews (new)", "arxiv": "arXiv (new)", "new_all": "New (both)"}


def read_preds(path):
    with open(path, encoding="utf-8") as f:
        return [{**r, "gold": int(r["gold"]), "pred": int(r["pred"]), "p": float(r["p_metaphor"])}
                for r in csv.DictReader(f)]


def prf(rows):
    tp = sum(1 for r in rows if r["pred"] == 1 and r["gold"] == 1)
    fp = sum(1 for r in rows if r["pred"] == 1 and r["gold"] == 0)
    fn = sum(1 for r in rows if r["pred"] == 0 and r["gold"] == 1)
    n = len(rows)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    acc = sum(1 for x in rows if x["pred"] == x["gold"]) / n if n else 0.0
    return {"n": n, "acc": 100 * acc, "precision": 100 * p, "recall": 100 * r, "f1": 100 * f,
            "pct_M_gold": 100 * sum(x["gold"] for x in rows) / n if n else 0.0,
            "pct_M_pred": 100 * sum(x["pred"] for x in rows) / n if n else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def bootstrap_f1(rows, iters=1000, seed=0):
    """Cluster (sentence-level) bootstrap 95% CI for F1: target words in the same
    sentence are not independent, so we resample whole sentences."""
    by_sent = defaultdict(list)
    for r in rows:
        by_sent[r["index"]].append(r)
    sents = list(by_sent.values())
    if len(sents) < 5:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [x for _ in sents for x in rng.choice(sents)]
        vals.append(prf(sample)["f1"])
    vals.sort()
    return [vals[int(0.025 * iters)], vals[int(0.975 * iters) - 1]]


def summarise(rows, with_ci=False):
    m = prf(rows)
    if with_ci:
        m["f1_ci95"] = bootstrap_f1(rows)
    return m


def fmt(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ours", help="prediction tag (results/preds/<tag>_*.csv)")
    ap.add_argument("--tag2", default=None, help="optional second checkpoint, e.g. official")
    ap.add_argument("--llm_labels", default=None, help="LLM annotator labels for the new data")
    ap.add_argument("--gold", default="data/NEW/gold.csv")
    a = ap.parse_args()

    pdir = Path("results/preds")
    out = {}
    tags = [t for t in (a.tag, a.tag2) if t]
    for tag in tags:
        out[tag] = {}
        for name in ["vua18", "vuaverb", "mohx", "trofi", "reviews", "arxiv", "new_all"]:
            fp = pdir / f"{tag}_{name}.csv"
            if not fp.exists():
                continue
            rows = read_preds(fp)
            is_new = name in ("reviews", "arxiv", "new_all")
            res = {"all": summarise(rows, with_ci=is_new)}
            if name == "vua18":
                cw = [r for r in rows if r["POS"] in CONTENT_POS]
                res["content_words"] = summarise(cw, with_ci=True)
            if name in ("vua18", "reviews", "arxiv", "new_all"):
                res["by_pos"] = {p: summarise([r for r in rows if r["POS"] == p]) for p in CONTENT_POS}
            out[tag][name] = res
            if is_new or name == "vua18":
                errs = sorted([r for r in rows if r["pred"] != r["gold"]], key=lambda r: -abs(r["p"] - 0.5))
                with open(f"results/errors_{tag}_{name}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["type", "word", "POS", "p_metaphor", "sentence", "index"])
                    for r in errs[:300] if name == "vua18" else errs:
                        w.writerow(["FP" if r["pred"] else "FN", r["word"], r["POS"], f"{r['p']:.3f}",
                                    r["sentence"], r["index"]])

    # LLM annotator as a zero-shot reference system on the new data
    if a.llm_labels and Path(a.llm_labels).exists() and Path(a.gold).exists():
        with open(a.llm_labels, encoding="utf-8") as f:
            llm = {r["item_id"]: int(r["label"]) for r in csv.DictReader(f) if r["label"] in ("0", "1")}
        with open(a.gold, encoding="utf-8") as f:
            gold = list(csv.DictReader(f))
        rows = [{"gold": int(g["label"]), "pred": llm[g["item_id"]], "index": g["sent_id"],
                 "POS": g["POS"], "domain": g["domain"]} for g in gold if g["item_id"] in llm]
        out["llm"] = {"new_all": {"all": summarise(rows, True)}}
        for dom in ("reviews", "arxiv"):
            d = [r for r in rows if r["domain"] == dom]
            if d:
                out["llm"][dom] = {"all": summarise(d, True),
                                   "by_pos": {p: summarise([r for r in d if r["POS"] == p]) for p in CONTENT_POS}}

    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(out, indent=2))
    tdir = Path("results/tables")
    tdir.mkdir(parents=True, exist_ok=True)
    main_tag = out.get(a.tag, {})

    # ---- Table: replication ----
    cols = "l" + "rrr" * (1 + len(tags)) + "r"
    head1 = r"& \multicolumn{3}{c}{Paper} " + "".join(
        rf"& \multicolumn{{3}}{{c}}{{{'Ours' if t == a.tag else 'Official ckpt'}}} " for t in tags) + r"& $\Delta$F1 \\"
    head2 = "Test set " + "& P & R & F1 " * (1 + len(tags)) + r"& \\"
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head1, head2, r"\midrule"]
    for name in ["vua18", "vuaverb", "mohx", "trofi"]:
        if name not in main_tag:
            continue
        pp = PAPER[name]
        cells = [SET_NAMES[name], fmt(pp["precision"]), fmt(pp["recall"]), fmt(pp["f1"])]
        for t in tags:
            m = out.get(t, {}).get(name, {}).get("all")
            cells += [fmt(m and m["precision"]), fmt(m and m["recall"]), fmt(m and m["f1"])]
        cells.append(f"{main_tag[name]['all']['f1'] - pp['f1']:+.1f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "replication.tex").write_text("\n".join(lines) + "\n")

    # ---- Table: new data ----
    lines = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
             r"Test set & $n$ & \%M gold & \%M pred & P & R & F1 & 95\% CI \\", r"\midrule"]

    def row(label, m):
        ci = m.get("f1_ci95")
        ci_s = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "--"
        return (f"{label} & {m['n']} & {m['pct_M_gold']:.1f} & {m['pct_M_pred']:.1f} & "
                f"{m['precision']:.1f} & {m['recall']:.1f} & {m['f1']:.1f} & {ci_s} \\\\")

    if "vua18" in main_tag:
        lines.append(row(r"VUA-18 test (V/N/Adj/Adv)", main_tag["vua18"]["content_words"]))
    for name in ["reviews", "arxiv", "new_all"]:
        if name in main_tag:
            lines.append(row(SET_NAMES[name], main_tag[name]["all"]))
    if "llm" in out:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{8}{l}{\textit{LLM annotator (zero-shot reference)}} \\")
        for name in ["reviews", "arxiv", "new_all"]:
            if name in out["llm"]:
                lines.append(row(SET_NAMES[name], out["llm"][name]["all"]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "new_data.tex").write_text("\n".join(lines) + "\n")

    # ---- Table: per POS ----
    sets = [s for s in ["vua18", "reviews", "arxiv"] if s in main_tag]
    lines = [r"\begin{tabular}{l" + "r" * (1 + len(sets)) + "}", r"\toprule",
             "POS & Paper (VUA-18) & " + " & ".join(SET_NAMES[s].replace(" (new)", "") for s in sets) + r" \\",
             r"\midrule"]
    for p in CONTENT_POS:
        cells = [p.capitalize(), fmt(PAPER_POS_F1[p])]
        for s in sets:
            m = main_tag[s]["by_pos"][p]
            cells.append(f"{m['f1']:.1f} ({m['n']})" if m["n"] else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tdir / "per_pos.tex").write_text("\n".join(lines) + "\n")

    # ---- Figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if sets:
            fig, ax = plt.subplots(figsize=(4.6, 2.4))
            width = 0.8 / (len(sets) + 1)
            xs = range(len(CONTENT_POS))
            colours = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
            ax.bar([x - 0.4 + width / 2 for x in xs], [PAPER_POS_F1[p] for p in CONTENT_POS], width,
                   label="Paper (VUA-18)", color="#BBBBBB")
            for k, s in enumerate(sets):
                ax.bar([x - 0.4 + width * (k + 1.5) for x in xs],
                       [main_tag[s]["by_pos"][p]["f1"] for p in CONTENT_POS], width,
                       label=SET_NAMES[s].replace(" (new)", "").replace("$^\\dagger$", ""), color=colours[k])
            ax.set_xticks(list(xs))
            ax.set_xticklabels([p.capitalize() for p in CONTENT_POS])
            ax.set_ylabel("F1")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=7, ncol=2, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            Path("results/figures").mkdir(parents=True, exist_ok=True)
            fig.savefig("results/figures/f1_by_pos.pdf")
    except ImportError:
        pass

    for t in out:
        for name, res in out[t].items():
            m = res["all"]
            ci = m.get("f1_ci95")
            print(f"{t:9s} {name:8s} n={m['n']:6d} P={m['precision']:5.1f} R={m['recall']:5.1f} "
                  f"F1={m['f1']:5.1f}" + (f" CI=[{ci[0]:.1f},{ci[1]:.1f}]" if ci else ""))
    print("tables -> results/tables/, metrics -> results/metrics.json")


if __name__ == "__main__":
    main()
