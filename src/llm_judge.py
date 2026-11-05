"""LLM-as-judge for the new data, in two roles.

1. annotate - the LLM labels every target word with MIPVU (M/L), blind to
   MelBERT and to the human labels. Comparing it with the human gold
   (src/agreement.py) tells us how far an LLM can stand in for a human
   annotator, and it is a strong zero-shot reference system for MelBERT.

2. judge - the LLM sees each MelBERT prediction (sentence, target word,
   MelBERT's label) and returns a verdict: CORRECT or INCORRECT plus a short
   reason. Comparing the verdicts with what the human gold says lets us
   measure how reliable LLM-as-judge evaluation would be on data that has no
   human labels, which is exactly the situation for most new domains.

Backends
  anthropic : needs ANTHROPIC_API_KEY  (pip install anthropic)
  openai    : needs OPENAI_API_KEY     (pip install openai)
  ollama    : a local model served by Ollama at http://localhost:11434
  manual    : no API. Writes prompt batches to results/llm_manual/<role>/batch_XX.txt.
              Paste each into a chat assistant and save its JSON reply as
              batch_XX.response.txt in the same folder, then rerun with
              --backend manual to parse them.

Usage
  python src/llm_judge.py annotate --backend anthropic --model claude-sonnet-5-5
  python src/llm_judge.py judge    --backend anthropic --model claude-sonnet-5-5 --preds results/preds/ours_new_all.csv
  python src/llm_judge.py annotate --backend manual          # step 1: writes prompts
  python src/llm_judge.py annotate --backend manual          # step 2: after saving responses, parses them
"""
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

MIPVU = """You are an expert metaphor annotator trained in MIPVU (Steen et al., 2010), the procedure used to build the VU Amsterdam Metaphor Corpus.
For a target word in a sentence:
1. Establish its contextual meaning in this sentence.
2. Establish its basic meaning: more concrete, bodily, precise or historically older (as in Macmillan / Longman learner's dictionaries, same part of speech).
3. If the contextual meaning differs from the basic meaning but can be understood by comparison with it, the word is metaphorical (M). Otherwise literal (L).
Conventional metaphors count as M (e.g. 'prices climbed', 'grasp an idea'). Personification of objects counts as M. If the contextual meaning is the basic meaning, or the word has only one sense, it is L."""

ANNOTATE_TASK = """Label each item below. Return ONLY a JSON array, one object per item, in the same order:
[{"item_id": "...", "basic_meaning": "<few words>", "contextual_meaning": "<few words>", "label": "M" or "L"}]

Items (the target word is in [brackets]):
"""

JUDGE_TASK = """A metaphor-detection system has labelled each target word below as METAPHORICAL or LITERAL.
For each item, judge whether the system's label is correct under MIPVU. Return ONLY a JSON array, one object per item, in the same order:
[{"item_id": "...", "verdict": "CORRECT" or "INCORRECT", "reason": "<one short sentence>"}]

Items (the target word is in [brackets]):
"""


def bracket(sentence, w_index):
    toks = sentence.split()
    toks[int(w_index)] = f"[{toks[int(w_index)]}]"
    return " ".join(toks)


def build_items(role, a):
    if role == "annotate":
        with open(a.items, encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        return [{"item_id": i["item_id"], "text": f'{i["item_id"]} | {bracket(i["sentence"], i["w_index"])} | target: {i["word"]} ({i["POS"]})'}
                for i in items]
    with open(a.preds, encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    out = []
    for p in preds:
        iid = f'{p["index"]}_{int(p["w_index"]):02d}'
        lab = "METAPHORICAL" if p["pred"] == "1" else "LITERAL"
        out.append({"item_id": iid, "text": f'{iid} | {bracket(p["sentence"], p["w_index"])} | target: {p["word"]} ({p["POS"]}) | system label: {lab}',
                    "pred": int(p["pred"]), "gold": int(p["gold"])})
    return out


def prompt_for(role, batch):
    task = ANNOTATE_TASK if role == "annotate" else JUDGE_TASK
    return task + "\n".join(x["text"] for x in batch)


def call_llm(backend, model, system, user):
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model=model, max_tokens=4096, temperature=0, system=system,
                                     messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    if backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(model=model, temperature=0,
                                           messages=[{"role": "system", "content": system},
                                                     {"role": "user", "content": user}])
        return r.choices[0].message.content
    if backend == "ollama":
        import urllib.request
        body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["message"]["content"]
    raise ValueError(backend)


def parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON array in response")
    return json.loads(m.group(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["annotate", "judge"])
    ap.add_argument("--backend", choices=["anthropic", "openai", "ollama", "manual"], default="manual")
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--items", default="data/annotation/items.csv")
    ap.add_argument("--preds", default="results/preds/ours_new_all.csv")
    ap.add_argument("--batch_size", type=int, default=25)
    a = ap.parse_args()

    items = build_items(a.role, a)
    batches = [items[i:i + a.batch_size] for i in range(0, len(items), a.batch_size)]
    raw_dir = Path(f"results/llm_{a.backend}/{a.role}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    pending = 0
    for k, batch in enumerate(batches):
        resp_file = raw_dir / f"batch_{k:02d}.response.txt"
        if not resp_file.exists():
            if a.backend == "manual":
                (raw_dir / f"batch_{k:02d}.txt").write_text(MIPVU + "\n\n" + prompt_for(a.role, batch))
                pending += 1
                continue
            for attempt in range(3):
                try:
                    resp_file.write_text(call_llm(a.backend, a.model, MIPVU, prompt_for(a.role, batch)))
                    break
                except Exception as e:  # network hiccups / rate limits
                    print(f"batch {k} attempt {attempt + 1} failed: {e}")
                    time.sleep(10 * (attempt + 1))
            else:
                print(f"giving up on batch {k}; rerun later to resume")
                continue
            print(f"batch {k + 1}/{len(batches)} done")
        try:
            for obj in parse_json_array(resp_file.read_text()):
                results[str(obj["item_id"]).strip()] = obj
        except Exception as e:
            print(f"could not parse {resp_file}: {e}  (fix or delete the file and rerun)")

    if pending:
        print(f"Wrote {pending} prompt files to {raw_dir}/. Paste each batch_XX.txt into the chat model, "
              f"save the JSON reply as batch_XX.response.txt, then rerun this command.")
        if not results:
            return

    missing = [x["item_id"] for x in items if x["item_id"] not in results]
    if missing:
        print(f"warning: {len(missing)} items without an LLM answer, e.g. {missing[:5]}")

    tag = a.model if a.backend != "manual" else "manual"
    meta = {"backend": a.backend, "model": tag, "n_items": len(items), "n_answered": len(items) - len(missing)}
    if a.role == "annotate":
        out = Path("data/annotation/labels_LLM.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "label", "low_conf", "note", "time"])
            for x in items:
                r = results.get(x["item_id"])
                if r and str(r.get("label", "")).upper().startswith(("M", "L")):
                    w.writerow([x["item_id"], 1 if str(r["label"]).upper().startswith("M") else 0, 0,
                                f'basic: {r.get("basic_meaning", "")}; context: {r.get("contextual_meaning", "")}', ""])
        Path("results/llm_annotate_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"LLM labels -> {out}. Next: python src/agreement.py --a data/NEW/gold_labels.csv --b {out} --name llm_vs_gold")
        return

    # judge: how good is the LLM judge, measured against the human gold?
    rows = []
    for x in items:
        r = results.get(x["item_id"])
        if not r:
            continue
        says_correct = str(r.get("verdict", "")).upper().startswith("C")
        truly_correct = x["pred"] == x["gold"]
        rows.append({"item_id": x["item_id"], "pred": x["pred"], "gold": x["gold"],
                     "judge_correct": int(says_correct), "truly_correct": int(truly_correct),
                     "reason": r.get("reason", "")})
    with open("results/llm_judge_verdicts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    pos = [r for r in rows if r["pred"] == 1]
    judge_acc = sum(r["judge_correct"] == r["truly_correct"] for r in rows) / n
    true_acc = sum(r["truly_correct"] for r in rows) / n
    est_acc = sum(r["judge_correct"] for r in rows) / n
    true_prec = sum(r["truly_correct"] for r in pos) / len(pos) if pos else float("nan")
    est_prec = sum(r["judge_correct"] for r in pos) / len(pos) if pos else float("nan")
    # judge-estimated recall: of items the judge believes are metaphors (pred=1 & correct, or pred=0 & incorrect)
    judge_gold = [r["pred"] if r["judge_correct"] else 1 - r["pred"] for r in rows]
    tp = sum(1 for r, g in zip(rows, judge_gold) if r["pred"] == 1 and g == 1)
    fn = sum(1 for r, g in zip(rows, judge_gold) if r["pred"] == 0 and g == 1)
    fp = sum(1 for r, g in zip(rows, judge_gold) if r["pred"] == 1 and g == 0)
    est_p = tp / (tp + fp) if tp + fp else 0
    est_r = tp / (tp + fn) if tp + fn else 0
    est_f1 = 2 * est_p * est_r / (est_p + est_r) if est_p + est_r else 0
    ttp = sum(1 for r in rows if r["pred"] == 1 and r["gold"] == 1)
    tfp = sum(1 for r in rows if r["pred"] == 1 and r["gold"] == 0)
    tfn = sum(1 for r in rows if r["pred"] == 0 and r["gold"] == 1)
    true_f1 = 2 * ttp / (2 * ttp + tfp + tfn) if ttp else 0.0
    summary = {**meta, "n_judged": n,
               "judge_agreement_with_gold_verdict": round(100 * judge_acc, 1),
               "melbert_accuracy_true": round(100 * true_acc, 1),
               "melbert_accuracy_judge_estimate": round(100 * est_acc, 1),
               "melbert_precision_true": round(100 * true_prec, 1),
               "melbert_precision_judge_estimate": round(100 * est_prec, 1),
               "melbert_f1_true": round(100 * true_f1, 1),
               "melbert_f1_judge_estimate": round(100 * est_f1, 1)}
    Path("results/llm_judge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Quantity (MelBERT on new data) & Human gold & LLM judge \\", r"\midrule",
             f"Accuracy & {100*true_acc:.1f} & {100*est_acc:.1f} \\\\",
             f"Precision & {100*true_prec:.1f} & {100*est_prec:.1f} \\\\",
             f"F1 & {100*true_f1:.1f} & {100*est_f1:.1f} \\\\",
             r"\midrule",
             f"\\multicolumn{{2}}{{l}}{{Judge verdict = gold verdict (\\%)}} & {100*judge_acc:.1f} \\\\",
             r"\bottomrule", r"\end{tabular}"]
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/tables/llm_judge.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
