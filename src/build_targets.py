"""Turn sampled sentences into word-level annotation items in the same shape
as VUA / MelBERT input:  one item per (sentence, target word).

* Sentences are whitespace-tokenised exactly as MelBERT does
  (sentence.split()), so `w_index` can be passed straight to MelBERT.
* Each whitespace token gets a Universal POS tag (VUA-18 uses UPOS) from
  spaCy, by aligning character offsets.
* Targets = content words (VERB, NOUN, ADJ, ADV). Proper nouns, auxiliaries,
  numbers and function words are excluded: almost all metaphor in VUA is
  carried by content words, and restricting targets keeps manual annotation
  feasible. When comparing to VUA we restrict VUA to the same POS set.
* A random subset of sentences (default 30%) is marked `double=1`; these
  are annotated independently by a second annotator to measure agreement.

Usage:
    python src/build_targets.py \
        --inputs data/raw/reviews_sentences.csv data/raw/arxiv_sentences.csv \
        --out data/annotation/items.csv --double_frac 0.3
"""
import argparse
import csv
import random
from pathlib import Path

import spacy

CONTENT_POS = {"VERB", "NOUN", "ADJ", "ADV"}


def align_pos(sentence: str, doc):
    """Return [(ws_token, POS, lemma)] for each whitespace token."""
    out, pos = [], 0
    for tok in sentence.split():
        start = sentence.index(tok, pos)
        end = start + len(tok)
        pos = end
        spans = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
        # prefer the first alphabetic, non-punctuation spaCy token inside the span
        main = next((t for t in spans if t.is_alpha), None) or next(
            (t for t in spans if not t.is_punct), spans[0] if spans else None
        )
        if main is None:
            out.append((tok, "X", tok.lower()))
        else:
            out.append((tok, main.pos_, main.lemma_.lower()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--double_frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=8240)
    a = ap.parse_args()

    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(a.seed)
    sentences = []
    for fp in a.inputs:
        with open(fp, encoding="utf-8") as f:
            sentences.extend(csv.DictReader(f))

    # stratify the double-annotated subset by domain
    double = set()
    for dom in sorted({s["domain"] for s in sentences}):
        ids = [s["sent_id"] for s in sentences if s["domain"] == dom]
        rng.shuffle(ids)
        double.update(ids[: round(len(ids) * a.double_frac)])

    items = []
    for s in sentences:
        sent = " ".join(s["sentence"].split())
        toks = align_pos(sent, nlp(sent))
        for i, (tok, pos, lemma) in enumerate(toks):
            if pos not in CONTENT_POS:
                continue
            word = tok.strip(".,;:!?\"'()[]")
            if not any(c.isalpha() for c in word):
                continue
            items.append({
                "item_id": f"{s['sent_id']}_{i:02d}",
                "sent_id": s["sent_id"],
                "domain": s["domain"],
                "subdomain": s.get("subdomain", ""),
                "double": int(s["sent_id"] in double),
                "w_index": i,
                "word": tok,
                "lemma": lemma,
                "POS": pos,
                "sentence": sent,
            })

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0].keys()))
        w.writeheader()
        w.writerows(items)
    n_sent = len({i["sent_id"] for i in items})
    n_double = sum(i["double"] for i in items)
    print(f"{len(items)} target items from {n_sent} sentences "
          f"({n_double} items in the double-annotated subset) -> {a.out}")
    for dom in sorted({i["domain"] for i in items}):
        d = [i for i in items if i["domain"] == dom]
        by_pos = {p: sum(x["POS"] == p for x in d) for p in sorted(CONTENT_POS)}
        print(f"  {dom}: {len(d)} items  {by_pos}")


if __name__ == "__main__":
    main()
