"""Sample candidate sentences for the new evaluation data from two public,
unlabelled corpora that MelBERT has not been evaluated on:

  reviews : Amazon product reviews (HuggingFace `fancyzhx/amazon_polarity`,
            Zhang et al. 2015, from McAuley & Leskovec 2013). Informal,
            evaluative written text - a register VUA does not contain.
  arxiv   : arXiv abstracts (HuggingFace `gfissore/arxiv-abstracts-2021`,
            CC0). Technical, terminology-dense scientific prose - a
            contrastive, expected-low-metaphor-density domain.

Neither corpus has metaphor labels; labels are added by our own MIPVU
annotation (see annotation/guidelines.md).

Sampling is streamed (no full download), shuffled with a fixed seed, takes at
most one sentence per source document (to maximise diversity), and applies
quality + privacy filters:
  * 8-40 whitespace tokens, mostly alphabetic, English-looking;
  * no URLs / e-mails / @handles / LaTeX maths;
  * no sentences containing a PERSON named entity (privacy: reviews often
    name family members, sellers, etc.);
  * exact-duplicate removal.

For arXiv we balance across top-level subject areas.

Usage:
    python src/sample_corpus.py --source reviews --n 120 --out data/raw/reviews_sentences.csv
    python src/sample_corpus.py --source arxiv   --n 120 --out data/raw/arxiv_sentences.csv
    # offline alternative: --local_file my.jsonl --text_field text
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import spacy

URL_RE = re.compile(r"(https?://|www\.|\.com\b|\.org\b|@\w|\S+@\S+)", re.I)
MATH_RE = re.compile(r"[$\\{}^_]")
ARXIV_AREAS = {
    "cs": "cs", "math": "math", "stat": "stat", "q-bio": "q-bio", "q-fin": "q-fin",
    "econ": "econ", "eess": "eess", "astro-ph": "physics", "cond-mat": "physics",
    "gr-qc": "physics", "hep-ex": "physics", "hep-lat": "physics", "hep-ph": "physics",
    "hep-th": "physics", "math-ph": "physics", "nlin": "physics", "nucl-ex": "physics",
    "nucl-th": "physics", "physics": "physics", "quant-ph": "physics",
}


def area_of(categories: str) -> str:
    first = (categories or "").split()[0] if categories else ""
    return ARXIV_AREAS.get(first.split(".")[0], "other")


def clean(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def good_sentence(s: str, nlp_doc) -> bool:
    toks = s.split()
    if not 8 <= len(toks) <= 40:
        return False
    if URL_RE.search(s) or MATH_RE.search(s):
        return False
    alpha = sum(any(c.isalpha() for c in t) for t in toks)
    if alpha / len(toks) < 0.8:
        return False
    if s.isupper() or not s[0].isalnum():
        return False
    if s.count('"') % 2 == 1:  # unbalanced quotes usually = broken splitting
        return False
    if any(ent.label_ == "PERSON" for ent in nlp_doc.ents):
        return False
    return True


def stream_docs(source: str, seed: int, local_file: str | None, text_field: str):
    """Yield (doc_id, text, meta) tuples."""
    if local_file:
        path = Path(local_file)
        if path.suffix == ".csv":
            with open(path, encoding="utf-8") as f:
                for i, r in enumerate(csv.DictReader(f)):
                    yield str(i), r[text_field], {}
        else:
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    r = json.loads(line)
                    yield str(r.get("id", i)), r[text_field], {"categories": r.get("categories", "")}
        return

    from datasets import load_dataset  # imported lazily so --local_file works without it

    if source == "reviews":
        ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=20_000)
        for i, r in enumerate(ds):
            yield f"amz{i}", r["content"], {"polarity": "pos" if r["label"] == 1 else "neg"}
    elif source == "arxiv":
        ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=50_000)
        for r in ds:
            yield r["id"], r["abstract"], {"categories": r.get("categories", "")}
    else:
        raise ValueError(source)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["reviews", "arxiv"], required=True)
    ap.add_argument("--n", type=int, default=120, help="number of sentences to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=8240)
    ap.add_argument("--max_docs", type=int, default=200_000)
    ap.add_argument("--local_file", default=None)
    ap.add_argument("--text_field", default="text")
    a = ap.parse_args()

    nlp = spacy.load("en_core_web_sm")
    rng = random.Random(a.seed)
    prefix = "rev" if a.source == "reviews" else "arx"
    # arXiv: equal quota per area for the five largest areas
    quotas = None
    if a.source == "arxiv":
        areas = ["physics", "math", "cs", "stat", "q-bio"]
        quotas = {ar: a.n // len(areas) + (1 if i < a.n % len(areas) else 0) for i, ar in enumerate(areas)}

    kept, seen, per_area = [], set(), Counter()
    for n_docs, (doc_id, text, meta) in enumerate(stream_docs(a.source, a.seed, a.local_file, a.text_field)):
        if n_docs >= a.max_docs or len(kept) >= a.n:
            break
        area = area_of(meta.get("categories", "")) if a.source == "arxiv" else None
        if quotas is not None and (area not in quotas or per_area[area] >= quotas[area]):
            continue
        doc = nlp(clean(text))
        cands = [clean(s.text) for s in doc.sents]
        rng.shuffle(cands)
        for s in cands:
            if s in seen:
                continue
            if good_sentence(s, nlp(s)):
                seen.add(s)
                kept.append({
                    "sent_id": f"{prefix}{len(kept):04d}",
                    "domain": a.source,
                    "source_doc_id": doc_id,
                    "subdomain": area or meta.get("polarity", ""),
                    "sentence": s,
                })
                if area:
                    per_area[area] += 1
                break  # at most one sentence per document

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sent_id", "domain", "source_doc_id", "subdomain", "sentence"])
        w.writeheader()
        w.writerows(kept)
    print(f"kept {len(kept)} sentences from {n_docs + 1} documents -> {a.out}")
    if per_area:
        print("per area:", dict(per_area))


if __name__ == "__main__":
    main()
