# Data

## New evaluation data (ours)

| Path | Contents |
|---|---|
| `raw/{reviews,arxiv}_sentences.csv` | sampled sentences with provenance (`source_doc_id`, arXiv subject area / review polarity) |
| `annotation/items.csv` | one row per target word (content words), `double=1` marks the agreement subset |
| `annotation/labels_A1.csv`, `labels_A2.csv` | each annotator's independent labels (1 = metaphorical) |
| `annotation/adjudication.csv` | A1/A2 disagreements with the adjudicated label and reason |
| `annotation/labels_LLM.csv` | LLM annotator labels (never used as gold) |
| `NEW/gold.csv`, `NEW/gold_labels.csv` | final gold standard |
| `NEW/{reviews,arxiv,all}/test.tsv` | MelBERT input format: `index, label, sentence, POS, w_index` |

**Datasheet summary**

* **Motivation:** test whether MelBERT, which is trained on VUA (BNC academic, news, fiction and conversation), generalises to informal user reviews and technical scientific prose.
* **Sources:**
  * `fancyzhx/amazon_polarity` (Zhang et al. 2015; reviews from McAuley & Leskovec 2013). Listed as Apache-2.0 on Hugging Face; original Amazon terms may restrict reuse.
  * `gfissore/arxiv-abstracts-2021` (CC0).
* **Collection:** seeded streaming sample, one sentence per document, 8–40 tokens. Sentences with URLs, e-mails, @handles or LaTeX maths were dropped, and so were sentences containing a PERSON entity (privacy).
* **Labels:** MIPVU (Steen et al. 2010), binary, word level, applied to content words (verbs, nouns, adjectives and adverbs).
  * A1 labelled every item; A2 independently labelled 30% of sentences.
  * Agreement is reported as Cohen's κ. Disagreements were adjudicated.
* **Intended use:** research evaluation of metaphor detection. Not for any decision about individuals.
* **Licence of our labels:** CC BY 4.0.

## Benchmarks (not redistributed)

Download the authors' pre-processed VUA-18 / VUA-20 / VUA-Verb / MOH-X / TroFi from the link in the official MelBERT README (the Colab notebook does this automatically), and place them as `data/VUA18/`, `data/VUAverb/`, `data/MOH-X/`, `data/TroFi/`. Then run `python src/prepare_transfer_sets.py`.
