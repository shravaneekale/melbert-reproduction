# Reproducing MelBERT for Metaphor Detection and Testing It on New Domains

COMP8240 (Applications of Data Science), Macquarie University, 2026. Shravanee Sanjeev Kale.

This repository reproduces **MelBERT** (Choi et al., NAACL 2021; official code: <https://github.com/jin530/MelBERT>) and evaluates it on **new, human-annotated data** from two domains the original benchmarks don't cover: Amazon product reviews and arXiv abstracts. The final report is [`report/MelBERT_Final_Report.pdf`](report/MelBERT_Final_Report.pdf).

## Results at a glance

**Replication (F1 on the metaphor class)**

| Test set | Paper | Authors' checkpoint | Ours (retrained, seed 42) |
|---|---|---|---|
| VUA-18 | 78.5 | 78.8 | **78.2** |
| VUA-Verb | 75.7 | 75.6 | **75.3** |
| MOH-X (zero-shot) | 79.2 | 79.1 | **80.8** |
| TroFi (zero-shot) | 62.0 | 61.0 | **62.3** |

**New data (388 hand-annotated content-word targets)**

| Test set | P | R | F1 |
|---|---|---|---|
| VUA-18 test, same POS (verbs, nouns, adjectives, adverbs) | 74.5 | 69.7 | 72.1 |
| Amazon reviews | 26.9 | 58.3 | 36.8 |
| arXiv abstracts | 21.8 | 57.1 | 31.6 |

Agreement between raters (Cohen's κ): the two human annotators reach 0.44, while each human against LLM1 (Claude) reaches only 0.15 and 0.07. Full analysis is in the report.

## Repository layout

```
setup/        clone + 3-line compatibility patch of the official MelBERT code
scripts/      get_data.sh, run_replication.sh, run_replication_mac.sh, run_predictions.sh
notebooks/    Google Colab notebook (GPU alternative)
src/          our code:
  sample_corpus.py        sample real sentences (reviews / arXiv) with privacy filters
  build_targets.py        word-level targets in MelBERT format (content words, UPOS)
  annotate.py             terminal annotation tool (blind, resumable)
  annotation_sheet.py     spreadsheet alternative for annotators
  agreement.py            Cohen's kappa + adjudication sheet
  make_gold.py            gold labels + MelBERT test files
  predict.py              per-token predictions from any checkpoint
  evaluate.py             metrics, bootstrap CIs, LaTeX tables, figure, error lists
  llm_judge.py            LLM prompts (MIPVU) for annotation / judging
  llm_reference_eval.py   LLM-as-judge: score MelBERT with LLM labels as reference
annotation/guidelines.md  MIPVU guidelines given to every rater
data/raw/                 150 sampled sentences with provenance
data/annotation/          items.csv (1,436 targets), items_subset.csv (388 hand-annotated)
                          labels_A1.csv (author), labels_A2.csv (second human),
                          labels_LLM.csv (LLM1: Claude), labels_LLM2.csv (LLM2: ChatGPT),
                          labels_A1_pilot.csv (discarded pilot), adjudication.csv
data/NEW/                 gold.csv + MelBERT-format test files (reviews / arxiv / all)
results/                  predictions, metrics.json, tables, figure, error lists, logs
report/                   LaTeX source (ACL template) + final PDF
```

## How to reproduce

### 0. Environment (macOS or Linux)
```bash
conda create -n melbert python=3.11 -y && conda activate melbert
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash setup/setup_melbert.sh      # clones official MelBERT @11b0f78 into external/ and patches it
bash scripts/get_data.sh         # authors' pre-processed datasets + released checkpoint
```
The patch changes three lines and nothing about the model or training:
1. `AdamW` is imported from `torch.optim`.
2. The device is chosen as CUDA, then Apple MPS, then CPU.
3. `torch.load(..., map_location=device)` is used when loading checkpoints.

### 1. Replication
```bash
bash scripts/run_replication.sh 42      # paper settings; ~5h45m on an Apple-Silicon Mac (MPS), less on a GPU
bash scripts/run_predictions.sh external/MelBERT/saves/roberta-base/<run_folder> ours
bash scripts/run_predictions.sh checkpoints/official/melbert_ckpt official
```
The GPU alternative is `notebooks/colab_replication.ipynb`.

### 2. New data
```bash
python src/sample_corpus.py --source reviews --n 75 --out data/raw/reviews_sentences.csv
python src/sample_corpus.py --source arxiv   --n 75 --out data/raw/arxiv_sentences.csv
python src/build_targets.py --inputs data/raw/reviews_sentences.csv data/raw/arxiv_sentences.csv \
       --out data/annotation/items.csv
```
The hand-annotated subset (`items_subset.csv`) is 20 random sentences per domain, seed 8240. To annotate and build the gold labels:
```bash
python src/annotate.py --annotator A1 --items data/annotation/items_subset.csv
python src/annotate.py --annotator A2 --items data/annotation/items_subset.csv --seed 2
python src/agreement.py --a data/annotation/labels_A1.csv --b data/annotation/labels_LLM.csv \
       --items data/annotation/items_subset.csv --name a1_vs_llm --disagreements data/annotation/adjudication.csv
python src/agreement.py --a data/annotation/labels_A1.csv --b data/annotation/labels_A2.csv \
       --items data/annotation/items_subset.csv --name human
python src/make_gold.py --items data/annotation/items_subset.csv --adjudication data/annotation/adjudication.csv
```
LLM labels were produced with the MIPVU prompt in `src/llm_judge.py`:
- **LLM1:** Claude Opus 5.5 via claude.ai, 5 Nov 2026.
- **LLM2:** ChatGPT via chatgpt.com, 5 Nov 2026.

`python src/llm_judge.py annotate --backend anthropic|openai|manual` reruns this step.

### 3. Evaluation
```bash
for d in reviews arxiv all; do
  python src/predict.py --checkpoint external/MelBERT/saves/roberta-base/<run_folder> \
         --data_dir data/NEW/$d --out results/preds/ours_$([ $d = all ] && echo new_all || echo $d).csv
done
python src/evaluate.py --tag ours --tag2 official
python src/llm_reference_eval.py --preds results/preds/ours_new_all.csv
```

### 4. Report
`report/main.tex` uses the ACL template. Run `cd report && latexmk -pdf main.tex`, or upload `report/` to Overleaf.

## Data and licences
- **VUA-18/20, VUA-Verb, MOH-X, TroFi:** downloaded from the authors' links; not redistributed here.
- **Amazon reviews:** `fancyzhx/amazon_polarity` (Apache-2.0 on Hugging Face; reviews originate from Amazon).
- **arXiv abstracts:** `gfissore/arxiv-abstracts-2021` (CC0).
- **What we release:** only the 150 sampled sentences plus labels, for research use.
- **Model weights:** not stored in git, because of size.
- **Licences:** our labels are CC BY 4.0; our code is MIT.
