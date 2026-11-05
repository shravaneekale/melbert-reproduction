# Reproducing MelBERT for Metaphor Detection and Testing It on New Domains

COMP8240 (Applications of Data Science), Macquarie University, 2026. Shravanee Sanjeev Kale.

This repository reproduces **MelBERT** (Choi et al., NAACL 2021; official code: <https://github.com/jin530/MelBERT>) and evaluates it on **new, human-annotated data** from two domains the original benchmarks don't cover: Amazon product reviews and arXiv scientific abstracts.

```
setup/        clone + minimal compatibility patch of the official MelBERT code
scripts/      replication / prediction shell scripts
notebooks/    Google Colab notebook for the full GPU replication
src/          our code: data sampling, annotation, agreement, prediction, evaluation, LLM-as-judge
annotation/   MIPVU annotation guidelines
data/raw/         sampled source sentences (+ provenance)
data/annotation/  annotation items, each annotator's labels, adjudication sheet
data/NEW/         final gold standard + MelBERT-format test files
results/      predictions, metrics.json, LaTeX tables, figures
report/       LaTeX source of the final report (ACL template)
```

---

## 0. Environment (Mac or Linux)

```bash
conda create -n melbert python=3.11 -y && conda activate melbert   # or python3.11 -m venv .venv
pip install -r requirements.txt
python -m spacy download en_core_web_sm
bash setup/setup_melbert.sh          # clones official MelBERT @11b0f78 into external/ and patches it
```

The patch (`setup/patch_melbert.py`) changes three lines and nothing about the model or training:

1. `AdamW` is imported from `torch.optim`, because it was removed from `transformers`.
2. The device is chosen as CUDA, then Apple MPS, then CPU, so the code runs on Apple Silicon.
3. `torch.load(..., map_location=device)` is used so a checkpoint trained on a GPU loads on a Mac.

---

## 1. Replication (Goal 1)

### 1a. Full replication on a GPU (recommended): Google Colab

Push this repo to GitHub, open `notebooks/colab_replication.ipynb` in Colab with a GPU runtime, set `REPO_URL`, and run all cells. The notebook:

* downloads the authors' pre-processed datasets (VUA-18, VUA-20, VUA-Verb, MOH-X, TroFi);
* scores the **authors' released checkpoint** (a quick sanity check of code + data);
* **retrains MelBERT** from RoBERTa-base with the paper's settings (RoBERTa-base, 3 epochs, batch 32, lr 3e-5, 2 warm-up epochs, seed 42);
* writes per-token predictions for every test set and copies everything to Google Drive.

### 1b. Reduced replication on a Mac (no GPU)

```bash
# put the official data in data/VUA18 (train.tsv, test.tsv) first
bash scripts/run_replication_mac.sh 10000 4     # 10k training sentences, 4 epochs, full test set
```

### Predictions with any checkpoint

```bash
bash scripts/run_predictions.sh <checkpoint_dir> ours        # or: checkpoints/official official
```

---

## 2. New data (Goal 2)

```bash
# 2.1 sample real sentences (streams from Hugging Face; no full download)
python src/sample_corpus.py --source reviews --n 75 --out data/raw/reviews_sentences.csv
python src/sample_corpus.py --source arxiv   --n 75 --out data/raw/arxiv_sentences.csv

# 2.2 split into word-level targets (content words) and mark a 30% double-annotation subset
python src/build_targets.py --inputs data/raw/reviews_sentences.csv data/raw/arxiv_sentences.csv \
       --out data/annotation/items.csv --double_frac 0.3

# 2.3 annotate (read annotation/guidelines.md first)
python src/annotate.py --annotator A1                    # primary annotator: every item
python src/annotate.py --annotator A2 --double_only      # second annotator: agreement subset
#   ...or give A2 a spreadsheet instead:
python src/annotation_sheet.py export --annotator A2 --double_only   # -> data/annotation/sheet_A2.xlsx
python src/annotation_sheet.py import --annotator A2

# 2.4 agreement + adjudication
python src/agreement.py --a data/annotation/labels_A1.csv --b data/annotation/labels_A2.csv \
       --name human --disagreements data/annotation/adjudication.csv
#   fill in adjudicated_label (0/1) + reason for each disagreement, then:
python src/make_gold.py --adjudication data/annotation/adjudication.csv
```

`make_gold.py` writes `data/NEW/{reviews,arxiv,all}/test.tsv` in MelBERT's input format.

---

## 3. Evaluation on the new data

```bash
bash scripts/run_predictions.sh <checkpoint_dir> ours

# LLM-as-judge, role 1: an LLM annotates every target (blind); compare with human gold
python src/llm_judge.py annotate --backend anthropic --model claude-sonnet-5-5   # or --backend manual
python src/agreement.py --a data/NEW/gold_labels.csv --b data/annotation/labels_LLM.csv --name llm_vs_gold

# LLM-as-judge, role 2: the LLM judges each MelBERT prediction; compare verdicts with gold
python src/llm_judge.py judge --backend anthropic --model claude-sonnet-5-5 --preds results/preds/ours_new_all.csv

# all metrics, bootstrap CIs, LaTeX tables, figure
python src/evaluate.py --tag ours --tag2 official --llm_labels data/annotation/labels_LLM.csv
```

`--backend manual` needs no API key. It writes prompt batches you paste into a chat assistant, and parses the saved replies on the next run.

---

## 4. Report

`report/main.tex` uses the ACL template and `\input`s the tables generated in `results/tables/`. Upload the `report/` folder and `results/` to Overleaf, or run `cd report && latexmk -pdf main.tex`.

## Data and licences

* **VUA-18/20, MOH-X, TroFi**: downloaded from the authors' links; not redistributed here.
* **Amazon reviews** (`fancyzhx/amazon_polarity`, Zhang et al. 2015, from McAuley & Leskovec 2013; Apache-2.0 on Hugging Face) and **arXiv abstracts** (`gfissore/arxiv-abstracts-2021`, CC0). We redistribute only the ~150 sampled sentences plus our labels, for research use.
* Our labels and code: CC BY 4.0 / MIT.
