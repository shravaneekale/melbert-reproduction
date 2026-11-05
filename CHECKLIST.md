# Plan to 9 November

Back up as you go: push to GitHub after every step, and the Colab notebook copies to Google Drive.

## Day 1 (Thu 5 Nov): start the long jobs
- [ ] Create a GitHub repo `melbert-reproduction`, push this folder, and keep it **public**.
- [ ] Open `notebooks/colab_replication.ipynb` in Colab (GPU) and run cells 1–5. Training runs in the background for a few hours.
- [ ] On your Mac:
  - [ ] Set up the environment (README §0).
  - [ ] Sample the data (README §2.1–2.2): `--n 75` per domain gives about 1,100–1,300 targets.
  - [ ] Read `annotation/guidelines.md` and start annotating: `python src/annotate.py --annotator A1`.
- [ ] Ask your second annotator. Send them `guidelines.md` plus `sheet_A2.xlsx` from `python src/annotation_sheet.py export --annotator A2 --double_only`.

## Day 2 (Fri 6 Nov): finish the labels
- [ ] Finish A1 labels. Import A2 labels.
- [ ] `python src/agreement.py ...`, then adjudicate together, then `python src/make_gold.py ...`
- [ ] Push `data/` to GitHub.
- [ ] In Colab, `git pull`, then run cells 6–7 to get predictions on the new data.
- [ ] Run the LLM judge, both roles (README §3).

## Day 3 (Sat 7 Nov): results and writing
- [ ] Run `python src/evaluate.py --tag ours --tag2 official --llm_labels data/annotation/labels_LLM.csv`
- [ ] Read `results/errors_ours_reviews.csv` and `results/errors_ours_arxiv.csv` and pick real examples for the report.
- [ ] Replace every `\TBD{...}` in `report/main.tex` (`grep -n TBD report/main.tex`).
- [ ] Replace `USERNAME` in `\repourl`.

## Day 4 (Sun 8 Nov): polish and buffer
- [ ] Compile on Overleaf (ACL template). The main body must be at most 10 pages.
- [ ] Check the repo is **public**. Open it in a private browser window to confirm.
- [ ] Submit the PDF.
