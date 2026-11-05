"""Run a trained MelBERT checkpoint on any MelBERT-format TSV and write
per-token predictions (with metaphor probabilities) to CSV.

The official main.py only logs aggregate metrics; for error analysis, per-POS
breakdowns, bootstrap confidence intervals and the LLM-as-judge step we need
every individual prediction, so this script re-uses the official model and
feature-conversion code but keeps the outputs.

Usage:
    python src/predict.py \
        --checkpoint external/MelBERT/saves/roberta-base/0_YYYYMMDD-HHMM \
        --data_dir data/VUA18 \
        --out results/preds_vua18_test.csv

The data_dir must contain test.tsv in MelBERT's format:
    index <tab> label <tab> sentence <tab> POS <tab> w_index
"""
import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MELBERT_DIR = ROOT / "external" / "MelBERT"
sys.path.insert(0, str(MELBERT_DIR))

from transformers import AutoTokenizer  # noqa: E402

from data_loader import load_test_data  # noqa: E402  (official code)
from main import load_pretrained_model  # noqa: E402  (official code)
from run_classifier_dataset_utils import processors  # noqa: E402  (official code)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("predict")


def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_weights(checkpoint: Path) -> Path:
    """Accept either a saves/ directory or a direct path to a .bin file."""
    if checkpoint.is_file():
        return checkpoint
    for name in ("pytorch_model.bin", "model.bin"):
        if (checkpoint / name).exists():
            return checkpoint / name
    bins = list(checkpoint.rglob("*.bin"))
    bins = [b for b in bins if "training_args" not in b.name]
    if len(bins) == 1:
        return bins[0]
    raise FileNotFoundError(f"No unique model .bin found under {checkpoint}: {bins}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="saves/... dir or pytorch_model.bin")
    ap.add_argument("--data_dir", required=True, help="dir containing test.tsv")
    ap.add_argument("--out", required=True, help="output CSV of per-token predictions")
    ap.add_argument("--bert_model", default="roberta-base")
    ap.add_argument("--max_seq_length", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--classifier_hidden", type=int, default=768)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    device = pick_device(a.cpu)
    args = SimpleNamespace(
        bert_model=a.bert_model,
        model_type="MELBERT",
        num_labels=2,
        classifier_hidden=a.classifier_hidden,
        drop_ratio=0.2,
        use_pos=True,
        use_local_context=True,
        max_seq_length=a.max_seq_length,
        eval_batch_size=a.batch_size,
        data_dir=a.data_dir,
        device=device,
        n_gpu=1 if device.type == "cuda" else 0,
        no_cuda=device.type != "cuda",
    )
    log.info(f"device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(a.bert_model)
    model = load_pretrained_model(args)
    state = torch.load(find_weights(Path(a.checkpoint)), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # position_ids buffers differ between transformers versions; anything else is a real problem
    real_missing = [k for k in missing if not k.endswith("position_ids")]
    if real_missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={real_missing[:10]} unexpected={unexpected[:10]}")
    model.eval()

    processor = processors["vua"]()
    guids, loader = load_test_data(
        args, log, processor, "vua", processor.get_labels(), tokenizer, "classification"
    )

    # Re-read the TSV so we can attach sentence / word / POS to each prediction.
    rows = {}
    with open(os.path.join(a.data_dir, "test.tsv"), encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        next(reader)
        for r in reader:
            key = f"test-{r[0]} {r[-1]}"
            rows[key] = r

    probs, golds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) for t in batch)
            ids, mask, seg, labels, _idx, ids2, mask2, seg2 = batch
            logp = model(
                ids,
                ids2,
                target_mask=(seg == 1),
                target_mask_2=seg2,
                attention_mask_2=mask2,
                token_type_ids=seg,
                attention_mask=mask,
            )
            probs.append(torch.exp(logp)[:, 1].float().cpu().numpy())
            golds.append(labels.cpu().numpy())
    p_met = np.concatenate(probs)
    gold = np.concatenate(golds)
    pred = (p_met >= 0.5).astype(int)  # identical to argmax for 2 classes

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["guid", "index", "w_index", "word", "POS", "gold", "pred", "p_metaphor", "sentence"])
        for g, gl, pr, p in zip(guids, gold, pred, p_met):
            r = rows.get(g)
            index, sent, pos, widx = (r[0], r[2], r[3], r[-1]) if r else ("", "", "", "")
            word = sent.split()[int(widx)] if r else ""
            w.writerow([g, index, widx, word, pos, int(gl), int(pr), f"{p:.5f}", sent])

    tp = int(((pred == 1) & (gold == 1)).sum())
    fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    log.info(f"n={len(gold)}  acc={(pred == gold).mean():.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    log.info(f"wrote {a.out}")


if __name__ == "__main__":
    main()
