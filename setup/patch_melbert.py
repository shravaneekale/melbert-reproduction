"""Apply the minimal compatibility patches needed to run the official MelBERT
code (github.com/jin530/MelBERT, commit 11b0f78) with current PyTorch /
transformers releases.

The patches do NOT change the model, the data processing or the training
procedure. They only:
  1. import AdamW from torch.optim (it was removed from transformers);
  2. select the best available device (CUDA -> Apple MPS -> CPU) instead of
     CUDA-or-CPU only, so the code can run on Apple Silicon;
  3. pass map_location to torch.load so checkpoints trained on a GPU can be
     loaded on a Mac / CPU (and vice versa).

The script is idempotent: running it twice is harmless.

Usage:
    python setup/patch_melbert.py external/MelBERT
"""
import sys
from pathlib import Path

PATCHES = {
    "main.py": [
        (
            "from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup",
            "from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup\n"
            "from torch.optim import AdamW  # [patch 1] AdamW was removed from transformers",
        ),
        (
            'device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")',
            "# [patch 2] pick CUDA, then Apple-Silicon MPS, then CPU\n"
            "    if torch.cuda.is_available() and not args.no_cuda:\n"
            '        device = torch.device("cuda")\n'
            "    elif getattr(torch.backends, \"mps\", None) is not None and torch.backends.mps.is_available() and not args.no_cuda:\n"
            '        device = torch.device("mps")\n'
            "    else:\n"
            '        device = torch.device("cpu")',
        ),
        (
            "model.module.load_state_dict(torch.load(output_model_file))",
            "model.module.load_state_dict(torch.load(output_model_file, map_location=args.device))  # [patch 3]",
        ),
        (
            "model.load_state_dict(torch.load(output_model_file))",
            "model.load_state_dict(torch.load(output_model_file, map_location=args.device))  # [patch 3]",
        ),
    ],
}


def main(repo_dir: str) -> None:
    root = Path(repo_dir)
    if not (root / "main.py").exists():
        sys.exit(f"Could not find main.py in {root}. Pass the path of the cloned MelBERT repo.")
    for fname, edits in PATCHES.items():
        path = root / fname
        text = path.read_text()
        for old, new in edits:
            if new in text:
                print(f"[skip] {fname}: already patched -> {new.splitlines()[0][:60]}")
            elif old in text:
                text = text.replace(old, new)
                print(f"[ok]   {fname}: {old[:60]}")
            else:
                sys.exit(f"[fail] {fname}: expected line not found:\n  {old}")
        path.write_text(text)
    print("All patches applied.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "external/MelBERT")
