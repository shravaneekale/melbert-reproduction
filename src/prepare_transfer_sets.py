"""Merge the 10 cross-validation test folds of MOH-X and TroFi into single
test-only files, so a VUA-trained MelBERT can be scored on the *whole* set
(zero-shot transfer, the setting of Table 3 in Choi et al. 2021).

The 10 test folds partition each dataset, so their union is the full dataset.

Usage:
    python src/prepare_transfer_sets.py --data_root data
Produces data/MOH-X_all/test.tsv and data/TroFi_all/test.tsv
"""
import argparse
from pathlib import Path


def merge(src_dir: Path, dst_dir: Path) -> None:
    folds = sorted(src_dir.glob("test[0-9].tsv"))
    if not folds:
        print(f"[skip] no test folds in {src_dir}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    seen, out, header = set(), [], None
    for fp in folds:
        lines = fp.read_text(encoding="utf-8").splitlines()
        header = header or lines[0]
        for line in lines[1:]:
            if line.strip() and line not in seen:
                seen.add(line)
                out.append(line)
    (dst_dir / "test.tsv").write_text(header + "\n" + "\n".join(out) + "\n", encoding="utf-8")
    print(f"{src_dir} -> {dst_dir/'test.tsv'}: {len(out)} rows from {len(folds)} folds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    a = ap.parse_args()
    root = Path(a.data_root)
    merge(root / "MOH-X" / "CLS", root / "MOH-X_all")
    merge(root / "TroFi" / "CLS", root / "TroFi_all")


if __name__ == "__main__":
    main()
