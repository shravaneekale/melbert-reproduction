"""Spreadsheet route for annotators who would rather not use the terminal tool
(e.g. a second annotator on another computer).

  export : writes an .xlsx with one row per target word, the target shown in
           [brackets], and a drop-down (M / L) in the label column.
  import : reads the filled-in sheet back into data/annotation/labels_<ID>.csv
           (same format as src/annotate.py).

Usage:
    python src/annotation_sheet.py export --annotator A2 --double_only
    # ... annotator fills in data/annotation/sheet_A2.xlsx ...
    python src/annotation_sheet.py import --annotator A2
"""
import argparse
import csv
import datetime as dt
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from annotate import load_items


def export(a):
    items = load_items(a.items, a.double_only, a.seed)
    wb = Workbook()
    ws = wb.active
    ws.title = "annotate"
    head = ["item_id", "sentence (target in [brackets])", "target", "POS", "label (M/L)", "low confidence? (x)", "note"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="305496")
    for it in items:
        toks = it["sentence"].split()
        wi = int(it["w_index"])
        toks[wi] = f"[{toks[wi]}]"
        ws.append([it["item_id"], " ".join(toks), it["word"], it["POS"], "", "", ""])
    dv = DataValidation(type="list", formula1='"M,L"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(items) + 1}")
    widths = [14, 90, 14, 7, 12, 18, 30]
    for col, wdt in zip("ABCDEFG", widths):
        ws.column_dimensions[col].width = wdt
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"

    g = wb.create_sheet("guidelines (short)")
    for line in [
        "Read annotation/guidelines.md first. Short version (MIPVU, Steen et al. 2010):",
        "1. Read the whole sentence and work out what the [target] word means HERE.",
        "2. Look the word up (Macmillan / Longman online). Is there a more BASIC meaning?",
        "   Basic = more concrete, bodily, physical, precise, or historically older.",
        "3. If the contextual meaning is different from the basic one BUT can be understood",
        "   by comparison with it -> M (metaphorical). Otherwise -> L (literal).",
        "Conventional metaphors count as M (e.g. 'prices CLIMBED', 'a HUGE improvement' is L if size is literal).",
        "Do not look at anyone else's labels. Mark 'x' in column F if you are unsure.",
    ]:
        g.append([line])
    g.column_dimensions["A"].width = 110
    out = Path(f"data/annotation/sheet_{a.annotator}.xlsx")
    wb.save(out)
    print(f"wrote {out} ({len(items)} items)")


def import_(a):
    src = Path(f"data/annotation/sheet_{a.annotator}.xlsx")
    ws = load_workbook(src, data_only=True)["annotate"]
    rows, missing = [], 0
    for r in ws.iter_rows(min_row=2, values_only=True):
        item_id, _, _, _, lab, low, note = (list(r) + [None] * 7)[:7]
        if not item_id:
            continue
        lab = (str(lab).strip().upper() if lab is not None else "")
        if lab not in ("M", "L"):
            missing += 1
            continue
        rows.append({
            "item_id": item_id,
            "label": 1 if lab == "M" else 0,
            "low_conf": int(bool(low and str(low).strip())),
            "note": note or "",
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        })
    out = Path(f"data/annotation/labels_{a.annotator}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "label", "low_conf", "note", "time"])
        w.writeheader()
        w.writerows(rows)
    print(f"imported {len(rows)} labels -> {out}" + (f"  ({missing} rows unlabelled!)" if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["export", "import"])
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--items", default="data/annotation/items.csv")
    ap.add_argument("--double_only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    export(a) if a.action == "export" else import_(a)


if __name__ == "__main__":
    main()
