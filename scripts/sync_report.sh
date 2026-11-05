#!/usr/bin/env bash
# Build the report PDF locally (tables/figures are read from ../results).
# For Overleaf: upload report/ AND results/ (keeping the folder layout), or
# zip the whole repo and use "New Project > Upload Project".
set -euo pipefail
cd "$(dirname "$0")/../report"
latexmk -pdf -interaction=nonstopmode main.tex
grep -c '\\TBD{' main.tex | xargs -I{} echo "{} \\TBD markers left in main.tex (replace them all before submitting)"
