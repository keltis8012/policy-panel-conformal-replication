#!/usr/bin/env bash
# Reproduce every table and figure of the manuscript from the bundled snapshot.
set -euo pipefail
cd "$(dirname "$0")"

echo "[0/5] unpacking data"
mkdir -p 02_datasets/raw 02_datasets/processed
for f in data/*.csv.gz; do [ -f "02_datasets/raw/$(basename "$f" .gz)" ] || gunzip -c "$f" > "02_datasets/raw/$(basename "$f" .gz)"; done

echo "[1/5] main decision pipeline  -> Tables 4,5,8,9 ; Figures 2,3,6,7"
python code/e01b_empirical.py

echo "[2/5] trigger-threshold grid  -> Table 9 (sensitivity)"
python code/tau_sensitivity.py

echo "[3/5] placebo outcomes        -> Table 6"
python code/aux_run.py

echo "[4/5] forward effective level -> Table 4, §4.2"
python code/_recompute_aeff.py
python code/_placebo_forward.py

echo "[5/5] figures                 -> Figures 1-7"
python code/e01b_figures.py || echo "  (figure script expects a run dir; pass it as argv[1])"

echo "done."
