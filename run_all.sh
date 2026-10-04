#!/usr/bin/env bash
# Causal Actionable Counterfactuals --- run the whole pipeline, start to finish.
#
# A linear alternative to `make all` for anyone without Make. It runs every
# stage unconditionally, in dependency order, and stops at the first failure.
# Everything is local and deterministic (random_state=42); no paid API and no
# network call at evaluation time. Run from the project root:
#
#     bash run_all.sh              # full pipeline
#     bash run_all.sh --paper      # also build paper/main.pdf (needs TeX Live)
#
set -euo pipefail

# Resolve the project root as this script's directory, so it works from anywhere
# (guards against the parent-dir shadowing noted in the README's Windows notes).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY="${PY:-python}"

BUILD_PAPER=0
[[ "${1:-}" == "--paper" ]] && BUILD_PAPER=1

step() { echo; echo "===== $* ====="; }

step "1/10  Train black boxes (downloads + preprocesses data on first run)"
"$PY" experiments/01_train_blackbox.py

step "2/10  Validate structural causal models (writes scm_* tables + figures)"
"$PY" -m src.scm.validate_scm

step "3/10  Build cost / actionability models (writes cost_model_* tables)"
"$PY" -m src.cost.cost_functions

step "4/10  Compare every method on every dataset (main results)"
"$PY" experiments/02_compare_methods.py

step "5/10  Significance (seed-repeated comparison + paired tests + tables)"
"$PY" experiments/06_significance.py

step "6/10  Objective ablation"
"$PY" experiments/03_ablation.py

step "7/10  Robustness (budget, k, model transfer, graph misspecification)"
"$PY" experiments/04_robustness.py

step "8/10  Qualitative cases (the per-applicant advice)"
"$PY" experiments/05_qualitative.py

step "9/10  Figures (result figures + recourse plans)"
"$PY" -m src.viz.plot_results
"$PY" -m src.viz.plot_counterfactuals

step "10/10  User-study analysis (synthetic responses)"
"$PY" user_study/analysis.py

if [[ "$BUILD_PAPER" == "1" ]]; then
  # Canonical manuscript: paper/main.tex, which \input's the tables and figures
  # produced above. (The repo-root main.tex is a frozen self-contained variant
  # and is intentionally not built here.)
  step "paper  Build paper/main.pdf (needs pdflatex + bibtex)"
  ( cd paper \
    && pdflatex -interaction=nonstopmode main.tex \
    && bibtex main \
    && pdflatex -interaction=nonstopmode main.tex \
    && pdflatex -interaction=nonstopmode main.tex )
  echo "Built paper/main.pdf"
fi

echo
echo "Done. Tables in paper/tables/, figures in paper/figures/, artifacts in results/."
echo "Explore interactively with:  python -m streamlit run app/app.py"
