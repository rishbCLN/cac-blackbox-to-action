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

step "1/12  Train black boxes (downloads + preprocesses data on first run)"
"$PY" experiments/01_train_blackbox.py

step "2/12  Validate structural causal models (writes scm_* tables + figures)"
"$PY" -m src.scm.validate_scm

step "3/12  Build cost / actionability models (writes cost_model_* tables)"
"$PY" -m src.cost.cost_functions

step "4/12  Compare every method on every dataset (main results)"
"$PY" experiments/02_compare_methods.py

step "5/12  Significance (seed-repeated comparison + paired tests + tables)"
"$PY" experiments/06_significance.py

step "6/12  Objective ablation"
"$PY" experiments/03_ablation.py

step "7/12  Robustness (budget, k, model transfer, graph misspecification)"
"$PY" experiments/04_robustness.py

step "8/12  Qualitative cases (the per-applicant advice)"
"$PY" experiments/05_qualitative.py

step "9/12  Figures (result figures + recourse plans)"
"$PY" -m src.viz.plot_results
"$PY" -m src.viz.plot_counterfactuals

step "10/12  User-study analysis (synthetic responses)"
"$PY" user_study/analysis.py

step "11/12  De-circularized causal audit (re-audit plans under an independent FCM)"
"$PY" experiments/07_decircular_audit.py

step "12/12  Known-ground-truth causal audit (synthetic worlds, true-SCM auditor)"
"$PY" experiments/08_known_scm.py

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
