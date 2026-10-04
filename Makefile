# Causal Actionable Counterfactuals --- reproducible pipeline.
#
#   make all      data -> models -> SCMs -> cost models -> all methods ->
#                 ablation -> robustness -> qualitative -> figures -> tables
#   make paper    build paper/main.pdf from the tables and figures `make all` wrote
#   make test     run the test suite
#   make app      launch the Streamlit demo over the stored results
#   make clean    remove generated artifacts (keeps raw data and source)
#
# Everything runs locally with no paid API and no network call at evaluation
# time. The pipeline is deterministic (random_state=42), so re-running `make all`
# reproduces every number in the paper. Run targets from the project root.
#
# On Windows, invoke through Git Bash (the shell this was verified in) or WSL.

PY ?= python
PYTHONPATH := $(CURDIR)
export PYTHONPATH

# Marker files under this dir record that a stage completed, so Make can skip
# stages whose inputs have not changed. Delete .build/ (or `make clean`) to force.
STAMP := .build

.PHONY: all paper test app clean clean-all help
.DEFAULT_GOAL := help

# --------------------------------------------------------------------------- #
# The pipeline, stage by stage. Each stage depends on the previous stamp.
# --------------------------------------------------------------------------- #
$(STAMP):
	@mkdir -p $(STAMP)

# 1. Black boxes (also triggers data download + preprocessing via build_dataset).
$(STAMP)/models: | $(STAMP)
	$(PY) experiments/01_train_blackbox.py
	@touch $@

# 2. Structural causal models: validate graphs, write scm_* tables and figures.
$(STAMP)/scm: $(STAMP)/models
	$(PY) -m src.scm.validate_scm
	@touch $@

# 3. Cost / actionability models: write cost_model_* tables.
$(STAMP)/cost: $(STAMP)/scm
	$(PY) -m src.cost.cost_functions
	@touch $@

# 4. The main comparison: every method on every dataset (§9 headline).
$(STAMP)/compare: $(STAMP)/cost
	$(PY) experiments/02_compare_methods.py
	@touch $@

# 5. Objective ablation.
$(STAMP)/ablation: $(STAMP)/compare
	$(PY) experiments/03_ablation.py
	@touch $@

# 6. Robustness: budget, k, model transfer, graph misspecification.
$(STAMP)/robustness: $(STAMP)/compare
	$(PY) experiments/04_robustness.py
	@touch $@

# 7. Qualitative cases: the per-applicant advice the paper and app walk through.
$(STAMP)/qualitative: $(STAMP)/compare
	$(PY) experiments/05_qualitative.py
	@touch $@

# 8. Figures: result figures (from the CSVs) and recourse-plan figures.
$(STAMP)/figures: $(STAMP)/ablation $(STAMP)/robustness $(STAMP)/qualitative
	$(PY) -m src.viz.plot_results
	$(PY) -m src.viz.plot_counterfactuals
	@touch $@

# 9. User-study analysis (synthetic responses; writes user_study table + figure).
$(STAMP)/user_study: $(STAMP)/qualitative
	$(PY) user_study/analysis.py
	@touch $@

all: $(STAMP)/figures $(STAMP)/user_study
	@echo "Pipeline complete. Tables in paper/tables/, figures in paper/figures/."

# --------------------------------------------------------------------------- #
# Paper. Needs a TeX Live install (pdflatex + bibtex). Two bibtex/pdflatex
# passes resolve the \cite and \ref cross-references.
# --------------------------------------------------------------------------- #
paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode main.tex && \
		pdflatex -interaction=nonstopmode main.tex
	@echo "Built paper/main.pdf"

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app/app.py

# Remove generated artifacts but keep raw data (data/raw) and source.
clean:
	rm -rf $(STAMP)
	rm -f results/*.csv paper/tables/*.tex
	rm -f paper/figures/results/* paper/figures/eda/* paper/figures/recourse/* \
		paper/figures/user_study/* paper/figures/scm_*.png paper/figures/scm_*.pdf \
		paper/figures/scm_*.mmd
	rm -f paper/main.pdf paper/main.aux paper/main.bbl paper/main.blg \
		paper/main.log paper/main.out
	@echo "Cleaned generated artifacts (raw data and models kept)."

# Also drop trained models and processed data (full rebuild next time).
clean-all: clean
	rm -f models/*.joblib models/metrics.json
	rm -rf data/processed
	@echo "Cleaned models and processed data too."

help:
	@echo "Targets: all | paper | test | app | clean | clean-all"
	@echo "  make all    - run the full deterministic pipeline (data..figures)"
	@echo "  make paper  - build paper/main.pdf (needs TeX Live: pdflatex+bibtex)"
	@echo "  make test   - run the test suite"
	@echo "  make app    - launch the Streamlit demo over stored results"
	@echo "  make clean  - remove generated artifacts (keeps raw data + models)"
