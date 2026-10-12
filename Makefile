# Causal Actionable Counterfactuals --- reproducible pipeline.
#
#   make all      data -> models -> SCMs -> cost models -> all methods ->
#                 ablation -> robustness -> qualitative -> figures ->
#                 de-circularized + known-SCM audits -> tables
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

.PHONY: all paper test app clean clean-all help prepare smoke significance research decircular known_scm
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

# 4b. Significance: repeat the comparison across seeds and test the gap.
#     paper/sections/09_experiments.tex inputs tables/significance.tex and
#     tables/significance_tests.tex, so the advertised paper build must produce
#     them. Depends on the same models/costs the comparison does.
$(STAMP)/significance: $(STAMP)/compare
	$(PY) experiments/06_significance.py
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

# 10. De-circularized causal audit (07): re-audit the §9 plans under an FCM fitted
#     on a disjoint fold, so CAC/MINT causal validity is tested out-of-sample.
#     Reads the saved results/counterfactuals_*.joblib from the comparison.
$(STAMP)/decircular: $(STAMP)/compare
	$(PY) experiments/07_decircular_audit.py
	@touch $@

# 11. Known-ground-truth causal audit (08): two synthetic worlds whose SCM is known
#     in closed form, so declared actions are scored through the TRUE equations
#     rather than the FCM the method optimised against. Trains its own synth black
#     boxes (additive), so it only needs models/metrics.json to exist first.
$(STAMP)/known_scm: $(STAMP)/models
	$(PY) experiments/08_known_scm.py
	@touch $@

all: $(STAMP)/figures $(STAMP)/user_study $(STAMP)/significance $(STAMP)/decircular $(STAMP)/known_scm
	@echo "Pipeline complete. Tables in paper/tables/, figures in paper/figures/."

# Named entry points the audit's gap-closure plan asks for, so a reader can run
# the inputs, a fast check, the full study, or the significance study on its own.
#
#   make prepare       data -> models -> SCMs -> cost models (everything the
#                      experiments read, nothing downstream of them)
#   make smoke         a fast 10-record comparison into a *separate* results dir,
#                      so a quick check never overwrites publication artifacts
#   make significance  just the seed-repeated significance study + its tables
#   make research      the full deterministic study (alias for `all`)
.PHONY: prepare smoke significance research decircular known_scm
prepare: $(STAMP)/cost
	@echo "Inputs ready: models/, data/processed/, SCMs, cost models."

smoke:
	CAC_RESULTS_DIR=$(CURDIR)/results/smoke \
	CAC_TABLES_DIR=$(CURDIR)/paper/tables/smoke \
		$(PY) experiments/02_compare_methods.py -n 10 -k 5
	@echo "Smoke comparison in results/smoke/ (separate from publication artifacts)."

significance: $(STAMP)/significance
	@echo "Significance tables in paper/tables/ (significance.tex, significance_tests.tex)."

decircular: $(STAMP)/decircular
	@echo "De-circularized audit in paper/tables/ (decircular_audit.tex)."

known_scm: $(STAMP)/known_scm
	@echo "Known-ground-truth audit in paper/tables/ (known_scm.tex)."

research: all

# --------------------------------------------------------------------------- #
# Paper. `paper/main.tex` is the canonical, generated manuscript: it \input's
# the tables and figures the pipeline writes, so it depends on the full artifact
# set and cannot be built with stale or missing inputs. (The repo-root main.tex
# is a frozen self-contained variant with hand-inlined numbers and is NOT built
# here; see the note at its head.) Needs a TeX Live install (pdflatex + bibtex);
# two bibtex/pdflatex passes resolve the \cite and \ref cross-references.
# --------------------------------------------------------------------------- #
paper: $(STAMP)/figures $(STAMP)/user_study $(STAMP)/significance
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
	rm -rf results/smoke paper/tables/smoke
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
	@echo "Targets: all | research | prepare | smoke | significance | decircular | known_scm | paper | test | app | clean | clean-all"
	@echo "  make all          - run the full deterministic pipeline (data..figures..significance)"
	@echo "  make research     - alias for 'all' (the full study)"
	@echo "  make prepare      - data -> models -> SCMs -> cost models (experiment inputs only)"
	@echo "  make smoke        - fast 10-record comparison into results/smoke (never touches publication artifacts)"
	@echo "  make significance - seed-repeated significance study + its two paper tables"
	@echo "  make decircular   - de-circularized causal audit (07): re-audit plans under an independent FCM"
	@echo "  make known_scm    - known-ground-truth causal audit (08): synthetic worlds, true-SCM auditor"
	@echo "  make paper        - build the canonical paper/main.pdf (needs TeX Live; depends on the full artifact set)"
	@echo "  make test         - run the test suite"
	@echo "  make app          - launch the Streamlit demo over stored results"
	@echo "  make clean        - remove generated artifacts (keeps raw data + models)"
