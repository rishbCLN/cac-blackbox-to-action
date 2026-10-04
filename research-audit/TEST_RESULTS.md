# Executed test results

Repository commit: `5c8e9f370941f721ef2f28f1c50c6388f724b6a1`. Audit date: 4 October 2026. Environment: Python 3.12.5, macOS arm64, repository-pinned requirements in a local `.venv`. Full environment and data hashes: [run-manifest.json](evidence/run-manifest.json), [pip-freeze.txt](evidence/pip-freeze.txt).

## Results

| Check | Outcome | Evidence |
|---|---|---|
| Pinned dependency installation | Completed; pip reports NumPy 2.4.0 is yanked for a backward-compatibility bug | [Installation log](evidence/dependency-install.log) |
| Dependency consistency | No broken requirements | [pip check](evidence/pip-check.log) |
| Fresh-clone pytest | Stopped after five fixture errors: processed Adult dataset missing | [Fresh test log](evidence/pytest-fresh-clone.log) |
| XGBoost import initially | Failed because macOS OpenMP runtime was not found; resolved for audit by pointing `DYLD_LIBRARY_PATH` at scikit-learn's installed `.dylibs` directory | Environment workaround below |
| Prepared-data full pytest | **318 passed, 1 skipped**, 112.35 seconds | [Full test and coverage log](evidence/pytest-prepared.log) |
| Statement coverage of `src` | **82% rounded**, 3,940 statements, 721 missed; not coverage of app/experiments | [Coverage JSON](evidence/coverage.json) |
| Four XGBoost models, no tuning | All trained; no synthetic data used; diabetes below configured AUC target | [Training log](evidence/train-xgb.log), [metrics](evidence/model-metrics.json) |
| Comparison smoke test | 4 datasets × 5 methods × 10 rejected applicants; k=5; **961 suggestions**, **0 recorded method exceptions**; strict check passed | [Run log](evidence/comparison-smoke.log), [method runs](evidence/smoke-results/method_runs.csv) |
| Actual CAC action replay | **200/200** suggestions match replay exactly and remain accepted | [Replay audit](evidence/actual-cac-replay.json) |
| Evaluator malformed-action probe | Fails to reject an empty action declaration; records zero-cost usable recourse despite unsuccessful action replay | [Probe output](evidence/probes.json) |
| Statistics duplication probe | Identical applicants copied across seeds produce spuriously stronger evidence | [Probe output](evidence/probes.json) |
| Metric definition probe | Paper's causal-success predicate and code's stronger fidelity gate disagree | [Probe output](evidence/probes.json) |
| Streamlit startup | AppTest home/startup completes with **0 exceptions** using smoke artifacts | [App log](evidence/app-startup.log) |
| Modular manuscript static check | No missing bibliography keys; 21 table inputs and 5 figure references unavailable at the checked stage | [Static check](evidence/paper-static-check.log) |

The fresh-clone fixture errors are setup dependencies, not evidence that the prepared tests fail. The static paper check was run after training and before generating a complete research artifact set. Most missing files require the omitted full experiments; the specific significance tables additionally have no producer in either advertised full pipeline command. No PDF compilation or visual proofread was attempted.

The README reports 301 passing tests and one skipped; this checkout actually produced 318 passing tests and one skipped in the prepared audit. A passing suite did not catch the independent validity probes.

## Trained classifier performance

| Dataset | Test rows | Test AUC | Repository target | Meets target |
|---|---:|---:|---:|---|
| Adult | 4,000 | 0.918186 | 0.90 | Yes |
| German | 200 | 0.784643 | 0.75 | Yes |
| HELOC | 1,975 | 0.788031 | 0.75 | Yes |
| Diabetes | 154 | 0.791111 | 0.80 | No |

These are no-tuning audit runs. They neither reproduce the manuscript's final models nor demonstrate that tuning cannot achieve the target. AUC targets are project settings, not publication acceptance criteria. The existing preprocessing concerns still apply to these audit runs.

## Observed usable rates in the small smoke comparison

These rates use the **current implementation's endpoint**, including its additional fidelity gate. Ten applicants per dataset is insufficient to establish general superiority. All methods see the same applicants within each dataset.

| Dataset | Random | Wachter | DiCE | Oracle | CAC |
|---|---:|---:|---:|---:|---:|
| Adult | 10/10 | 10/10 | 10/10 | 8/10 | 10/10 |
| German | 7/10 | 4/10 | 9/10 | 8/10 | 10/10 |
| HELOC | 7/10 | 0/10 | 6/10 | 8/10 | 10/10 |
| Diabetes | 5/10 | 6/10 | 10/10 | 10/10 | 10/10 |

Proposed-state validity is 10/10 in every cell except HELOC Random and Wachter, which each reach 8/10. See [headline.csv](evidence/smoke-results/headline.csv) for costs and query counts. Query counts differ substantially between methods, so this is not a matched-budget experiment.

Removing only the additional fidelity gate changes, for example, German Random to 10/10, Wachter to 7/10, DiCE to 10/10, and Oracle to 10/10. Full alternative summaries are in [metric-definition-comparison.csv](evidence/metric-definition-comparison.csv). This comparison retains existing inferred-action repair; it is a definition-sensitivity check, not a complete corrected benchmark.

## Reproduction commands

Run from the cloned repository root. The three audit Python scripts live beside the repository in `../research-audit/`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Audit-only macOS setup, using the installed OpenMP library.
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
export OMP_NUM_THREADS=2
export MPLCONFIGDIR=/private/tmp/cac-mpl
export PYTHONPATH="$PWD"

# Copies bundled Adult/HELOC/diabetes to the loader paths; downloads original UCI German.
.venv/bin/python ../research-audit/prepare_data.py
.venv/bin/python experiments/01_train_blackbox.py -m xgb --no-tune

.venv/bin/python -m pytest -o addopts='' -q --cov=src \
  --cov-report=term-missing \
  --cov-report=json:../research-audit/evidence/coverage.json

# Keep smoke artifacts separate from canonical publication artifacts.
CAC_RESULTS_DIR="$PWD/../research-audit/evidence/smoke-results" \
CAC_TABLES_DIR="$PWD/../research-audit/evidence/smoke-tables" \
  .venv/bin/python experiments/02_compare_methods.py -n 10 -k 5 --strict

.venv/bin/python ../research-audit/probe_findings.py
.venv/bin/python ../research-audit/audit_saved_results.py
.venv/bin/python scratch/check_paper.py
.venv/bin/python -m pip check
```

Data staging intentionally used the bundled diabetes copy to test the repository's existing cleanup procedure, not to endorse that source for the revised study. The bundled German CSV uses unsupported German field names/numeric categories, so the audit downloaded UCI's symbolic original and renamed columns to the loader's supported `Attribute1…Attribute20` plus `class` schema. The downloaded zip and staged-data checksums are retained.

## Limits

Not executed: full 100-query comparison, five-seed significance study, complete ablation/robustness sweeps, MLP training/comparison, every interactive app flow, human-subject evaluation, or LaTeX PDF compilation. The existing significance implementation was tested with controlled inputs; its published p-values were not independently reproduced. The scientific shortcomings in the assessment distinguish these reproduced defects from code-inspection concerns and missing experiments.

The repository's tracked files remained unchanged (`git status --short` was empty after the audit). Generated models/data/logs live in ignored project directories; independent evidence and deliverables live in `../research-audit/`. Nothing was committed or pushed.
