"""Generate notebooks/01–04 as valid, unexecuted .ipynb files.

Kept in scratch/: the notebooks are the deliverable, this is the throwaway that
stamps them out so their JSON is always well-formed.  Every code cell is written
to run top-to-bottom against the pipeline's public API and to *read from*
``results/`` rather than recompute a 15-minute experiment, so the set opens and
runs in seconds.  No cell magics: visuals go through ``save_figure`` (rule #12)
and a display helper that shows the PNG in a notebook and no-ops in a plain
script, which is what lets ``scratch/smoke_notebooks.py`` execute them headless.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

# The preamble every notebook opens with: put the project root on the path (the
# notebook is launched from notebooks/, and a stray site-packages ``src`` must
# not win), and define one script-safe image display helper.
PREAMBLE = '''\
import sys, pathlib
_here = pathlib.Path.cwd()
ROOT = _here if (_here / "src").exists() else _here.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 120)

RESULTS = ROOT / "results"

def show_image(path):
    """Display a saved figure inline in a notebook; no-op-print in a script."""
    path = pathlib.Path(path)
    try:
        from IPython.display import Image, display  # noqa: PLC0415
        display(Image(filename=str(path)))
    except Exception:
        print("figure saved at", path)

def read_result(name):
    """Read a results CSV, or say which experiment writes it."""
    p = RESULTS / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run the experiment that writes it")
    return pd.read_csv(p)
'''


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def write(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = OUT / name
    nbf.write(nb, str(path))
    print("wrote", path.relative_to(ROOT))


# --------------------------------------------------------------- 01 · EDA

def eda() -> list:
    return [
        md("""
# 01 · Exploratory data analysis

**Black Box → Actionable Insight.** Four public datasets, one framing: each is a
consequential *reject* decision a person would want overturned, and the label is
the outcome a counterfactual would have to flip.

| dataset | decision | flip target |
|---|---|---|
| **Adult** (primary) | income ≤ 50K | earn > 50K |
| **German credit** | bad credit risk | good risk |
| **HELOC** | line declined | line granted |
| **Diabetes** | positive screen | no diabetes |

Nothing here touches a model — this is the raw material the black box, the SCM
and the cost model are all built on. Everything loads through the pipeline's own
`load_dataset`, so a notebook and an experiment see identical splits.
"""),
        code(PREAMBLE),
        code('''
from src.config import DATASETS
from src.data import load_dataset

bundles = {d: load_dataset(d) for d in DATASETS}
summary = pd.DataFrame([
    {"dataset": d,
     "rows": b.frame.shape[0],
     "features": b.X.shape[1],
     "train": len(b.train_idx),
     "test": len(b.test_idx),
     "positive rate": round(float((b.y == 1).mean()), 3),
     "flip target": b.spec.positive_meaning}
    for d, b in bundles.items()
]).set_index("dataset")
summary
'''),
        md("""
## The feature taxonomy is the whole game

Recourse only means anything once features are sorted into what a person **can**
change, what they **cannot**, and what is **protected**. The cost model and the
SCM both read this taxonomy; a plan that asks someone to change their age or race
is not cheap, it is *inadmissible*.
"""),
        code('''
b = bundles["adult"]
feat = pd.DataFrame([
    {"feature": f.name, "kind": f.kind, "unit": f.unit,
     "protected": f.protected, "manipulable": f.manipulable,
     "monotone": f.monotone}
    for f in b.spec.features
])
feat
'''),
        code('''
# Class balance across the four datasets, saved through the paper's style.
from src.utils.io import save_figure
from src.viz import figure, label_of, ACCENT, WARN

fig, axes = figure(1, len(DATASETS), figsize=(3.0 * len(DATASETS), 2.8),
                   squeeze=False)
for ax, d in zip(axes.ravel(), DATASETS):
    y = bundles[d].y
    ax.bar(["rejected\\n(flip from)", "accepted\\n(flip to)"],
           [int((y == 0).sum()), int((y == 1).sum())], color=[WARN, ACCENT])
    ax.set_title(label_of(d))
fig.suptitle("Class balance: the minority class is the one recourse serves", y=1.02)
paths = save_figure(fig, "eda_class_balance", subdir="eda")
show_image(paths[0])
'''),
        md("""
## Deviation — the diabetes dataset

Diabetes was added as a fourth dataset. Two things were done to it, both recorded
in the README's deviations:

1. **Label inverted** to match the others' "reject to overturn" framing:
   `no_diabetes = 1 - Outcome`, so the positive class is the good outcome a plan
   flips *to*.
2. **A class-conditional-imputation leak was found and removed** — zeros in
   `Glucose`, `BloodPressure`, etc. had been imputed with the class mean, which
   leaks the label into the features. They are treated as missing without peeking
   at the outcome.
"""),
        code('''
d = bundles["diabetes"]
print("diabetes target:", d.spec.target,
      "| positive =", d.spec.positive_meaning)
print("balance:", dict(pd.Series(d.y).value_counts().sort_index()))
d.frame.describe().T[["mean", "std", "min", "max"]].round(2)
'''),
    ]


# --------------------------------------------------------------- 02 · SCM

def scm() -> list:
    return [
        md("""
# 02 · Causal structure and its validation

The SCM is the audit that binds. A baseline can flip the box on paper by moving a
feature the world would move *back* the moment the applicant acts; only a plan
that agrees with the causal structure actually comes true. This notebook shows
the graphs and how far the data supports them.

The graphs are hand-built from domain knowledge (§6), then **checked** against the
data with conditional-independence tests — not fitted, because a graph fitted to
the same data the black box saw would launder the model's own correlations back
in as "causes".
"""),
        code(PREAMBLE),
        code('''
from src.config import DATASETS
from src.scm import get_scm

rows = []
for d in DATASETS:
    s = get_scm(d).summary()
    rows.append({"dataset": d, "nodes": s["n_nodes"], "edges": s["n_edges"],
                 "features": s["n_features"], "actionable": s["n_actionable"],
                 "acyclic": s["acyclic"]})
pd.DataFrame(rows).set_index("dataset")
'''),
        code('''
# The primary graph, drawn and saved through the SCM plotting layer.
from src.viz.plot_graphs import save_scm_figure

paths = save_scm_figure("adult")
show_image(paths[0])
'''),
        md("""
## How well does the data back each edge?

Each hypothesised edge implies a conditional (in)dependence the data can be asked
about. The summary below counts how many held. The counts are *evidence about the
graph*, not a score to maximise — a contradicted edge we keep on domain grounds is
a documented modelling choice, not a bug.
"""),
        code('''
try:
    display(read_result("scm_validation_summary"))
except FileNotFoundError as e:
    print(e)
'''),
        code('''
# A few individual CI tests, most confident first.
try:
    ci = read_result("scm_ci_tests")
    cols = [c for c in ["dataset", "x", "y", "given", "verdict", "p_value",
                        "statistic"] if c in ci.columns]
    display(ci[cols].head(12))
except FileNotFoundError as e:
    print(e)
'''),
        md("""
## Deviation — graph provenance and limits

- The **Adult** SCM is a 10-node / 25-edge superset of the 8-feature graph
  sketched in §6.1; the extra structure (and the latent node) is documented in the
  README.
- The **German** graph is the weakest: of 37 implied independences, roughly
  **18 are supported, 1 is weak, and 18 are contradicted** by CI tests. It is kept
  as written for domain reasons and flagged as a limitation — the robustness
  experiment (§04) re-runs the whole comparison under a perturbed graph to show the
  conclusions do not hinge on any single contested edge.
"""),
    ]


# --------------------------------------------------------------- 03 · methods

def methods() -> list:
    return [
        md("""
# 03 · The black box, the methods, and the three audits

**The black box is sacred.** It exposes only `predict` / `predict_proba`; no
method here reads a tree, a gradient or a feature importance. Every method sees the
same opaque function and the same feature taxonomy.

Four methods plus two references are compared:

- **CAC** (ours) — jointly optimises causal validity, cost and feasibility.
- **DiCE**, **Wachter**, **Alibi** — established counterfactual methods.
- **Cost oracle** — cheapest flipping plan, ignoring causality (a reference).
- **Random** — a feasible random walk (a floor).
"""),
        code(PREAMBLE),
        code('''
# How good is the box we are explaining? (trained in experiment 01)
try:
    display(read_result("blackbox_performance"))
except FileNotFoundError as e:
    print(e, "- run experiments/01_train_blackbox.py")
'''),
        code('''
from src.methods import available_methods
print("registered methods:", available_methods())
'''),
        md("""
## One plan, read the way its recipient would

A counterfactual is not a row of numbers — it is advice. Below is a single CAC plan
for one Adult applicant: what it asks the person to change, and (because CAC is
causal) what the SCM says will follow from it.
"""),
        code('''
from src.utils.io import load_joblib

res = load_joblib(RESULTS / "counterfactuals_adult.joblib")
record = sorted(res["cac"])[0]
cf = res["cac"][record][0]
print("method:", cf.method, "| record:", record)
print("flips the box:", cf.valid, "| feasible:", cf.feasible,
      "| effort:", round(float(cf.cost), 3))
print("asked to change:", list(cf.changed_features))
cf.frame()  # before / after / changed, per feature
'''),
        md("""
## The three audits, and the gap they open

Every method flips the box (`validity ≈ 1.00`). What separates them is **usable** —
the share of applicants with one plan that flips the box *and* survives the SCM
*and* is something a person could actually do. `usable` is the conjunction of the
three independent audits, and it is where the methods diverge.
"""),
        code('''
headline = read_result("headline")
cols = [c for c in ["dataset", "method", "validity", "causal_validity",
                    "actionability", "usable", "cost_mean"] if c in headline.columns]
headline[cols].sort_values(["dataset", "usable"], ascending=[True, False])
'''),
    ]


# --------------------------------------------------------------- 04 · results

def results() -> list:
    return [
        md("""
# 04 · Results — the validity/usable gap, drawn

This is the argument in four pictures. Each is regenerated here from the CSVs the
experiments wrote, so nothing below re-runs a method or touches a black box.

1. **The gap** — every method flips the box; far fewer give *usable* recourse.
2. **The funnel** — which of the three audits each method loses people to.
3. **The cost of validity** — what closing the gap costs in effort.
4. **The ablation** — which CAC term is doing the closing.
"""),
        code(PREAMBLE),
        code('''
from src.viz.plot_results import plot_all_results

figs = plot_all_results()
for name in ["usable_gap", "recourse_funnel", "cost_of_validity", "ablation"]:
    if name in figs:
        show_image(figs[name][0])
'''),
        md("""
## The advice itself, one applicant at a time

Rates cannot show what a plan *is*. The recourse-plan figure takes the flagship
applicant — chosen by a rule fixed in advance, blind to the numbers it will draw —
and lays every method's plan side by side: solid bars for what a method asks,
faint hatched bars for what the SCM propagates from it, and a red ring where an
audit objected.
"""),
        code('''
from src.viz.plot_counterfactuals import plot_all_counterfactuals, default_flagship

cases = read_result("qualitative_cases")
print("flagship applicant:", default_flagship(cases))
plans = plot_all_counterfactuals(cases)
show_image(plans["flagship"][0])
'''),
        code('''
# The clearest contrast: a HELOC applicant where the baselines fail the SCM audit
# while CAC stays usable. (Skips cleanly if that case was not in this run.)
key = "heloc_side_effect"
if key in plans:
    show_image(plans[key][0])
else:
    print(key, "not produced in this run; available:", sorted(plans))
'''),
        md("""
## What to take away

- Flipping the model is **not** the same as giving usable recourse — the gap is
  large and it is the SCM audit that opens most of it.
- CAC closes the gap, and the ablation shows the **causal term** is what does it:
  remove it and CAC's usable rate collapses toward the baselines'.
- CAC is not free — it typically asks for **more effort** — but it is effort spent
  on a plan that will actually come true. That trade is the paper's claim.
"""),
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write("01_eda.ipynb", eda())
    write("02_scm.ipynb", scm())
    write("03_methods.ipynb", methods())
    write("04_results.ipynb", results())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
