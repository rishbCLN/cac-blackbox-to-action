"""Comprehensive demo walkthrough: exercise every page's data path and verdicts."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from app import components as C
from app.ui import theme, state as S
from app.ui import page_home as H
from app.ui import page_sandbox as SB


def hr(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


# ---- MICROSTEP 0: landing dataset
opts = C.datasets_present()
default = S._preferred_default(opts)
print("Naive user lands on dataset:", default)

# ---- MICROSTEP 1: HOME
hr("HOME page")
cases = C.load_csv("qualitative_cases")
block = cases[cases.dataset == default]
chosen = H._pick_case(block)
story_ds, borrowed = default, False
if chosen is None or H._first_failing(block[block.case == chosen]) is None:
    alt_ds, alt_case = H._best_hook_elsewhere(cases, exclude=default)
    story_ds, chosen, borrowed = alt_ds, alt_case, True
print(f"story dataset={story_ds} case={chosen} borrowed={borrowed}")
rows = cases[(cases.dataset == story_ds) & (cases.case == chosen)]
fail = H._first_failing(rows)
cac = H._row(rows, "cac")
rec = int(rows["record"].iloc[0])
print(f"applicant #{rec}")
print("  failing baseline:", None if fail is None else fail["method"],
      "| why:", None if fail is None else H._why_fails(fail))
print("  CAC usable(theme):", None if cac is None else theme.is_usable(cac))

# ---- MICROSTEP 2: THE GAP
hr("THE GAP page (headline.csv, corrected)")
h = C.load_csv("headline")
hb = h[h.dataset == story_ds].set_index("method")
cacu = float(hb.loc["cac", "usable"]) if "cac" in hb.index else float("nan")
base = hb.drop(index=[m for m in ("cac",) if m in hb.index])
wm = base["usable"].astype(float).idxmin()
print(f"  CAC usable={cacu:.0%}  worst baseline={wm} {float(base.loc[wm,'usable']):.0%}  gap=+{(cacu-float(base.loc[wm,'usable'])):.0%}")
try:
    C.gap_figure(h, story_ds); C.funnel_figure(h, story_ds)
    print("  figures build: OK")
except Exception as e:
    print("  figures FAILED:", repr(e))

# ---- MICROSTEP 3: HEAD TO HEAD (recompute)
hr(f"HEAD-TO-HEAD page ({story_ds} #{rec})")
fr = C.per_record(story_ds)
for m in C.methods_for(story_ds, fr):
    r = C.best_row(fr, m, rec)
    if r is None:
        print(f"  {m:8s} no plan"); continue
    print(f"  {m:8s} USABLE={bool(r.usable)}  (flip={bool(r.causal)} action={bool(r.actionable)} fid={bool(r.state_fidelity)})")
try:
    png = C.recourse_png(story_ds, rec, fr)
    print("  recourse PNG:", "OK" if png and png.exists() else "none")
except Exception as e:
    print("  recourse PNG FAILED:", repr(e))

# ---- MICROSTEP 4: LIVE SANDBOX
hr(f"LIVE SANDBOX ({story_ds})")
from app import live as L
eng = L.get_engine(story_ds)
recs = L.rejected_records(story_ds, 40)
r0 = recs[0]
x = eng.factual_row(r0)
print(f"  applicant #{r0}: model says {eng.predict_label(x)} (0=reject), target={eng.default_target(x)}")
# seed CAC plan
plans = L.generate_plans(story_ds, "cac", r0, 3)
print("  CAC produced", len(plans), "plan(s)")
if plans:
    seeded = L.seed_from_plan(eng, x, plans[0])
    a = eng.audit(x, seeded)
    print(f"  seeded-CAC audit: valid={a.valid} flip_survives={a.flip_survives} actionable={a.actionable} USABLE={a.usable}")
    # now a naive 'max out' edit on first mutable feature -> expect over-budget/impossible
    meta = eng.feature_meta(x)
    mut = next((m for m in meta if not m["immutable"] and not m["levels"]), None)
    if mut:
        j = eng.index[mut["name"]]
        bad = x.copy(); bad[j] = mut["hi"]
        ab = eng.audit(x, bad)
        print(f"  'max out {mut['name']}' audit: actionable={ab.actionable} over_budget={ab.over_budget} USABLE={ab.usable}")
        fcs = SB.cascade_state(ab, eng.consequence_frame(ab))
        print(f"  cascade state: {fcs}")

# ---- MICROSTEP 5: CAUSAL MAP
hr(f"CAUSAL MAP ({story_ds})")
from src.scm.define_scm import get_scm
scm = get_scm(story_ds)
mani = list(scm.manipulable)[:2]
print("  manipulable (sample):", mani)
print("  downstream of", mani, "=", scm.downstream(mani, include_target=True))

# ---- MICROSTEP 6: EVIDENCE
hr("EVIDENCE page CSV availability")
for name in ["significance_aggregate", "significance_tests", "robustness_headline",
             "robustness_model", "ablation_headline"]:
    df = C.load_csv(name)
    print(f"  {name:26s}", "present" if df is not None else "MISSING",
          "" if df is None else f"({len(df)} rows, has state_fidelity={'state_fidelity' in df.columns})")
