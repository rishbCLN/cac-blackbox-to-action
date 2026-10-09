"""Regenerate headline.csv from stored per-cf CSVs using corrected feasibility.headline (Finding 3)."""
import pandas as pd

from src.config import CONFIG, DATASETS
from src.evaluation import feasibility, metrics as M

frame_all = pd.read_csv(CONFIG.RESULTS_DIR / "counterfactuals.csv")
faith_all = pd.read_csv(CONFIG.RESULTS_DIR / "faithfulness_per_cf.csv")
feas_all = pd.read_csv(CONFIG.RESULTS_DIR / "feasibility_per_cf.csv")

heads = []
for ds in DATASETS:
    fr = frame_all[frame_all.dataset == ds]
    if fr.empty:
        continue
    fa = faith_all[faith_all.dataset == ds]
    fe = feas_all[feas_all.dataset == ds]
    ratios = M.cost_ratio(fr)
    h = feasibility.headline(fr, fa, fe, ratios=ratios)
    if "dataset" not in h.columns:
        h.insert(0, "dataset", ds)
    heads.append(h)

out = pd.concat(heads, ignore_index=True)
print("NEW headline columns:", list(out.columns))
print(out[["dataset", "method", "validity", "causal_validity", "state_fidelity",
           "actionability", "usable"]].to_string(index=False))
out.to_csv(CONFIG.RESULTS_DIR / "headline.csv", index=False)
print("\nWrote", CONFIG.RESULTS_DIR / "headline.csv")
