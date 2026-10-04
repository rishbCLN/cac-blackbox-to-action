"""Compare headline definitions and verify actual CAC declared-action replay."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from src.evaluation import feasibility
from src.scm.fcm import get_fcm
from src.model.blackbox import load_blackbox

OUT = Path(__file__).resolve().parent / "evidence"
R = OUT / "smoke-results"
m = pd.read_csv(R / "counterfactuals.csv")
f = pd.read_csv(R / "faithfulness_per_cf.csv")
a = pd.read_csv(R / "feasibility_per_cf.csv")
joined = feasibility.join_audits(m, f, a)
joined["paper_definition_usable"] = joined.valid & joined.flip_survives & joined.actionable
summary = joined.groupby(["dataset", "method", "record"])[["usable", "paper_definition_usable"]].any()
summary = summary.groupby(["dataset", "method"]).mean().reset_index()
summary.to_csv(OUT / "metric-definition-comparison.csv", index=False)
print(summary.to_string(index=False))

reports = []
for dataset in ("adult", "german", "heloc", "diabetes"):
    fcm = get_fcm(dataset)
    model = load_blackbox(dataset)
    plans = joblib.load(R / f"counterfactuals_{dataset}.joblib")["cac"]
    max_gap = 0.0
    lost = 0
    total = 0
    for cfs in plans.values():
        for cf in cfs:
            actions = {n: cf.x_cf[fcm.index[n]] for n in cf.changed_features}
            replay = fcm.do(cf.x_original, actions)
            gap = np.max(np.abs(replay-cf.x_cf) / np.maximum(fcm.bounds[:,1]-fcm.bounds[:,0], 1e-12))
            max_gap = max(max_gap, float(gap))
            lost += int(model.predict(replay.reshape(1,-1))[0] != cf.y_target)
            total += 1
    reports.append(dict(dataset=dataset, n_cac_suggestions=total,
                        max_normalized_declared_replay_gap=max_gap, rejected_after_replay=lost))
(OUT / "actual-cac-replay.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2))
