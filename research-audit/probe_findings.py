"""Reproduce audit findings using the unmodified repository implementation.
Run from the repository root after data/model preparation; PYTHONPATH=.
These are diagnostic probes, not new benchmark results.
"""
import importlib.util
import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pandas as pd
import networkx as nx
from src.data.schema import DatasetSpec, FeatureSpec
from src.scm.define_scm import SCM
from src.scm.fcm import FCM, NodeEquation
from src.methods.base import Counterfactual, get_method
from src.evaluation import metrics, faithfulness, feasibility

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent / "evidence"
findings = {}

# 1. Repeating the very same applicants can turn non-significance into significance.
spec = importlib.util.spec_from_file_location("exp06", ROOT / "experiments/06_significance.py")
exp06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp06)
for seeds in (1, 5):
    rows = []
    for seed in range(seeds):
        for record in range(20):
            for method in ("cac", "oracle"):
                rows.append(dict(dataset="german", method=method, seed=seed,
                                 record=record, usable=method == "cac" or record >= 5))
    result = exp06._paired_tests(pd.DataFrame(rows)).iloc[0]
    findings[f"identical_records_repeated_{seeds}_seeds"] = {
        k: result[k].item() if hasattr(result[k], "item") else result[k]
        for k in ("n_pairs", "mcnemar_p", "significant", "usable_gap", "gap_lo", "gap_hi")}

# 2. The manuscript's flip-survival definition differs from the implemented gate.
features = (FeatureSpec("a", "numeric", bounds=(0, 10)),
            FeatureSpec("b", "numeric", bounds=(0, 10)))
ds = DatasetSpec(name="toy", features=features, target="target",
                 positive_meaning="accept", negative_meaning="reject", source="known SCM", citation="audit")
graph = nx.DiGraph([("a", "b"), ("b", "target")])
scm = SCM("toy", graph, "target", (), (), {"a": .1, "b": .1},
          {"a": "low", "b": "low"}, spec=ds)
fcm = FCM("toy", scm=scm, spec=ds)
class Twice:
    def predict(self, X):
        return 2 * np.asarray(X)[:, 0]
fcm.equations = {
    "a": NodeEquation("a", "numeric", (), (), ()),
    "b": NodeEquation("b", "numeric", ("a",), ("numeric",), (0,), model=Twice()),
}
class Box:
    def predict_proba(self, X):
        p = np.where(np.asarray(X)[:, 0] >= 1, .9, .1)
        return np.column_stack([1-p, p])
cf = Counterfactual(np.array([0., 0.]), np.array([1., 0.]), 0, 1, 1,
                    .1, ("a",), (), "probe")
ev = SimpleNamespace(fcm=fcm, span=np.array([10., 10.]), model=Box(),
                     feature_names=("a", "b"), dataset="toy")
audit = faithfulness.assess(ev, {0: [cf]}).iloc[0]
findings["paper_vs_code_causal_gate"] = {
    "proposed": cf.x_cf.tolist(), "replayed": fcm.do(cf.x_original, {"a": 1}).tolist(),
    **{k: audit[k].item() if hasattr(audit[k], "item") else audit[k]
       for k in ("valid", "flip_survives", "gap_max", "promise_kept")}}

# 3. A real valid CAC plan with its action list removed still passes the evaluator.
ev = metrics.Evaluator.build("german")
record = int(metrics.query_instances("german", model=ev.model, bundle=ev.bundle, n=1)[0])
x = ev.bundle.X_test[record]
method = get_method("cac", "german", model=ev.model)
cfs = method.generate(x, y_target=1, k=1)
if not cfs or not cfs[0].valid:
    raise RuntimeError("Probe requires one valid CAC suggestion")
for label, proposal in [("original", cfs[0]), ("empty_declared_actions", replace(cfs[0], changed_features=()))]:
    result = {record: [proposal]}
    m = ev.table(result, method=proposal.method)
    f = faithfulness.assess(ev, result)
    a = feasibility.assess(ev, result)
    joined = feasibility.join_audits(m, f, a).iloc[0]
    findings[f"action_metadata_{label}"] = {
        k: joined[k].item() if hasattr(joined[k], "item") else joined[k]
        for k in ("valid", "cost", "causal_valid", "actionable", "usable")}
    replay = ev.fcm.do(x, {n: proposal.x_cf[ev.feature_names.index(n)] for n in proposal.changed_features})
    findings[f"action_metadata_{label}"]["declared_action_replay_probability"] = float(ev.model.predict_proba(replay.reshape(1,-1))[0,1])

path = OUT / "probes.json"
path.write_text(json.dumps(findings, indent=2) + "\n")
print(json.dumps(findings, indent=2))
