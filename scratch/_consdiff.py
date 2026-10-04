import warnings
warnings.filterwarnings("ignore")
import numpy as np
from app import components as C, live as L

def b(v): return bool(v) and not (isinstance(v, float) and np.isnan(v))

for ds in ["adult", "heloc", "diabetes"]:
    eng = L.get_engine(ds)
    frame = C.per_record(ds)
    store = C.plans(ds)
    print("===", ds, "===")
    shown = 0
    for m, byrec in store.items():
        for r in list(byrec.keys())[:4]:
            x = eng.factual_row(r)
            for rank, cf in enumerate(byrec[r]):
                row = frame[(frame.method == m) & (frame.record == r) & (frame["rank"] == rank)]
                if row.empty:
                    continue
                row = row.iloc[0]
                a = eng.audit_plan(x, cf)
                fields = {
                    "valid": (a.valid, row.valid),
                    "causal": (a.causal_valid, row.causal),
                    "actionable": (a.actionable, row.actionable),
                    "usable": (a.usable, row.usable),
                }
                diffs = {k: (b(lv), b(sv)) for k, (lv, sv) in fields.items() if b(lv) != b(sv)}
                if diffs and shown < 8:
                    shown += 1
                    print(f"  {m}#{r} r{rank} diffs={diffs} "
                          f"gap={a.gap_max:.3f} cost={a.cost:.3f} n_ch={a.n_changed} "
                          f"asked={list(cf.changed_features)}")
