import time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from app import components as C, live as L

def b(v): return bool(v) and not (isinstance(v, float) and np.isnan(v))

t0 = time.time()
total = 0
for ds in L.datasets_live():
    eng = L.get_engine(ds)
    frame = C.per_record(ds)
    store = C.plans(ds)
    mism = 0
    for m, byrec in store.items():
        for r in list(byrec.keys())[:4]:
            x = eng.factual_row(r)
            for rank, cf in enumerate(byrec[r]):
                row = frame[(frame.method == m) & (frame.record == r) & (frame["rank"] == rank)]
                if row.empty:
                    continue
                row = row.iloc[0]
                a = eng.audit_plan(x, cf)
                total += 1
                for lv, sv in [(a.valid, row.valid), (a.causal_valid, row.causal),
                               (a.actionable, row.actionable), (a.usable, row.usable)]:
                    if b(lv) != b(sv):
                        mism += 1
    print(ds, "mism", mism)
print("total", total, "secs", round(time.time() - t0, 1))
