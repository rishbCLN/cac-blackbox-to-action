"""Final cross-page consistency check for the demo (post-regeneration)."""
import matplotlib
matplotlib.use("Agg")
from app import components as C
from app.ui import page_evidence as PE  # noqa: F401  (import crash check)

# Gap (headline) vs Head-to-head (per_record) usable rate, per method, heloc.
ds = "heloc"
h = C.load_csv("headline")
hb = h[h.dataset == ds].set_index("method")
fr = C.per_record(ds)
recs = C.records_for(fr)
print("Cross-page usable-rate check (%s, %d records):" % (ds, len(recs)))
print("  method  | headline | per_record(any-k) ")
for m in C.methods_for(ds, fr):
    hv = float(hb.loc[m, "usable"]) if m in hb.index else float("nan")
    sub = fr[fr.method == m]
    per = sub.groupby("record")["usable"].any()
    pv = per.mean() if len(per) else float("nan")
    flag = "" if abs(hv - pv) < 0.02 else "  <-- MISMATCH"
    print("  %-7s |  %.3f   |  %.3f%s" % (m, hv, pv, flag))

# Evidence significance tab: does current aggregate have what the app reads?
agg = C.load_csv("significance_aggregate")
need = ["usable_mean", "usable_lo", "usable_hi", "method", "dataset"]
print("\nsignificance_aggregate present:", agg is not None,
      "| has app-required cols:", all(c in agg.columns for c in need) if agg is not None else None)
tests = C.load_csv("significance_tests")
tneed = ["method", "usable_gap", "gap_lo", "gap_hi", "mcnemar_p_holm", "significant"]
print("significance_tests present:", tests is not None,
      "| has app-required cols:", all(c in tests.columns for c in tneed) if tests is not None else None)
