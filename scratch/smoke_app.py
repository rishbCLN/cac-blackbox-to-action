"""Headless smoke test for app/components.py --- no Streamlit runtime needed.

Streamlit's cache decorators execute the wrapped function directly when there is
no script run (they only warn), so every data/view helper can be exercised as a
plain library call.  This proves the demo's logic works against the real
artifacts before anyone launches ``streamlit run``; the UI wiring in app.py is
thin enough to read by eye.  Run from the project root.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")  # silence the "missing ScriptRunContext" noise

from app import components as C  # noqa: E402


def check_components() -> None:
    datasets = C.datasets_present()
    print("datasets present:", datasets)
    assert datasets, "no counterfactual joblibs found; run make all"

    headline = C.load_csv("headline")
    assert headline is not None and not headline.empty

    for dataset in datasets:
        frame = C.per_record(dataset)
        assert not frame.empty, f"no per-record rows for {dataset}"
        methods = C.methods_for(dataset, frame)
        records = C.records_for(frame)
        assert "cac" in methods and records
        assert {"valid", "causal", "actionable", "usable"} <= set(frame.columns)

        # the gap chart renders
        fig = C.gap_figure(headline, dataset)
        import matplotlib.pyplot as plt  # noqa: PLC0415
        plt.close(fig)

        # a representative applicant: verdicts, moves, and a live plan figure
        record = records[0]
        row = C.best_row(frame, "cac", record)
        assert row is not None
        _ = C.verdict_markdown(row)
        moves = C.moves_table(row)
        # Gap C: side-effect features must render with clean names, never the
        # internal ``:immutable``/``:sensitive`` encoding.
        assert not any(":" in str(f) for f in moves["feature"]), \
            f"moves_table leaked a raw feature tag for {dataset}#{record}"
        png = C.recourse_png(dataset, record, frame)
        assert png is not None and png.exists(), f"no plan png for {dataset}#{record}"
        print(f"  {dataset}: {len(methods)} methods, {len(records)} applicants, "
              f"plan -> {png.name}")


def check_live() -> None:
    """Exercise the live engine the way the sandbox will --- no Streamlit runtime."""

    from app import live as L  # noqa: PLC0415

    datasets = L.datasets_live()
    print("datasets live:", datasets)
    assert datasets, "no trained models found; run experiments/01 first"

    for dataset in datasets:
        eng = L.get_engine(dataset)
        methods = L.live_methods(dataset)
        assert "cac" in methods, f"cac unavailable for {dataset}"
        records = L.rejected_records(dataset, 5)
        assert records, f"no rejected applicants for {dataset}"

        record = records[0]
        x = eng.factual_row(record)
        y0 = eng.predict_label(x)
        yt = eng.default_target(x)
        assert y0 != yt, "a rejected applicant must want the other class"

        # Factual == no plan: nothing changed, no effort, no causal gap.
        base = eng.audit(x, x.copy())
        assert base.n_changed == 0 and base.cost == 0.0
        assert not base.valid  # still rejected, by definition of the query set
        if eng.has_causal:
            assert base.gap_max == 0.0 and base.causal_valid

        # Editing an immutable feature is never actionable.
        meta = eng.feature_meta(x)
        imm = next((m for m in meta if m["immutable"]), None)
        if imm is not None:
            j = eng.index[imm["name"]]
            bumped = x.copy()
            lo, hi = eng.bounds[j]
            bumped[j] = hi if x[j] < hi else lo
            bad = eng.audit(x, bumped)
            assert not bad.actionable, f"{imm['name']} edit should be inadmissible"

        # A real CAC plan, scored the way the paper does (declared endpoint,
        # priced on its asked subset), should be usable.
        plans = L.generate_plans(dataset, "cac", record, 3)
        assert plans, f"cac produced no plan for {dataset}#{record}"
        audited = [eng.audit_plan(x, cf) for cf in plans]
        assert any(a.usable for a in audited), f"no usable cac plan for {dataset}#{record}"
        best = next(a for a in audited if a.usable)
        assert best.valid and best.actionable
        assert best.n_changed <= eng.max_actions
        # usable is exactly the conjunction the headline leads with.
        assert best.usable == (best.valid and best.promise_kept and best.actionable)

        # The free-typed path: seeding only the asked features and letting the SCM
        # fill in the rest audits the *action-only* point a non-causal tool shows.
        typed = eng.audit(x, L.seed_from_plan(eng, x, plans[0]))
        assert typed.n_changed <= len(plans[0].changed_features)
        _ = typed.flip_survives  # may differ from the declared plan's verdict

        # Gap B: the sandbox cascade caption must agree with the verdict. When the
        # flip does not survive the repair, cascade_state must never say "survives".
        from app.ui.page_sandbox import cascade_state  # noqa: PLC0415

        for cf in plans:
            a = eng.audit(x, L.seed_from_plan(eng, x, cf))
            state = cascade_state(a, eng.consequence_frame(a))
            # The four states are a total, exclusive function of (causal_valid,
            # flip_survives) once a dataset has an SCM; lock every branch so the
            # caption can never contradict the audit's truth.
            if not a.flip_survives:
                assert state != "survives", (
                    f"cascade claims survival while flip_survives is False "
                    f"({dataset}#{record})")
            if a.promise_kept:
                assert state == "survives", (
                    f"cascade denies survival while promise_kept is True "
                    f"({dataset}#{record})")
            if eng.has_causal:
                broke = not a.causal_valid
                expected = ("breaks_survives" if broke and a.flip_survives else
                            "breaks_lost" if broke else
                            "no_survive" if not a.flip_survives else "survives")
                assert state == expected, (
                    f"cascade_state {state!r} != {expected!r} for "
                    f"causal_valid={a.causal_valid} flip_survives={a.flip_survives} "
                    f"({dataset}#{record})")

        # Display frames build and decode round-trips.
        _ = eng.plan_frame(best)
        _ = eng.consequence_frame(best)
        _ = eng.decode(x)
        _ = best.as_dict()
        if eng.has_causal:
            print(f"  {dataset}: {len(methods)} methods, {len(records)} rejected, "
                  f"cac usable plan changes {best.n_changed} feature(s), "
                  f"gap={best.gap_max:.3f}")
        else:
            print(f"  {dataset}: {len(methods)} methods, {len(records)} rejected "
                  f"(no SCM), cac usable plan changes {best.n_changed} feature(s)")


def check_pages() -> None:
    """Exercise the page-level figure/logic helpers headlessly.

    Streamlit view calls (st.*) no-op without a runtime; we only drive the pure
    logic and matplotlib builders each page relies on, against the real CSVs.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from app.ui import page_evidence as E  # noqa: PLC0415

    for dataset in C.datasets_present():
        agg = C.load_csv("significance_aggregate")
        if agg is not None:
            block = agg[agg["dataset"] == dataset]
            if not block.empty:
                E._ci_figure(block)
                plt.close("all")

        abl = C.load_csv("ablation_headline")
        if abl is not None:
            ab = abl[abl["dataset"] == dataset]
            if not ab.empty:
                E._ablation_figure(ab)
                plt.close("all")
                full = E._val(ab, "cac", "usable")
                noc = E._val(ab, "cac_no_causal", "usable")
                if full == full and noc == noc:  # noqa: PLR0124 - NaN check
                    assert full >= noc, "removing the causal term should not help"

        # Gap D: the k and budget robustness sweeps build for every dataset.
        E._rob_sweep(dataset, arm="k", label="k", x_label="k",
                     order=("k=1", "k=3", "k=5", "k=10"), heading="k", caption="c")
        E._rob_sweep(dataset, arm="budget", label="budget", x_label="budget",
                     order=("actions=1", "actions=2", "actions=3"),
                     heading="b", caption="c")
        plt.close("all")

        # Causal-map figures for datasets that have an SCM.
        try:
            from src.scm.define_scm import get_scm  # noqa: PLC0415
            from src.viz import plot_graphs  # noqa: PLC0415

            scm = get_scm(dataset)
        except Exception:  # noqa: BLE001 - dataset may have no graph
            continue
        picked = list(scm.manipulable)[:1]
        plt.close(plot_graphs.plot_scm(scm, show_costs=True))
        if picked:
            plt.close(plot_graphs.plot_intervention_paths(scm, picked))
            _ = scm.downstream(picked, include_target=True)
    print("APP PAGES OK")


def main() -> int:
    check_components()
    check_live()
    check_pages()
    print("APP COMPONENTS OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
