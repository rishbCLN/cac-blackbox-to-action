"""Live Sandbox --- the judge drives (Phase 4).

Pick a real rejected applicant, edit any feature, and watch the three audits
recompute live on the real model plus the fitted causal model.  The reveal is the
causal-cascade table: change one feature and the structural model fills in the
consequences you did not ask for --- exactly what a non-causal tool hides.

Everything runs through :mod:`app.live`; the engine only ever *queries*
``predict_proba`` and runs one SCM repair per edit.  Nothing is trained here.
"""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from app.ui import state as S
from app.ui import theme
from src.viz import label_of


def render() -> None:
    st.title("Live sandbox: type a plan, watch the world react")

    dataset = S.current_dataset()
    if dataset is None:
        return

    try:
        from app import live as L  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        st.error(f"The live engine could not be imported: {exc}")
        return

    if dataset not in L.datasets_live():
        st.warning(
            f"**{label_of(dataset)}** has no trained model / processed data on "
            "this machine, so the live sandbox is unavailable. Run "
            "`experiments/01_train_blackbox.py` (or `make all`). The other pages "
            "work from stored results.")
        return

    with st.spinner("Loading the model and causal graph once\u2026"):
        eng = L.get_engine(dataset)

    record = _pick_record(L, dataset)
    if record is None:
        return

    x = eng.factual_row(record)
    meta = eng.feature_meta(x)

    st.caption(
        f"Model's current answer for #{record}: "
        f"**{_label(eng.predict_label(x))}**. Goal: flip it to "
        f"**{_label(eng.default_target(x))}**.")

    _seed_controls(L, eng, dataset, record, x, meta)
    st.divider()

    left, right = st.columns([2, 3])
    with left:
        st.markdown("#### Your plan")
        x_edit = _editors(eng, dataset, record, x, meta)
    with right:
        audit = eng.audit(x, x_edit)
        _verdict(audit)
        st.divider()
        _cascade(eng, audit)

    st.divider()
    _details(eng, audit)

    theme.presenter_note(
        "Seed from CAC's plan, show it's USABLE, then hand the mouse over. Ask the "
        "judge to 'max out' a feature or move a downstream one --- watch A go red "
        "(over budget / impossible) or the cascade table reveal side effects that "
        "flip S to red. Every number recomputes on the real model.")


# ------------------------------------------------------------------- controls
def _pick_record(L, dataset: str) -> int | None:
    records = L.rejected_records(dataset, 40)
    if not records:
        st.warning("No rejected applicants found for this dataset.")
        return None
    stored = st.session_state.get(S.K_RECORD)
    default = stored if stored in records else records[0]
    record = int(st.select_slider("Rejected applicant #", options=records,
                                  value=default, key=f"_sbx_rec_{dataset}"))
    st.session_state[S.K_RECORD] = record
    return record


def _seed_controls(L, eng, dataset: str, record: int, x, meta) -> None:
    st.markdown("**Start from a method's plan** (then edit it), or reset:")
    methods = L.live_methods(dataset)
    cols = st.columns(len(methods) + 1)
    for col, method in zip(cols, methods, strict=False):
        if col.button(f"Seed: {label_of(method)}", key=f"_seed_{method}"):
            plans = L.generate_plans(dataset, method, record, 3)
            if plans:
                seeded = L.seed_from_plan(eng, x, plans[0])
                _store_edit(dataset, record, eng, seeded)
            else:
                st.toast(f"{label_of(method)} produced no plan for #{record}.")
    if cols[-1].button("Reset to applicant", key="_seed_reset"):
        _store_edit(dataset, record, eng, x.copy())

    # Features handed over from the causal-map page.
    queued = S.take_seed_features(dataset, record)
    if queued:
        st.info(f"Pre-selected from the causal map: {', '.join(queued)}. "
                "Adjust them below.")


def _editors(eng, dataset: str, record: int, x, meta) -> np.ndarray:
    edit = _current_edit(dataset, record, x)
    key_base = f"{dataset}_{record}"

    for m in meta:
        j = eng.index[m["name"]]
        name = m["name"]
        label = _feat_label(m)
        cur = float(edit[j])

        if m["immutable"]:
            st.text_input(label, value=eng.pretty(name, cur), disabled=True,
                          key=f"_imm_{key_base}_{name}",
                          help="This feature cannot be changed by anyone.")
            continue

        levels = m["levels"]
        if levels:
            idx = int(np.clip(round(cur), 0, len(levels) - 1))
            picked = st.select_slider(label, options=list(range(len(levels))),
                                      value=idx,
                                      format_func=lambda i, lv=levels: lv[int(i)],
                                      key=f"_lev_{key_base}_{name}")
            edit[j] = float(picked)
        else:
            lo, hi = float(m["lo"]), float(m["hi"])
            if m["integer"]:
                val = st.slider(label, min_value=int(lo), max_value=int(hi),
                                value=int(round(cur)), step=1,
                                key=f"_num_{key_base}_{name}")
            else:
                step = max((hi - lo) / 100.0, 1e-6)
                val = st.slider(label, min_value=lo, max_value=hi,
                                value=float(np.clip(cur, lo, hi)), step=step,
                                key=f"_num_{key_base}_{name}")
            edit[j] = float(val)

    _put_edit(dataset, record, edit)
    return edit


def _feat_label(m: dict) -> str:
    tags = []
    if m.get("sensitive"):
        tags.append("sensitive")
    mono = m.get("monotone")
    if mono in ("increase", "decrease"):
        tags.append(f"{mono}-only")
    suffix = f"  \u00b7 _{', '.join(tags)}_" if tags else ""
    return f"{m['name']}{suffix}"


# --------------------------------------------------------------------- verdict
def _verdict(audit) -> None:
    st.markdown("#### Verdict")
    st.markdown(theme.big_verdict(audit), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("P(target) at your plan", f"{audit.proba_target:.2f}")
    if audit.has_causal:
        c2.metric("P(target) after the world reacts",
                  f"{audit.proba_repaired_target:.2f}",
                  delta=f"{audit.proba_repaired_target - audit.proba_target:+.2f}")
    c3.metric("Changes", f"{audit.n_changed} / {_cap(audit)}")

    # effort meter
    frac = float(min(max(audit.normalised, 0.0), 1.0))
    st.progress(frac, text=f"Effort {audit.normalised:.2f} of the 1.00 budget")
    warns = []
    if audit.over_budget:
        warns.append("over the effort budget")
    if audit.over_actions:
        warns.append("too many changes")
    if not audit.feasible:
        warns.append("asks for something impossible")
    if warns:
        st.warning("Not doable: " + ", ".join(warns) + ".")


def _cap(audit) -> int:
    from src.config import CONFIG  # noqa: PLC0415

    return int(CONFIG.MAX_ACTIONS)


# --------------------------------------------------------------------- cascade
def cascade_state(audit, frame) -> str:
    """Which cascade message applies, derived from the audit's truth (not the frame).

    Pure decision so the verdict chips and the cascade caption cannot disagree.
    Four states, because "breaks the graph" and "loses the flip" are independent:

    ``breaks_lost`` --- a feature drifted past tolerance *and* the repaired
    probability fell back under the decision threshold, so the 'yes' is gone.
    ``breaks_survives`` --- a feature drifted past tolerance but the model still
    says 'yes' at the repaired point; the flip holds yet the plan is not causally
    sound, so it misdescribes the world you'd actually land on.
    ``no_survive`` --- nothing broke the graph, but the repaired probability fell
    back under the threshold (``flip_survives`` is False).
    ``survives`` --- the flip holds and the graph agrees.

    Reads ``audit.causal_valid`` (``gap_max <= tol``) rather than the display
    frame's ``breaks causal`` column, so the decision does not depend on the frame
    being present or non-empty.  ``frame`` is kept for signature stability.
    """
    if not getattr(audit, "has_causal", False):
        # No SCM to contradict the plan; the no-causal banner is handled by the
        # caller, so only survives / no_survive can apply here.
        return "survives" if bool(getattr(audit, "flip_survives", True)) else "no_survive"
    broke = not bool(getattr(audit, "causal_valid", True))
    survives = bool(getattr(audit, "flip_survives", True))
    if broke and survives:
        return "breaks_survives"
    if broke:
        return "breaks_lost"
    if not survives:
        return "no_survive"
    return "survives"


def _cascade(eng, audit) -> None:
    st.markdown("#### What the world does in response")
    if not audit.has_causal:
        st.caption("This dataset has no causal model, so nothing is inferred.")
        return
    frame = eng.consequence_frame(audit)
    state = cascade_state(audit, frame)

    if not frame.empty:
        st.dataframe(frame, hide_index=True, width="stretch")

    if state == "breaks_lost":
        st.error("Highlighted rows break causal validity: the world moved a "
                 "feature far from what your plan assumed, so the 'yes' no longer "
                 "holds once reality fills in.")
    elif state == "breaks_survives":
        off = ", ".join(audit.off_features) if audit.off_features else "a feature"
        st.error(
            "The model would still approve after the world reacts, but your plan "
            f"is not causally sound: it assumed **{off}** would stay put and the "
            "graph moves it beyond tolerance. The world you'd actually land on is "
            "not the one your plan describes, so the 'yes' can't be trusted.")
    elif state == "no_survive":
        from src.config import CONFIG  # noqa: PLC0415

        st.error(
            "No single feature breaks the graph, but once the world reacts the "
            f"model's confidence falls from {audit.proba_target:.2f} to "
            f"{audit.proba_repaired_target:.2f} --- back under the "
            f"{float(CONFIG.DECISION_THRESHOLD):.0%} line --- so the 'yes' does not "
            "survive.")
    elif frame.empty:
        st.success("Your plan is causally self-consistent: no hidden side "
                   "effects, so the flip survives.")
    else:
        st.caption("The world moved these features, but within tolerance --- the "
                   "flip still survives.")


# --------------------------------------------------------------------- details
def _details(eng, audit) -> None:
    with st.expander("Plan detail (priced) and full verdict"):
        plan = eng.plan_frame(audit)
        if not plan.empty:
            st.markdown("**Your priced actions**")
            st.dataframe(plan, hide_index=True, width="stretch")
        if audit.off_features:
            st.markdown("**Breaks causal on:** " + ", ".join(audit.off_features))
        if audit.violations:
            viol = ", ".join(f"{f} ({why})" for f, why in audit.violations)
            st.markdown("**Rule violations:** " + viol)
        st.download_button(
            "Download this verdict (JSON)",
            data=json.dumps(audit.as_dict(), indent=2),
            file_name=f"verdict_{audit.dataset}.json", mime="application/json")


# ----------------------------------------------------------------- edit store
def _edit_key(dataset: str, record: int) -> str:
    return f"_edit_{dataset}_{record}"


def _current_edit(dataset: str, record: int, x) -> np.ndarray:
    key = _edit_key(dataset, record)
    if key not in st.session_state:
        st.session_state[key] = np.asarray(x, dtype=float).copy()
    return np.asarray(st.session_state[key], dtype=float).copy()


def _put_edit(dataset: str, record: int, edit) -> None:
    st.session_state[_edit_key(dataset, record)] = np.asarray(edit, dtype=float).copy()


def _store_edit(dataset: str, record: int, eng, edit) -> None:
    """Seed the edit and clear stale widget keys so sliders repopulate."""
    _put_edit(dataset, record, edit)
    prefix = f"{dataset}_{record}"
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) and prefix in k
               for p in ("_lev_", "_num_", "_imm_")):
            st.session_state.pop(k, None)
    st.rerun()


def _label(y: int) -> str:
    return "approve / favourable" if int(y) == 1 else "reject / unfavourable"
