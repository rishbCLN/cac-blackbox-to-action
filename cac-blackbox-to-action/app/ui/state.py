"""Shared session state, sidebar pickers and preflight for the demo pages.

One module owns the sidebar so the judge's dataset/applicant choice follows them
across every page (Streamlit reruns top-to-bottom, so persistence lives in
``st.session_state``).  It also runs a single preflight that decides which pages
can work: results-only pages need ``results/*.csv``; live pages additionally need
trained models + processed data.  Missing artefacts disable a page with a banner
instead of crashing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app import components as C
from app.ui import theme
from src.viz import label_of

# session_state keys
K_DATASET = "cac_dataset"
K_RECORD = "cac_record"
K_DEMO = "cac_demo_mode"
K_SEED = "cac_seed_features"  # sandbox: features to prefill from another page


@dataclass(frozen=True)
class Availability:
    """What the current machine can actually show."""

    results: list[str]   # datasets with stored counterfactual results
    live: list[str]      # datasets with trained models + processed data

    @property
    def any_results(self) -> bool:
        return bool(self.results)

    @property
    def any_live(self) -> bool:
        return bool(self.live)


@st.cache_data(show_spinner=False)
def _live_datasets() -> list[str]:
    """Datasets whose model + processed data exist (import guarded for tests)."""
    try:
        from app import live as L  # noqa: PLC0415

        return list(L.datasets_live())
    except Exception:  # noqa: BLE001 - live engine is optional
        return []


def availability() -> Availability:
    return Availability(results=C.datasets_present(), live=_live_datasets())


def demo_mode() -> bool:
    return bool(st.session_state.get(K_DEMO, False))


def current_dataset() -> str | None:
    return st.session_state.get(K_DATASET)


def set_seed_features(dataset: str, record: int, features: list[str]) -> None:
    """Hand a set of features to the sandbox (used by the causal-map page)."""
    st.session_state[K_SEED] = {"dataset": dataset, "record": int(record),
                                 "features": list(features)}


def take_seed_features(dataset: str, record: int) -> list[str]:
    """Consume any features queued for this (dataset, record); one-shot."""
    seed = st.session_state.get(K_SEED)
    if not seed or seed.get("dataset") != dataset or seed.get("record") != int(record):
        return []
    st.session_state.pop(K_SEED, None)
    return list(seed.get("features", []))


# ------------------------------------------------------------------- sidebar
def sidebar(avail: Availability) -> None:
    """Render the shared sidebar and populate session_state.

    Chooses the dataset every page reads from ``st.session_state`` and offers the
    demo-mode switch and glossary.  Uses the union of results/live datasets so the
    picker is never empty when *some* artefacts exist.
    """
    st.sidebar.title("\U0001F9ED  CAC demo")
    st.sidebar.caption("Black Box \u2192 Actionable Insight")

    options = _dataset_options(avail)
    if not options:
        st.sidebar.error("No artefacts found. Run `make all` in the project root.")
        return

    default = st.session_state.get(K_DATASET) or _preferred_default(options)
    if default not in options:
        default = options[0]
    dataset = st.sidebar.selectbox(
        "Dataset", options, index=options.index(default),
        format_func=label_of, key="_dataset_widget")
    st.session_state[K_DATASET] = dataset
    st.sidebar.caption(theme.DATASET_BLURB.get(dataset, ""))

    st.sidebar.toggle("Demo mode (big text)", key=K_DEMO,
                      help="Larger fonts and cleaner layout for a projector.")

    with st.sidebar.expander("Glossary", expanded=False):
        for term, gloss in theme.GLOSSARY:
            st.markdown(f"**{term}** \u2014 {gloss}")

    st.sidebar.divider()
    st.sidebar.caption("Read-only over stored results; the black box is only ever "
                       "*queried*, never trained here. No network at demo time.")


def _dataset_options(avail: Availability) -> list[str]:
    from src.config import DATASETS  # noqa: PLC0415

    have = set(avail.results) | set(avail.live)
    return [d for d in DATASETS if d in have]


def _preferred_default(options: list[str]) -> str:
    """First dataset (in ``options`` order) that has a real Home hook, else the first.

    A hook needs a curated case with *both* a failing baseline and a usable CAC
    plan; otherwise the Home page opens on the empty ``everyone_agrees`` fallback
    (this is why landing on ``adult`` --- whose only case is all-usable --- shows no
    contrast).  Used only to seed the dataset when the judge has not chosen one, so
    it never overrides an explicit later selection.
    """
    cases = C.load_csv("qualitative_cases")
    if cases is None or cases.empty:
        return options[0]
    for dataset in options:
        block = cases[cases["dataset"] == dataset]
        for case in block["case"].unique():
            sub = block[block["case"] == case]
            usable = sub["usable"].astype(bool)
            cac = sub[sub["method"] == "cac"]
            cac_ok = bool(cac["usable"].astype(bool).iloc[0]) if not cac.empty else False
            if (~usable).any() and cac_ok:
                return dataset
    return options[0]


# ------------------------------------------------------- applicant selection
def applicant_picker(dataset: str, *, live_records: bool = False) -> int | None:
    """A shared applicant chooser: a curated case, or any rejected record.

    ``live_records`` picks the rejected-applicant list from the live engine (the
    sandbox needs indices into ``X_test``); otherwise it uses the records that
    have stored plans.  The chosen record is remembered per dataset.
    """
    cases = C.load_csv("qualitative_cases")
    dcases = cases[cases["dataset"] == dataset] if cases is not None else None

    mode = st.radio(
        "Choose an applicant by",
        ["Curated case study", "Any rejected applicant"],
        horizontal=True, key=f"_appmode_{dataset}",
        help="Curated cases are the ones the paper walks through; or browse the "
             "whole evaluation set.")

    record: int | None = None
    if mode.startswith("Curated") and dcases is not None and not dcases.empty:
        case = st.selectbox("Case", sorted(dcases["case"].unique()),
                            key=f"_case_{dataset}",
                            format_func=_case_label)
        record = int(dcases[dcases["case"] == case]["record"].iloc[0])
        st.caption(f"applicant #{record}")
    else:
        if mode.startswith("Curated"):
            st.info("No curated cases for this dataset; browsing all.")
        records = _record_options(dataset, live_records)
        if not records:
            st.warning("No applicants available for this dataset.")
            return None
        stored = st.session_state.get(K_RECORD)
        default = stored if stored in records else records[0]
        record = int(st.select_slider("Applicant #", options=records, value=default,
                                       key=f"_recslider_{dataset}"))

    if record is not None:
        st.session_state[K_RECORD] = int(record)
    return record


def _record_options(dataset: str, live_records: bool) -> list[int]:
    if live_records:
        try:
            from app import live as L  # noqa: PLC0415

            return list(L.rejected_records(dataset, 40))
        except Exception:  # noqa: BLE001
            return []
    frame = C.per_record(dataset)
    return [] if frame.empty else C.records_for(frame)


def _case_label(case: str) -> str:
    pretty = {
        "everyone_agrees": "Everyone agrees (easy case)",
        "cac_only": "Only CAC gives a usable plan",
        "cheaper_but_wrong": "A baseline is cheaper but wrong",
        "side_effect": "A plan with causal side effects",
        "immutable_advice": "A baseline asks the impossible",
    }
    return pretty.get(case, case.replace("_", " "))


__all__ = [
    "K_DATASET", "K_RECORD", "K_DEMO", "K_SEED",
    "Availability", "availability", "demo_mode", "current_dataset",
    "set_seed_features", "take_seed_features",
    "sidebar", "applicant_picker",
]
