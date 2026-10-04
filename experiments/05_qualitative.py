"""Experiment 05 — the advice itself, read the way the person receiving it would.

    python experiments/05_qualitative.py                    # every dataset, every case
    python experiments/05_qualitative.py -d adult -c cac_only
    python experiments/05_qualitative.py --records 12 34    # follow specific applicants

Every other experiment reports rates.  A rate is the right unit for a claim about
a method and the wrong one for the claim this project actually makes — that the
advice a counterfactual method hands a person can be *wrong in a way the person
cannot see*.  That is a statement about a sentence someone reads, so this
experiment prints the sentences.

Nothing is generated here.  The plans are the ones ``results/counterfactuals_
<dataset>.joblib`` recorded in experiment 02, re-scored through the same three
audits, so a case study and a row of ``results/headline.csv`` cannot disagree.

Five cases, each a question a reader arrives with, and each selected by a stated
rule rather than by hand:

``cac_only``          CAC's plan is usable and the strongest baseline's is not.
                      The headline case: what does causal validity change about
                      what a person is told to do?
``cheaper_but_wrong`` the advice a baseline would hand over flips the model *more
                      cheaply* than CAC's, and the SCM says its world does not
                      exist.  The honest counter-case --- this is the advice an
                      applicant would rather hear, and following it would not work.
``immutable_advice``  a baseline's flipping plan asks for a change to something
                      nobody can change.  Inactionability at its starkest.
``side_effect``       the shipped plan with the largest disagreement between what
                      it proposes and what the world would do in response.
``everyone_agrees``   every method is usable.  Included on purpose: a paper that
                      only ever shows the cases its method wins looks like one.

Anecdote becomes evidence only with a denominator, so every case is also
*counted*: ``results/qualitative_frequency.csv`` reports how many records of the
query set fall in each case, and the narrative shows one of them.

Outputs
-------
``results/qualitative_cases.csv``       one row per (case, method) --- flags, cost, advice
``results/qualitative_frequency.csv``   how many records each case covers
``results/qualitative_narratives.txt``  the side-by-side a human reads
``paper/tables/case_study.tex``         the flagship case, one row per method
``paper/tables/qualitative_cases.tex``  the frequency table
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python experiments/05_...` without installing
    sys.path.insert(0, str(ROOT))

from src.config import CONFIG, DATASETS  # noqa: E402
from src.evaluation import faithfulness, feasibility, metrics  # noqa: E402
from src.methods.base import Counterfactual  # noqa: E402
from src.utils.io import load_joblib, save_dataframe, save_latex_table  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402

LOGGER = get_logger("experiments.05_qualitative")

#: The baseline the cases are argued against — the one §9 finds strongest.
RIVAL = "dice"

#: The dataset whose flagship case becomes ``paper/tables/case_study.tex``.
FLAGSHIP = "adult"

#: Reading order for the methods inside one narrative.
METHOD_ORDER = ("cac", "oracle", "dice", "wachter", "random")

#: The published methods a case is allowed to *accuse*.  ``oracle`` sees the SCM
#: and ``random`` is a control, so "a baseline undercuts CAC with an impossible
#: plan" is only an interesting sentence about a library somebody actually uses.
#: The cases that *credit* every method keep both — see :func:`_comparison`.
LIBRARY = ("dice", "wachter", "alibi")

OUTPUTS: dict[str, str] = {
    "cases": "qualitative_cases.csv",
    "frequency": "qualitative_frequency.csv",
}

_DATASET_LABELS = {"adult": "Adult", "german": "German", "heloc": "HELOC",
                   "diabetes": "Diabetes"}


@dataclasses.dataclass(frozen=True)
class Case:
    """One question a reader arrives with, and the rule that answers it.

    ``pick`` takes the per-record comparison table and returns the records that
    qualify, most instructive first.  It returns *all* of them, not one: the count
    is the denominator that turns the narrative from an anecdote into a claim
    about how often this happens.
    """

    name: str
    question: str
    caption: str
    pick: Callable[[pd.DataFrame], pd.Series]
    #: Which of the ``k`` suggestions the narrative shows.  ``"usable"`` — the
    #: default — is the one the method would actually ship, and the one the
    #: headline funnel counted.  Two cases are about a *rule being broken* rather
    #: than about the plan on top of the pile, and for those the narrative has to
    #: show the suggestion that breaks it, or the prose and the count would be
    #: about different sentences.  All ``k`` are put to the applicant either way.
    prefer: str = "usable"


def _prefer(rows: pd.DataFrame, prefer: str) -> pd.DataFrame:
    """Order one method's suggestions for one record, best-to-show first."""
    valid = rows[rows["valid"]]
    if prefer == "immutable" and len(valid) and bool((valid["immutable"] > 0).any()):
        return valid[valid["immutable"] > 0].sort_values("cost", kind="stable")
    for pool in (rows[rows["usable"]], valid, rows):
        if len(pool):
            return pool.sort_values("cost", kind="stable")
    return rows


def _per_record(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per (record, method): what that method managed for that person.

    Per record rather than per suggestion, and by the same rule the headline
    funnel uses — five suggestions are one piece of advice with options, and a
    method needs one that survives all three tests, not three that each survive a
    different one.
    """
    rows: list[dict[str, Any]] = []
    for (record, method), group in joined.groupby(["record", "method"], sort=True):
        valid = group[group["valid"]]
        usable = group[group["usable"]]
        # The plan this method would actually hand over, so a rule written about
        # "its advice" and a narrative that prints it cannot pick different rows.
        shown = _prefer(group, "usable").iloc[0]
        rows.append({
            "record": int(record),
            "method": str(method),
            "valid": bool(len(valid)),
            "causal": bool((valid["flip_survives"]).any()),
            "actionable": bool((valid["actionable"]).any()),
            "usable": bool(len(usable)),
            "cost_usable": float(usable["cost"].min()) if len(usable) else np.nan,
            "cost_valid": float(valid["cost"].min()) if len(valid) else np.nan,
            # The shipped plan's own effort and verdict, which are not the
            # method-level answers above: a method whose bargain plan is impossible
            # and whose dearer plan is fine passes ``causal`` on the second and
            # ships the second, so an accusation aimed at the first would be about
            # advice nobody is given.  ``NaN``/``False`` when nothing flipped —
            # comparing the price of a plan that does not work is meaningless.
            "cost_shown": float(shown["cost"]) if bool(shown["valid"]) else np.nan,
            "causal_shown": bool(shown["valid"]) and bool(shown["flip_survives"]),
            "gap_max": float(valid["gap_max"].max()) if len(valid) else np.nan,
            "gap_shown": float(shown["gap_max"]) if bool(shown["valid"]) else np.nan,
            "immutable": int(valid["immutable"].max()) if len(valid) else 0,
            "sensitive_shift": (float(valid["sensitive_shift"].max())
                                if len(valid) else 0.0),
        })
    return pd.DataFrame(rows)


def _comparison(per: pd.DataFrame) -> pd.DataFrame:
    """CAC beside every baseline, one row per record.

    ``rival_*`` columns are the strongest baseline's, and ``any_*``/``all_*`` the
    best and worst any baseline managed: a case that only holds against one
    library is a case about that library, so the rules below are written against
    both.

    The two families have deliberately different denominators.  ``any_``/``all_``
    range over *every* baseline including ``oracle`` and ``random``, because a
    case that credits everybody should be as hard to satisfy as possible.
    ``best_``/``worst_`` range over :data:`LIBRARY` only, because those cases
    accuse a baseline by name and the accusation is only interesting about a
    method somebody would actually run.
    """
    wide = per.pivot(index="record", columns="method")
    out = pd.DataFrame(index=wide.index)
    others = [m for m in per["method"].unique() if m != "cac"]
    library = [m for m in others if m in LIBRARY]
    for column in ("valid", "causal", "actionable", "usable", "cost_usable",
                   "cost_valid", "cost_shown", "gap_max", "gap_shown", "immutable"):
        frame = wide[column]
        out[f"cac_{column}"] = frame["cac"] if "cac" in frame else np.nan
        # The rival's columns exist only if the rival ran.  A missing *value* means
        # it answered this record with nothing, which is a case worth printing; a
        # missing *column* means there was nobody to compare against, and the rules
        # below have to tell those apart — read as the first, a run of CAC alone
        # would report every applicant as one CAC served and the rival did not.
        if RIVAL in frame:
            out[f"rival_{column}"] = frame[RIVAL]
        accused = [m for m in library if m in frame]
        present = [m for m in others if m in frame]
        # ``idxmin``/``idxmax`` are asked only about records where somebody
        # produced something: an all-NA row means no library method answered, and
        # naming a winner among no candidates is a question with no answer (and,
        # since pandas 2.1, a warning that becomes an exception).
        if column in ("cost_usable", "cost_valid", "cost_shown") and accused:
            block = frame[accused]
            answered = block.notna().any(axis=1)
            out[f"best_{column}"] = block.min(axis=1)
            out[f"best_{column}_by"] = block[answered].idxmin(axis=1) \
                                                      .reindex(block.index)
        elif column in ("gap_max", "gap_shown", "immutable") and accused:
            block = frame[accused]
            answered = block.notna().any(axis=1)
            out[f"worst_{column}"] = block.max(axis=1)
            out[f"worst_{column}_by"] = block[answered].idxmax(axis=1) \
                                                       .reindex(block.index)
        elif present:
            out[f"any_{column}"] = frame[present].any(axis=1)
            out[f"all_{column}"] = frame[present].all(axis=1)
    # Whether the baseline that undercut CAC on price is one the SCM believes.
    # The saving only matters if it is not real, and only the plan that achieved it
    # can say — so the flag is the *shipped* plan's, looked up per record from the
    # method that shipped it.  ``True`` when nobody undercut CAC, so a missing
    # comparison never reads as an accusation.
    if "best_cost_shown_by" in out.columns:
        causal = wide["causal_shown"]
        out["undercut_causal"] = [
            bool(causal.loc[record, method])
            if isinstance(method, str) and method in causal.columns else True
            for record, method in zip(out.index, out["best_cost_shown_by"])]
    return out.reset_index()


def _pick_cac_only(cmp: pd.DataFrame) -> pd.Series:
    if "rival_usable" not in cmp.columns:
        return pd.Series(dtype=int)
    hit = cmp[cmp["cac_usable"].fillna(False) & ~cmp["rival_usable"].fillna(False)]
    return hit.sort_values("rival_gap_max", ascending=False)["record"]


def _pick_cheaper_but_wrong(cmp: pd.DataFrame) -> pd.Series:
    """The advice a baseline would hand over, cheaper than CAC's and not real.

    Both halves are about the *shipped* plan.  Asked of the cheapest flipping plan
    instead, this fires on a baseline whose bargain is impossible but whose shipped
    advice is sound and cheap — an accusation about a suggestion the applicant is
    not the one being given.
    """
    if "best_cost_shown" not in cmp.columns:
        return pd.Series(dtype=int)
    hit = cmp[cmp["cac_usable"].fillna(False)
              & (cmp["best_cost_shown"] < cmp["cac_cost_usable"])
              & ~cmp["undercut_causal"].fillna(True)]
    # Derived from ``hit`` rather than from ``cmp``: an ``assign`` of a full-length
    # column onto an empty selection puts every record back, and a case with no
    # examples would publish the whole query set as its frequency.
    saving = (hit["cac_cost_usable"] - hit["best_cost_shown"]) / \
             hit["cac_cost_usable"].replace(0.0, np.nan)
    return hit.assign(_saving=saving).sort_values("_saving", ascending=False)["record"]


def _pick_immutable_advice(cmp: pd.DataFrame) -> pd.Series:
    if "worst_immutable" not in cmp.columns:
        return pd.Series(dtype=int)
    hit = cmp[cmp["worst_immutable"].fillna(0) > 0]
    return hit.sort_values("worst_immutable", ascending=False)["record"]


def _pick_side_effect(cmp: pd.DataFrame) -> pd.Series:
    if "worst_gap_shown" not in cmp.columns:
        return pd.Series(dtype=int)
    hit = cmp[cmp["worst_gap_shown"].fillna(0.0) > float(CONFIG.CAUSAL_TOLERANCE)]
    return hit.sort_values("worst_gap_shown", ascending=False)["record"]


def _pick_everyone_agrees(cmp: pd.DataFrame) -> pd.Series:
    if "all_usable" not in cmp.columns:
        return pd.Series(dtype=int)
    hit = cmp[cmp["cac_usable"].fillna(False) & cmp["all_usable"].fillna(False)]
    return hit.sort_values("cac_cost_usable")["record"]


CASES: tuple[Case, ...] = (
    Case("cac_only",
         f"CAC gives this person a plan that works; {RIVAL} does not.",
         "CAC's plan is usable and the strongest baseline's is not",
         _pick_cac_only),
    Case("cheaper_but_wrong",
         "A baseline undercuts CAC on effort with a plan the SCM rejects.",
         "a baseline's plan is cheaper than CAC's and causally impossible",
         _pick_cheaper_but_wrong),
    Case("immutable_advice",
         "One of the plans a baseline offers asks for something nobody can change.",
         "a flipping plan touches an immutable feature",
         _pick_immutable_advice, prefer="immutable"),
    Case("side_effect",
         "The advice a baseline would actually hand over, and the world it "
         "would not produce.",
         "a shipped plan disagrees most widely with the SCM",
         _pick_side_effect),
    Case("everyone_agrees",
         "An applicant every method serves — CAC is not always different.",
         "every method's plan is usable",
         _pick_everyone_agrees),
)


# ------------------------------------------------------------------ the re-score

def _score(dataset: str) -> tuple[metrics.Evaluator,
                                  dict[str, dict[int, list[Counterfactual]]],
                                  pd.DataFrame]:
    """Load experiment 02's plans and put them through the same three audits.

    Re-scored rather than re-read from ``results/counterfactuals.csv`` because a
    narrative needs what a rate does not: the repaired world, the named
    off-target features, the rendered actions.  Re-*generating* them would be the
    mistake — a case study of a plan the headline table never saw would be a
    different experiment wearing this one's numbers.
    """
    path = CONFIG.RESULTS_DIR / f"counterfactuals_{dataset}.joblib"
    if not path.exists():
        raise SystemExit(f"{path} is missing — run experiments/02_compare_methods.py "
                         f"-d {dataset} first")
    results: dict[str, dict[int, list[Counterfactual]]] = load_joblib(path)
    evaluator = metrics.Evaluator.build(dataset)
    tables, faith, feas = [], [], []
    for method, per_record in results.items():
        tables.append(evaluator.table(per_record, method=method))
        faith.append(faithfulness.assess(evaluator, per_record, method=method))
        feas.append(feasibility.assess(evaluator, per_record, method=method))
    joined = feasibility.join_audits(pd.concat(tables, ignore_index=True),
                                     pd.concat(faith, ignore_index=True),
                                     pd.concat(feas, ignore_index=True),
                                     columns=True)
    LOGGER.info("%s: re-scored %d proposals from %d methods over %d records",
                dataset, int((joined["rank"] >= 0).sum()),
                joined["method"].nunique(), joined["record"].nunique())
    return evaluator, results, joined


def _chosen(joined: pd.DataFrame, record: int, method: str,
            prefer: str = "usable") -> "pd.Series[Any] | None":
    """The suggestion this method would be judged on for this person.

    The cheapest usable one; failing that the cheapest that at least flips the
    decision; failing that whatever it ranked first — the same preference order
    :func:`~src.evaluation.feasibility.headline` applies, with the fallbacks added
    because a case study has to be able to print a failure.  ``prefer`` overrides
    it for the cases that are about a broken rule rather than about the plan on
    top of the pile; see :attr:`Case.prefer`.
    """
    rows = joined[(joined["record"] == record) & (joined["method"] == method)
                  & (joined["rank"] >= 0)]
    if rows.empty:
        return None
    return _prefer(rows, prefer).iloc[0]


# ------------------------------------------------------------------ the sentences

def _profile(evaluator: metrics.Evaluator, x: np.ndarray) -> str:
    """The applicant, in the words the dataset uses rather than in floats."""
    pretty = evaluator.cost_model.pretty
    parts = [f"{name} = {pretty(name, value)}"
             for name, value in zip(evaluator.feature_names, np.asarray(x, float))]
    return textwrap.fill(", ".join(parts), width=84,
                         initial_indent="    ", subsequent_indent="    ")


def _verdict(row: "pd.Series[Any]") -> str:
    """The three audits as three boxes, in funnel order.

    "The SCM agrees" is replayed-action success — the person acts, the SCM fills
    in the consequences, and the box still accepts the result (Finding 3).  State
    fidelity (how far the proposed state strays from that replayed one) is a
    separate quantity, shown in the side-effect lines below rather than gated here.
    """
    marks = (("flips the decision", row.get("valid")),
             ("the SCM agrees", row.get("flip_survives")),
             ("a person could do it", row.get("actionable")))
    return "   ".join(f"[{'x' if bool(v) else ' '}] {label}" for label, v in marks)


def _num(value: Any, digits: int = 3) -> str:
    """A number, or ``n/a`` — a case study that prints ``nan`` looks like a bug."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _side_effects(evaluator: metrics.Evaluator, cf: Counterfactual,
                  limit: int = 4) -> list[str]:
    """Where the plan and the world disagree, worst first.

    The one thing a rate cannot print.  ``gap_max`` says a plan is 0.31 wrong;
    this says the plan assumes the applicant's hours stay at 40 while the SCM,
    given the education it just asked for, puts them at 46 — which is the
    sentence that makes causal invalidity mean something to a reader.
    """
    x = np.asarray(cf.x_original, dtype=float).reshape(1, -1)
    x_cf = np.asarray(cf.x_cf, dtype=float).reshape(1, -1)
    repaired = evaluator.fcm.repair(x, x_cf)[0]
    gaps = np.abs(repaired - x_cf[0]) / evaluator.span
    pretty = evaluator.cost_model.pretty
    lines: list[str] = []
    for j in np.argsort(gaps)[::-1][:limit]:
        if gaps[int(j)] <= float(CONFIG.CAUSAL_TOLERANCE):
            break
        name = evaluator.feature_names[int(j)]
        lines.append(f"{name}: the plan assumes {pretty(name, x_cf[0, int(j)])}, "
                     f"the SCM says {pretty(name, repaired[int(j)])} "
                     f"(gap {gaps[int(j)]:.2f})")
    return lines


def _plan(evaluator: metrics.Evaluator,
          cf: Counterfactual) -> tuple[list[str], list[str]]:
    """What the method *asked for*, and what merely *followed* from it.

    Two lists because they are two different things to a reader, and conflating
    them is the single easiest way to misread CAC: its plans move features it
    never asked anybody to move, because the SCM propagated them.  Charging those
    to the applicant would make the only method that reports its consequences
    look like the one that demands the most.
    """
    names = evaluator.feature_names
    x = np.asarray(cf.x_original, dtype=float)
    x_cf = np.asarray(cf.x_cf, dtype=float)
    asked = set(cf.changed_features)
    acted = np.array([n in asked for n in names], dtype=bool)
    pretty = evaluator.cost_model.pretty
    followed = [f"{name}: {pretty(name, x[j])} -> {pretty(name, x_cf[j])}"
                for j, name in enumerate(names)
                if not acted[j] and not np.isclose(x[j], x_cf[j], atol=1e-9)]
    return evaluator.cost_model.explain(x, np.where(acted, x_cf, x)), followed


def _methods(joined: pd.DataFrame) -> list[str]:
    """Every method that ran, in reading order rather than alphabetical order."""
    present = set(joined["method"].astype(str))
    ordered = [m for m in METHOD_ORDER if m in present]
    return ordered + sorted(present - set(ordered))


def _original(results: dict[str, dict[int, list[Counterfactual]]],
              record: int) -> np.ndarray | None:
    """The applicant's own record, from whichever method managed to answer.

    Every method was handed the same row of the same test split, so any of them
    will do; they are asked in turn only because a method that crashed on this
    record stored an empty list.
    """
    for per_record in results.values():
        for cf in per_record.get(record, ()):
            return np.asarray(cf.x_original, dtype=float)
    return None


def _method_block(evaluator: metrics.Evaluator, results: dict[str, Any],
                  joined: pd.DataFrame, record: int, method: str,
                  prefer: str = "usable") -> list[str]:
    """One method's advice to one person, and what the three audits make of it."""
    row = _chosen(joined, record, method, prefer)
    if row is None:
        return [f"  {method}: produced nothing for this applicant.", ""]
    cf = results[method][record][int(row["rank"])]
    plan, followed = _plan(evaluator, cf)
    lines = [f"  {method}  ({_verdict(row)})",
             f"    effort {_num(row['cost'])} of the budget — {_num(row['raw_cost'])} "
             f"in §7's own units — over {int(row['n_actions'])} action(s); the black "
             f"box then says P(accept) = {_num(row.get('proba_target'))}, the SCM's "
             f"repaired world {_num(row.get('proba_repaired'))}"]
    lines += [f"      do: {line}" for line in plan] or ["      do: nothing"]
    if followed:
        lines += [f"      and the SCM says this follows: {line}" for line in followed]
    for line in _side_effects(evaluator, cf):
        lines.append(f"      but {line}")
    blocked = str(row.get("blocked_features") or "")
    if blocked:
        lines.append(f"      refused by §7: {blocked}")
    if not bool(row["valid"]):
        lines.append("      -> the black box does not change its mind, so there is "
                     "nothing here to follow.")
    elif not bool(row["flip_survives"]):
        lines.append("      -> the flip does not survive the SCM's consequences; "
                     "carrying this out would not produce the accepted world.")
    elif not bool(row["actionable"]):
        lines.append("      -> a description of a better applicant, not advice this "
                     "one can act on.")
    else:
        reached = "      -> usable: it flips, survives the SCM's consequences, and "
        tail = "a person could do it."
        if not bool(row.get("promise_kept", True)):
            tail += "  (The proposed state strays from the replayed one — see the "
            tail += "side effects above; usable, but low fidelity.)"
        lines.append(reached + tail)
    return lines + [""]


def _narrative(evaluator: metrics.Evaluator, results: dict[str, Any],
               joined: pd.DataFrame, case: Case, record: int, rank: int,
               total: int) -> str:
    """One applicant, every method's advice, and the denominator it stands for."""
    dataset = evaluator.dataset
    label = _DATASET_LABELS.get(dataset, dataset.title())
    x = _original(results, record)
    head = [
        "=" * 84,
        f"{label} — {case.name} — test record {record}",
        f"  {case.question}",
        f"  Shown because: {case.caption}; {total} of the query set qualify, this "
        f"is #{rank + 1}.",
        "=" * 84,
    ]
    if x is None:
        return "\n".join(head + ["  No method produced anything for this record.", ""])
    proba = float(evaluator.model.predict_proba(x.reshape(1, -1))[0, 1])
    head += [
        f"  The black box scores this applicant at P(accept) = {proba:.3f} against a "
        f"threshold of {float(CONFIG.DECISION_THRESHOLD):.2f}, so the answer is no.",
        "  What the file says about them:",
        _profile(evaluator, x),
        "",
        "  What each method tells them to do:",
        "",
    ]
    body: list[str] = []
    for method in _methods(joined):
        body += _method_block(evaluator, results, joined, record, method,
                              case.prefer)
    return "\n".join(head + body)


def _case_rows(evaluator: metrics.Evaluator, results: dict[str, Any],
               joined: pd.DataFrame, case: Case, record: int) -> list[dict[str, Any]]:
    """The narrative's contents as data, so a reader can check the prose.

    Every claim a narrative makes is a column here: the three flags, the effort,
    the actions in §7's own units, the features the SCM moved that the plan did
    not.  A case study whose numbers live only inside a paragraph cannot be
    audited, and this project's whole argument is that unauditable advice is the
    problem.
    """
    rows: list[dict[str, Any]] = []
    for method in _methods(joined):
        row = _chosen(joined, record, method, case.prefer)
        if row is None:
            rows.append({"dataset": evaluator.dataset, "case": case.name,
                         "record": record, "method": method, "answered": False})
            continue
        cf = results[method][record][int(row["rank"])]
        plan, followed = _plan(evaluator, cf)
        rows.append({
            "dataset": evaluator.dataset,
            "case": case.name,
            "record": record,
            "method": method,
            "answered": True,
            "rank": int(row["rank"]),
            "valid": bool(row["valid"]),
            "causal": bool(row["flip_survives"]),
            "actionable": bool(row["actionable"]),
            "usable": bool(row["usable"]),
            "cost": float(row["cost"]),
            "raw_cost": float(row.get("raw_cost", float("nan"))),
            "n_actions": int(row["n_actions"]),
            "gap_max": float(row.get("gap_max", float("nan"))),
            "plausibility": float(row.get("plausibility", float("nan"))),
            "advice": " | ".join(plan),
            "consequences": " | ".join(followed),
            "off_features": str(row.get("off_features") or ""),
            "blocked_features": str(row.get("blocked_features") or ""),
        })
    return rows


#: For ``--records``: an applicant the reader asked after, who need not be a case.
_REQUESTED = Case("requested", "An applicant named on the command line.",
                  "asked for by hand", lambda cmp: pd.Series(dtype=int))


def _records(picked: "pd.Series[Any]") -> list[int]:
    """A rule's answer as record ids, refusing to count a row it did not select.

    A ``NaN`` here is not a record and never a missing one: every row of the
    comparison table has an id, so the only way to arrive holding one is a rule
    that widened its own selection.  That is worth a crash rather than a silent
    drop --- it publishes a frequency, and the number would be wrong.
    """
    values = pd.Series(picked).dropna()
    if len(values) != len(picked):
        raise ValueError(f"a case rule returned {len(picked) - len(values)} row(s) "
                         f"with no record id; its selection is wider than its hits")
    return [int(v) for v in values]


def _run_dataset(dataset: str, wanted: Sequence[str],
                 records: Sequence[int] | None) -> dict[str, Any]:
    """Every case on one dataset: the counts, the rows, and the prose.

    The frequency table covers all five cases whatever ``--cases`` says, because
    it is a statement about the query set rather than about what is being read
    out loud; ``--cases`` and ``--records`` choose which narrative gets rendered.
    """
    evaluator, results, joined = _score(dataset)
    comparison = _comparison(_per_record(joined))
    total = int(comparison.shape[0])
    hits = {case.name: _records(case.pick(comparison)) for case in CASES}
    frequency = [{
        "dataset": dataset,
        "case": case.name,
        "question": case.question,
        "n_records": len(hits[case.name]),
        "n_queries": total,
        "share": len(hits[case.name]) / total if total else float("nan"),
        "shown": hits[case.name][0] if hits[case.name] else -1,
    } for case in CASES]

    chosen: list[tuple[Case, int, int, int]] = []
    if records:
        for record in (int(r) for r in records):
            case = next((c for c in CASES if record in hits[c.name]), _REQUESTED)
            picks = hits.get(case.name, [])
            chosen.append((case, record,
                           picks.index(record) if record in picks else 0,
                           len(picks) or 1))
    else:
        for case in CASES:
            if wanted and case.name not in wanted:
                continue
            if not hits[case.name]:
                LOGGER.info("%s/%s: no record qualifies — nothing to narrate",
                            dataset, case.name)
                continue
            chosen.append((case, hits[case.name][0], 0, len(hits[case.name])))

    rows: list[dict[str, Any]] = []
    narratives: list[str] = []
    for case, record, rank, n_hits in chosen:
        narratives.append(_narrative(evaluator, results, joined, case, record,
                                     rank, n_hits))
        rows.extend(_case_rows(evaluator, results, joined, case, record))
        LOGGER.info("%s/%s: narrating record %d of %d that qualify",
                    dataset, case.name, record, n_hits)
    return {"cases": pd.DataFrame(rows), "frequency": pd.DataFrame(frequency),
            "narratives": narratives, "comparison": comparison}


# ---------------------------------------------------------------- what §9 prints

_TICK = {True: r"\checkmark", False: r"$\times$"}

#: LaTeX-hostile characters, in an order that cannot double-escape the backslash.
_ESCAPES = (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
            ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
            ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
            ("—", "---"), ("§", r"\S{}"))


def _tex(text: Any) -> str:
    """Escape a cell by hand, because the ticks in these tables are LaTeX.

    ``save_latex_table(escape=True)`` would turn ``\\checkmark`` into the literal
    string; ``escape=False`` would let ``hours_per_week`` compile as a subscript.
    So the two columns that carry prose are escaped here and the tables are
    written unescaped.
    """
    out = str(text)
    for character, replacement in _ESCAPES:
        out = out.replace(character, replacement)
    return out.replace("->", r"$\rightarrow$")


def _flagship(cases: pd.DataFrame) -> pd.DataFrame:
    """The one case the paper prints in full, chosen by a rule fixed in advance.

    The rule walks a declared order and takes the first pair that *exists*:
    :data:`FLAGSHIP` before the other datasets in :data:`DATASETS` order, and
    within a dataset the cases in :data:`CASES` order, which is the order §9
    argues them in.  Nothing here reads a number, so the table cannot be tuned by
    picking the applicant who flatters CAC most; the only input is whether a case
    has any examples at all, and a case that has none has nothing to print.

    The primary dataset is not guaranteed to supply the contrastive case: on Adult
    every method is usable on every record, so all four cases that accuse a
    baseline are empty there and the walk moves on.  That is worth a warning
    rather than a silent substitution, and the frequency this function's caller
    prints beside the table --- zero out of a hundred on Adult --- is what stops
    the reader from taking the case as typical of the primary dataset.
    """
    if cases.empty:
        return cases
    order = [FLAGSHIP] + [d for d in DATASETS if d != FLAGSHIP]
    available = {(str(d), str(c)) for d, c in zip(cases["dataset"], cases["case"])}
    for dataset in order:
        for case in CASES:
            if (dataset, case.name) in available:
                if (dataset, case.name) != (FLAGSHIP, CASES[0].name):
                    LOGGER.warning("no %s/%s case in this run; the flagship table "
                                   "shows %s/%s instead", FLAGSHIP, CASES[0].name,
                                   dataset, case.name)
                return cases[(cases["dataset"] == dataset)
                             & (cases["case"] == case.name)]
    return cases.iloc[:0]


def _case_study_table(cases: pd.DataFrame) -> pd.DataFrame:
    """One row per method: the three verdicts, the effort, and the advice itself."""
    rows: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        answered = bool(row.get("answered"))
        actions = [a for a in str(row.get("advice") or "").split(" | ") if a]
        rows.append({
            "Method": str(row["method"]),
            "Flips": _TICK[answered and bool(row.get("valid"))],
            "SCM agrees": _TICK[answered and bool(row.get("causal"))],
            "Actionable": _TICK[answered and bool(row.get("actionable"))],
            "Effort": _num(row.get("cost")) if answered else "--",
            "What the applicant is told to do": _tex(_advice(actions)),
        })
    return pd.DataFrame(rows)


def _advice(actions: Sequence[str], width: int = 80, keep: int = 2) -> str:
    """The first actions in full, then a count — a table cell cannot hold a plan.

    Truncating mid-sentence would misquote the advice and dropping the rest
    silently would understate what the applicant is being asked for, so the
    remainder is counted rather than cut: the whole plan is in
    ``results/qualitative_narratives.txt``, which is what the appendix prints.
    Per-action costs come off, because the row already carries the plan's total and
    a cell that spends its width on prices has none left for the advice.
    """
    if not actions:
        return "nothing"
    plain = [_COST_SUFFIX.sub("", action) for action in actions]
    text = textwrap.shorten("; ".join(plain[:keep]), width=width, placeholder=" ...")
    rest = len(plain) - min(keep, len(plain))
    return text if not rest else f"{text} (+{rest} more)"


#: ``explain`` prices each action inline; :func:`_advice` drops that for the table.
_COST_SUFFIX = re.compile(r"\s*\(cost [^)]*\)")


def _frequency_table(frequency: pd.DataFrame) -> pd.DataFrame:
    """How often each case happens, so a narrative is evidence and not an anecdote."""
    rows: list[dict[str, Any]] = []
    for _, row in frequency.iterrows():
        rows.append({
            "Dataset": _DATASET_LABELS.get(str(row["dataset"]),
                                           str(row["dataset"]).title()),
            "Case": _tex(str(row["case"]).replace("_", " ")),
            "What it means": _tex(str(row["question"])),
            "Records": int(row["n_records"]),
            "of": int(row["n_queries"]),
            "Share": _num(row["share"], 2),
        })
    return pd.DataFrame(rows)


def _check(cases: pd.DataFrame, frequency: pd.DataFrame) -> list[str]:
    """Would a reader of the prose and a reader of the CSV agree?

    The narratives are selected by rules over one frame and rendered from another,
    so the two can drift apart in exactly one way: a record that qualified for a
    case no longer looking like that case by the time its row is written.  That
    would put a false sentence in the paper, so it is checked rather than trusted.
    """
    problems: list[str] = []
    if cases.empty:
        return ["no case rendered on any dataset"]
    for (dataset, case), group in cases.groupby(["dataset", "case"], sort=True):
        cac = group[group["method"] == "cac"]
        rival = group[group["method"] == RIVAL]
        record = int(group["record"].iloc[0])
        tag = f"{dataset}/{case} (record {record})"
        if case == "cac_only" and not (len(cac) and bool(cac["usable"].iloc[0])):
            problems.append(f"{tag}: selected as CAC-only but CAC is not usable")
        if case == "cac_only" and len(rival) and bool(rival["usable"].iloc[0]):
            problems.append(f"{tag}: selected as CAC-only but {RIVAL} is usable too")
        if case == "everyone_agrees" and not group["usable"].fillna(False).all():
            problems.append(f"{tag}: selected as unanimous but some method is not "
                            f"usable")
        if case == "immutable_advice" and not group["blocked_features"].astype(str) \
                .str.contains("immutable").any():
            problems.append(f"{tag}: no method's plan touches an immutable feature")
        # The two cases about shipped advice, checked against the rendered rows —
        # the rules read the plan a method would hand over, and these are those
        # plans, so a heading that survives here cannot contradict its own block.
        accused = group[group["method"].isin(LIBRARY) & group["answered"].fillna(False)]
        if case == "cheaper_but_wrong" and len(cac):
            undercut = accused[(accused["cost"] < float(cac["cost"].iloc[0]))
                               & ~accused["usable"].fillna(False)]
            if undercut.empty:
                problems.append(f"{tag}: no baseline's advice is both cheaper than "
                                f"CAC's and unusable")
        if case == "side_effect" and not (
                accused["gap_max"].fillna(0.0) > float(CONFIG.CAUSAL_TOLERANCE)).any():
            problems.append(f"{tag}: no baseline's advice strays from the SCM")
    for _, row in frequency.iterrows():
        if int(row["n_queries"]) == 0:
            problems.append(f"{row['dataset']}: the query set is empty")
    return problems


# ------------------------------------------------------------------------- driver

def _write(merged: dict[str, pd.DataFrame], narratives: Sequence[str]) -> None:
    """Two CSVs, the prose, and the two tables §9 includes."""
    for key, filename in OUTPUTS.items():
        save_dataframe(merged[key], CONFIG.RESULTS_DIR / filename)
        LOGGER.info("Wrote results/%s (%d rows)", filename, merged[key].shape[0])
    path = CONFIG.RESULTS_DIR / "qualitative_narratives.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(narratives), encoding="utf-8")
    LOGGER.info("Wrote results/qualitative_narratives.txt (%d case(s))",
                len(narratives))

    flagship = _flagship(merged["cases"])
    if not flagship.empty:
        dataset = str(flagship["dataset"].iloc[0])
        record = int(flagship["record"].iloc[0])
        case = next((c for c in CASES if c.name == str(flagship["case"].iloc[0])),
                    _REQUESTED)
        # The denominator in the caption, so a reader is never left to assume the
        # printed applicant is typical: on the primary dataset the contrastive
        # cases are empty and the flagship falls through to one where every method
        # agrees, which is true of most of the query set and needs saying.
        freq = merged["frequency"]
        row = freq[(freq["dataset"] == dataset) & (freq["case"] == case.name)]
        share = (f" --- {int(row['n_records'].iloc[0])} of {int(row['n_queries'].iloc[0])} "
                 f"query-set applicants fall in this case" if not row.empty else "")
        save_latex_table(
            _case_study_table(flagship), "case_study",
            caption=(f"One applicant, five methods. {_DATASET_LABELS.get(dataset, dataset)} "
                     f"test record {record}, selected as the case where {case.caption}"
                     f"{share}. "
                     f"\\emph{{Effort}} is the plan's cost as a fraction of the budget "
                     f"of \\S7.2; the advice column gives the plan's first actions and "
                     f"counts the rest --- the appendix prints each plan in full, priced "
                     f"action by action --- and is what the method asks for, not the "
                     f"consequences the SCM adds to it."),
            label="tab:case-study", escape=False, column_format="llccrp{0.42\\linewidth}")
    if not merged["frequency"].empty:
        save_latex_table(
            _frequency_table(merged["frequency"]), "qualitative_cases",
            caption=("How often each case study happens. \\emph{Records} counts the "
                     "query-set records that satisfy the case's rule out of "
                     "\\emph{of}; the narrative in the appendix shows one of them, "
                     "so each case study is a representative rather than an anecdote."),
            label="tab:qualitative-cases", escape=False,
            column_format="llp{0.34\\linewidth}rrr")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Case studies: the advice itself, re-scored from experiment 02.")
    parser.add_argument("-d", "--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS), help="datasets to read (default: all)")
    parser.add_argument("-c", "--cases", nargs="+", default=[],
                        choices=[c.name for c in CASES],
                        help="which narratives to render; the frequency table always "
                             "covers all five, being a fact about the query set")
    parser.add_argument("--records", nargs="+", type=int, default=[],
                        help="narrate these test records instead, whichever case "
                             "they fall in")
    args = parser.parse_args(argv)
    setup_logging("05_qualitative")

    merged: dict[str, list[pd.DataFrame]] = {key: [] for key in OUTPUTS}
    narratives: list[str] = []
    for dataset in args.datasets:
        LOGGER.info("=== %s ===", dataset)
        out = _run_dataset(dataset, args.cases, args.records)
        for key in OUTPUTS:
            merged[key].append(out[key])
        narratives.extend(out["narratives"])

    frames = {key: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())
              for key, parts in merged.items()}
    _write(frames, narratives)
    for problem in _check(frames["cases"], frames["frequency"]):
        LOGGER.warning("CHECK: %s", problem)
    print("\n".join(narratives))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
