# Sandbox cascade accuracy: 4-state messaging

## Context / what we did so far

Working in `cac-blackbox-to-action/`, on the live Streamlit demo (uncommitted work
on `app/app.py`, `app/components.py`, new `app/live.py`, `app/ui/`, `scratch/`).

The demo's **Live sandbox** page (`app/ui/page_sandbox.py`) audits a user-typed or
seeded plan live on the real model + fitted SCM and shows a "what the world does in
response" cascade table plus a one-line caption. The caption is chosen by
`cascade_state(audit, frame)` (page_sandbox.py:205).

**Bug found.** `cascade_state` derives "breaks" purely from the display frame's
`breaks causal` column and only distinguishes `no_survive` vs `survives` by
`flip_survives`. It never separates two genuinely different worlds:

- **breaks + flip lost** — a feature drifted past tolerance AND the repaired
  probability fell under threshold (message is correct).
- **breaks + flip survives** — a feature drifted past tolerance but the model still
  says "yes" at the repaired point (`flip_survives` True). The current "breaks"
  message says "the 'yes' no longer holds", which is **factually wrong** here.

Measured on HELOC (6 rejected records x {cac,oracle,dice,wachter} x k=3 seeded plans,
free-typed `audit` path):

| bucket | count |
|---|---|
| breaks + flip SURVIVES (current message wrong) | 29 |
| breaks + flip LOST (current message right) | 12 |
| no_survive (no break, flip lost) | 11 |
| survives | 15 |

So ~29/67 cascade captions can assert a flip is lost when it is not.

The smoke test (`scratch/smoke_app.py:117-131`) only guards the *negative*
direction (`not flip_survives => state != "survives"`) and the `promise_kept =>
survives` case, so it does not catch this.

**Decision (confirmed with user):** implement the 4-state, per-case-accurate
version. `breaks + survives` is still shown as NOT-usable (it is not causally
valid), but worded honestly: the world moves features beyond tolerance so the plan
misdescribes the world you'd actually get, even though the model still says yes at
that repaired point.

## Goal

`cascade_state` returns one of four states derived from the `Audit` object's truth
fields (`causal_valid`, `flip_survives`), not from the display frame's boolean
column, and `_cascade` renders an accurate caption for each. Add a smoke assertion
that locks the mapping.

## Affected files (source edits — hand to an implementation agent)

- `cac-blackbox-to-action/app/ui/page_sandbox.py` — `cascade_state`, `_cascade`.
- `cac-blackbox-to-action/scratch/smoke_app.py` — the cascade block (lines ~117-131).

No changes to `app/live.py`, evaluation code, or results CSVs: the `Audit` already
carries everything needed (`causal_valid`, `flip_survives`, `off_features`,
`proba_target`, `proba_repaired_target`, `has_causal`).

## Design

### State definition (truth-based, order matters)

Replace the frame-driven logic in `cascade_state` (page_sandbox.py:205-220) with a
decision that reads the audit. Keep the `frame` parameter for signature
compatibility with callers/tests, but drive the decision from the audit:

```
def cascade_state(audit, frame) -> str:
    # no SCM -> treat as self-consistent; caller handles the no-causal banner
    if not getattr(audit, "has_causal", False):
        return "survives" if getattr(audit, "flip_survives", True) else "no_survive"
    broke = not bool(getattr(audit, "causal_valid", True))
    survives = bool(getattr(audit, "flip_survives", True))
    if broke and survives:
        return "breaks_survives"
    if broke:
        return "breaks_lost"
    if not survives:
        return "no_survive"
    return "survives"
```

Rationale for using `audit.causal_valid` rather than `frame["breaks causal"].any()`:
`consequence_frame` builds `breaks causal` from `audit.off_features`
(live.py:358), and `causal_valid = gap_max <= tol` (live.py:237) is the same
truth without depending on the display frame being non-empty or present. This also
removes the `frame is None/empty` fragility.

Note the four states are `breaks_lost`, `breaks_survives`, `no_survive`,
`survives` (the old `"breaks"` name is gone).

### Rendering (`_cascade`, page_sandbox.py:223-252)

The table still renders whenever `not frame.empty`. Caption per state:

- **breaks_lost** (error): keep the current "breaks" wording — the world moved a
  feature far from what the plan assumed, so the 'yes' no longer holds once reality
  fills in.
- **breaks_survives** (error, NEW wording): the model still says yes at the
  repaired point, but the world moves highlighted feature(s) beyond what your plan
  assumed, so the plan misdescribes the world you'd actually get — it is not
  causally sound. Include `audit.off_features` (or the highlighted rows) so the
  judge sees which feature drifted. Example:
  "The model would still approve after the world reacts, but your plan is not
  causally sound: it assumed **{off}** would stay put and the graph moves it beyond
  tolerance. The number you'd actually land on is not the one the plan describes."
- **no_survive** (error): keep existing wording (confidence falls from
  `proba_target` to `proba_repaired_target`, back under the threshold).
- **survives**: keep existing empty-frame success message and the within-tolerance
  caption.

Update the `if/elif` chain to branch on all four states; the `state == "breaks"`
branch becomes two branches (`breaks_lost`, `breaks_survives`).

### Verdict consistency check

`_verdict` / `theme.big_verdict` already show NOT USABLE for these plans because the
`causal` chip reads `promise_kept` (False when the gap breaks), so the big banner and
the new caption agree. No change needed there — but confirm during implementation
that a `breaks_survives` plan shows the causal chip red (it must, since
`promise_kept = valid & flip_survives & causal_valid` and `causal_valid` is False).

## Smoke test changes (`scratch/smoke_app.py`)

In the cascade loop (lines 117-131) tighten the invariants to lock all four states:

- `causal_valid and flip_survives` (via `audit`)  => `state == "survives"`.
- `promise_kept`                                    => `state == "survives"`
  (already present; keep).
- `not causal_valid and flip_survives`             => `state == "breaks_survives"`.
- `not causal_valid and not flip_survives`         => `state == "breaks_lost"`.
- `causal_valid and not flip_survives`             => `state == "no_survive"`.
- keep the existing guard `not flip_survives => state != "survives"`.

Drive it from `eng.audit(x, L.seed_from_plan(eng, x, cf))` as today so both the
`breaks_survives` and `breaks_lost` buckets are actually exercised (HELOC produces
both; the loop already iterates all live datasets and their plans).

## Validation

Run from `cac-blackbox-to-action/` (Windows: Git Bash/WSL or set `PYTHONPATH` to the
project root, per README):

1. `python scratch/smoke_app.py` — must print `APP COMPONENTS OK` with the new
   assertions passing across all live datasets (adult, german, heloc, diabetes).
2. `python -m pytest -q` — the suite (301 passing, 1 skipped) must stay green;
   `cascade_state` has no direct unit test today, so no test rename is required, but
   confirm nothing imports the old `"breaks"` literal (grep shows only page_sandbox
   and smoke_app reference these states).
3. Manual (optional): `make app`, open Live sandbox on HELOC, seed a plan that
   breaks-but-survives, confirm the caption no longer claims the yes is lost while
   the verdict still reads NOT USABLE.

## Risks / notes

- Keep `cascade_state`'s `frame` parameter in the signature so existing callers
  (`_cascade`) and the smoke test need no call-site changes beyond the new asserts.
- The change is display-only; it does not touch how `usable`/`promise_kept` are
  computed, so paper headline numbers and `results/*.csv` are unaffected.
- `has_causal == False` datasets: none of the four "breaks" states can occur
  (`causal_valid` is forced True in live.py:243); the guard returns `survives` /
  `no_survive` off `flip_survives`, matching the existing no-SCM banner in
  `_cascade`.

## Out of scope

- No changes to evaluation, SCM, or stored results.
- No new page or figure.
- Reconciling the free-typed `audit` path vs the declared `audit_plan` path — the
  sandbox intentionally uses the action-only `audit` to reveal hidden fallout.
