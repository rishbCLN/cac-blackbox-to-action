# CAC Demo — Interactive / Cinematic UI Upgrade (autonomous one-shot)

**Target:** take the existing six-page Streamlit demo (`app/`) from "functional and honest" to "cinematic, interactive, projector-ready, full-marks." Every microstep choreographed for **speed** and **animation**, plus the **surrounding elements** (wayfinding, cover, takeaway, presenter HUD, empty/error states, microcopy) finalized in detail.

**Mode of work (per user):** fully autonomous, one-shot, no mid-stream questions/checkpoints. Follow the recommended path at every fork. Build → run the validation loop → keep improving until green and polished.

**Hard constraints (unchanged from the project's rules):**
- Streamlit only. **No new runtime deps** (`streamlit-lottie` is NOT installed → do not require it; animation uses CSS + `st.components.v1.html` only).
- **Additive**: keep every public function/signature in `app/components.py`, `app/live.py`, `app/ui/*` so `scratch/smoke_app.py` and `pytest` stay green. The engine still only *queries* `predict_proba`; nothing is trained; no `make all`; no edits to `src/`, `experiments/`, `paper/`, notebooks.
- **No hardcoded result numbers in copy.** Every number the judge sees is read live from `results/*.csv` or the live `Audit`. (The old plan hardcoded "13%"; the regenerated data says **18%** — hardcoding rots. Compute "worst baseline", gaps, deltas at runtime.)

---

## 1. Grounding (verified this session)

**Stack:** Streamlit **1.63.0**, Python 3.14. All modern levers present: `st.fragment`, `st.write_stream`, `st.status`, `st.toast`, `st.segmented_control`, `st.pills`, `st.navigation`/`st.Page`, `width="stretch"`, `st.components.v1.html`.

**App shape (reuse, do not rebuild):**
- `app/app.py` — entrypoint: `st.set_page_config`, `theme.inject_css`, `state.sidebar`, preflight, `_warm_up`, `st.navigation([6 pages])`.
- `app/ui/theme.py` — all copy/badges/CSS: `AUDIT_META`, `THESIS`, `DATASET_BLURB`, `GLOSSARY`, `chips/chip/verdict_badge/big_verdict/audit_legend`, palette tokens (`PALETTE_OK/WARN/...`), `_BASE_CSS`, `_DEMO_CSS`, `inject_css`, `presenter_note`, `is_usable/audit_states`.
- `app/ui/state.py` — session keys (`K_DATASET,K_RECORD,K_DEMO,K_SEED`), `sidebar`, `applicant_picker`, dataset defaulting (`_preferred_default` lands on a dataset with a real hook).
- `app/ui/page_home.py` (hook), `page_gap.py` (thesis bars + funnel), `page_applicant.py` (head-to-head cards + recourse PNG), `page_sandbox.py` (live engine, cascade, `cascade_state` pure helper), `page_map.py` (SCM graph + intervention trace), `page_evidence.py` (significance / robustness / ablation tabs).
- `app/components.py` — read-only CSV/joblib layer + matplotlib `gap_figure`/`funnel_figure`/`recourse_png`/`moves_table`.
- `app/live.py` — `LiveEngine.audit/audit_plan/consequence_frame/plan_frame/feature_meta`, `Audit` dataclass, cached constructors.
- `src/viz` — `figure()`, `method_colour()`, `label_of()`, `PALETTE`, `OK/WARN/NEUTRAL`.
- `.streamlit/config.toml` — warm light theme (`primaryColor=#2B5C6F`, bg `#F7F5F0`), `headless=true`, stats off.
- `scratch/smoke_app.py` — headless harness (`check_components/check_live/check_pages`) that drives all pure logic + matplotlib builders against real artifacts. **This is the test loop.**

**Authoritative current numbers (from `results/headline.csv`, `ablation_headline.csv`) — for choreography, not for hardcoding:**

| dataset | worst baseline (usable) | CAC usable | story role |
|---|---|---|---|
| HELOC | **Wachter 0.18** | 1.00 | the money slide (gap 0.82) |
| German | Wachter 0.71 | 1.00 | strong gap |
| Diabetes | Random 0.93 | 1.00 | moderate |
| Adult | Oracle 0.87 | 1.00 | honest small gap |

Validity ≈ 1.0 for everyone everywhere → the gap is the whole point. Usable range **18%–100%**.

**Correctness note folded into this work (the ablation page is now wrong):** on *usable*, ablation barely moves (`cac`→`cac_no_causal`: adult 1.00→0.95, heloc 1.00→0.96, german/diabetes 1.00→1.00). The real signal is per-plan **`flip_survival`** (adult 1.00→0.60, heloc 1.00→0.79, german 1.00→0.965, diabetes 1.00→0.977) and the state-fidelity gap. `page_evidence._ablation` currently plots `usable` and asserts "usable falls" — now false. The ablation hero must switch to **flip_survival** (with state_fidelity beside it), matching the corrected paper narrative.

---

## 2. Design principles (student + teacher lens)

- **Fast by default, cinematic on cue.** Live interaction (slider edits, dataset/applicant switches) is **instant** (fragment-scoped, no entrance animation). The *reveals* that earn the "wow" (the world's reaction, the gap filling, the ablation collapse) are **presenter-triggered** staged sequences that play **once**, not on every rerun. This is the single most important rule: never re-animate on a keystroke.
- **Student test (every microstep):** can a non-expert, in one glance, see *what changed*, *why it passed/failed*, and *what to look at next*? One dominant number per screen; one sentence of takeaway; the three chips always legible.
- **Teacher test (full marks + wow):** does each beat land a point, build on the last, pre-empt a skeptic, and look polished enough that the mechanism feels real and honest? Is there a clear arc (hook → thesis → proof → you-drive → mechanism → evidence → mic-drop)?
- **Honesty is the credibility.** Keep every honest caption (Adult small gap, imperfect graphs, read-only, no network). Wow comes from clarity and motion, never from overclaiming.
- **Accessibility:** respect `prefers-reduced-motion` (→ force Instant). Keep the ✓/✗ glyphs so pass/fail never relies on color alone. Projector-readable in demo mode.

---

## 3. Motion system (lock these tokens; implement in a new `app/ui/motion.py`)

A single source of truth so motion is consistent and testable. `motion.py` holds **pure string/CSS builders** + one CSS injector; no `st.*` state logic (that lives in `flow.py`).

**Speed presets** (session `K_SPEED`, default `smooth`; `prefers-reduced-motion` or Instant preset → all durations 0, no stagger, reveals jump to final frame):

| preset | entrance | bar fill | reveal stagger | confetti/chime | use |
|---|---|---|---|---|---|
| Instant | 0 | 0 | 0 | off | Q&A, skeptic cross-examination |
| Smooth (default) | 180ms | 420ms | 120ms | off | normal talk |
| Cinematic | 260ms | 650ms | 180ms | on (demo mode) | the big reveals |

**Easing tokens:** standard `cubic-bezier(.4,0,.2,1)`; entrance (decelerate) `cubic-bezier(0,0,.2,1)`; verdict stamp (overshoot) `cubic-bezier(.2,.8,.2,1.2)`.

**Keyframes (inject once via `inject_motion_css(speed, reduced)`):** `kf-fade-up`, `kf-chip-pop`, `kf-shake` (fail), `kf-stamp` (verdict), `kf-grow-x` (bars), `kf-pulse` (metric change), `kf-shimmer` (skeleton), `kf-ripple` (map node). All gated behind a root class `motion-smooth|motion-cinematic|motion-instant` set on a wrapper `<div>` so the preset switches instantly without recomputing per-element CSS. Honor `@media (prefers-reduced-motion: reduce){ * { animation:none!important; transition:none!important } }`.

**HTML/JS component builders (iframe via `st.components.v1.html`, fixed height, inline fonts/colors to match theme):**
- `count_up(value, fmt, color, size, duration)` — animates `0→value` on mount (JS `requestAnimationFrame`); used by every headline metric so numbers "tick up." Instant preset renders the final value with no tween.
- `gap_bars_html(rows, speed)` — the **hero** validity-vs-usable bars as CSS: faint full-width validity track + solid `usable` fill that animates width `0→pct` with the number counting up and the worst-baseline row flagged; CAC row emphasized. (Keep `components.gap_figure` as the "static" fallback behind a toggle.)
- `verdict_stamp_html(usable, speed)` — big USABLE/NOT-USABLE badge that stamps in (overshoot) + soft shadow.
- `confetti_html()` — tiny self-contained canvas burst (no deps); only in Cinematic + demo mode on a USABLE verdict.
- `skeleton(kind)` — shimmer placeholder cards for loading states.

**Animation inventory → where used** (full choreography in §6):
1. Section/page entrance: `kf-fade-up`, once per navigation.
2. F/S/A chips: `kf-chip-pop` staggered; a failing chip adds one `kf-shake` + ✗.
3. Verdict: `kf-stamp`; USABLE optionally confetti.
4. Gap hero: `gap_bars_html` width-grow + `count_up`; gap bracket draws between worst baseline and CAC.
5. Funnel: four bars `kf-grow-x` left→right staggered.
6. Sandbox live edits: **no** entrance anim; probability metric `kf-pulse` + delta on change (fragment-scoped, instant).
7. Sandbox reveal (button "Reveal the world's reaction ▶"): staged `st.empty()` sequence — your point holds → cascade rows drop in (stagger) → off-graph rows shake red → probability `count_up` from `proba_target`→`proba_repaired_target` → verdict re-stamps. ≤1.6s Cinematic; Instant jumps to final. Guarded by a per-(page,record) "already revealed" flag so slider nudges don't replay it.
8. Causal map: on intervention select, chosen nodes `kf-ripple`, a 2-frame fade base→highlighted figure via `st.empty` (full topological wave = stretch goal).
9. Evidence: CI bars grow; significant rows get a check-pop; **ablation** animates the **flip_survival** bar collapsing full→no-causal with the lost delta counting down.
10. Loading: `skeleton` + `st.status("loading the model once…")`.

---

## 4. Speed & performance plan

- **`@st.fragment` for all live/interactive regions** so a widget change reruns only its island, not the whole script:
  - Sandbox: wrap editors + verdict + cascade in one `@st.fragment _sandbox_panel(eng, dataset, record, x, meta)`. Slider moves recompute `eng.audit` + verdict + cascade only. Seed/Reset buttons stay outside and do a full `st.rerun()` (occasional, fine).
  - Gap: dataset switcher + both charts in a fragment so flipping HELOC↔Adult is instant.
  - Head-to-head: applicant picker + cards + figure in a fragment.
  - Evidence: each tab body in a fragment.
- **Warm-up upgrade (`app/app.py:_warm_up`):** pre-instantiate `get_engine(default_dataset)`, pre-generate plans for **all** `live_methods` on the demo record (not just CAC) so every "Seed: X" button is instant, and pre-render the Causal-Map `plot_scm` figure. Keep best-effort/guarded.
- **Cache the map figures:** add a `@st.cache_resource` wrapper (in `components.py`, additive) around `plot_scm(scm, show_costs=True)` and `plot_intervention_paths(scm, picked)` keyed by `(dataset, tuple(picked))`, so the Map page and the reveal don't rebuild matplotlib each rerun.
- **Cache `recourse_png`** results per `(dataset, record)` (already writes a file; ensure the path is memoized so Head-to-head switches are instant).
- **`.streamlit/config.toml` demo hardening (additive):** `[server] runOnSave=false` (no accidental reruns mid-talk), `[client] toolbarMode="minimal"` (hide dev hamburger clutter). Keep theme + `headless`. Document a commented demo-only `showErrorDetails="none"` but leave it **on** by default for the build.
- **Perf budget (acceptance):** after warm-up — sandbox edit → verdict < ~300ms; dataset/applicant switch < ~250ms; page nav instant; reveal ≤1.6s Cinematic / 0 Instant.

---

## 5. Surrounding elements (the "every small aspect" layer)

These are the non-animation pieces that make it a *demo*, not a dashboard. Implement a new `app/ui/flow.py` owning the story spine + presenter tooling + new session keys (`K_STEP="cac_step"`, `K_SPEED="cac_speed"`, `K_PRESENTER="cac_presenter"`, `K_REVEALED="cac_revealed"`).

- **Act 0 — Cover / title (new `page_cover.py`, becomes `default=True`):** project title, the one-line thesis, the four dataset chips, a big **"Start the demo ▶"**, speed-preset + demo-mode + presenter-mode controls, and a readiness row (small green ticks for results/models present via `availability()`). Subtle animated accent background (Cinematic only). This is Act 0 of the stepper.
- **Global story stepper + wayfinding (`flow.stepper()` rendered at top of every page):** a 0–7 progress rail (Cover · Problem · Gap · Head-to-head · Sandbox · Map · Evidence · Takeaway) with big **◀ Back / Next ▶** buttons and "Step n of 7." `K_STEP` persists and drives `st.navigation` via `st.switch_page`. The presenter never needs the sidebar mid-talk. Best-effort keyboard ←/→ via a tiny `components.html` JS that clicks Back/Next (documented as optional; buttons are the contract).
- **Per-page takeaway banner:** each page opens with one bold sentence ("What the judge should take away") + the single dominant number (count-up). Keeps the student oriented.
- **Presenter HUD (upgrade `theme.presenter_note` → `flow.presenter_hud`):** a toggleable panel (session `K_PRESENTER`) with the script line, the exact click list, the expected on-screen number to point at (read live), a per-page **timing budget** (30s/45s/60s…) and a start/stop **stopwatch**, and the next-page cue. Off by default; on for rehearsal/stage.
- **Reset demo:** sidebar + Cover button clears `K_RECORD`, sandbox `_edit_*`, `K_SEED`, `K_STEP`, `K_REVEALED` so every run starts clean.
- **Act 7 — Takeaway / mic-drop (new `page_takeaway.py`):** recap the thesis; an animated four-dataset bar of **CAC usable vs worst baseline** (computed live from headline.csv); the three skeptic answers in one line each (significant · survives wrong graph/model · causal term does the work — link to Evidence); a **"Restart demo ↺"**. The closing beat.
- **Standardized states (small helpers in `flow.py`/`theme.py`):** one friendly card for *loading* (skeleton), *missing artifact* ("Run `make all`…", not a trace), and *figure render error* (wrap every `st.pyplot`/`components.html` so a failure shows a caption, never a stack trace). Audit all six pages to route through these.
- **Microcopy pass:** finalize every button label, caption, metric label; keep vocabulary consistent with `theme.GLOSSARY`/`AUDIT_META`. Persistent tiny legend row (✓ pass · ✗ fail · USABLE) available on data pages.
- **Offline + honesty captions:** keep "read-only / only queried / no network at demo time," Adult-small-gap and imperfect-graph notes.

---

## 6. Page-by-page microstep choreography (Acts 0–7)

For each: **student expectation · teacher wow · beats (with motion + speed) · data**. Beats are the exact on-screen sequence.

### Act 0 — Cover
- **Student:** "What am I about to see?" → title + one-line promise + a single Start button.
- **Teacher:** confident, uncluttered title card; environment is clearly ready.
- **Beats:** (1) title + thesis fade-up; (2) four dataset chips pop in staggered; (3) controls + readiness ticks; (4) **Start ▶** → `flow.goto(1)` → Problem. Cinematic adds a slow background accent.
- **Data:** `availability()`.

### Act 1 — Home / The Problem (hook, 30s)
- **Student:** one real rejected person; the normal tool's advice; *see it fail*; CAC holds.
- **Teacher:** a cinematic contrast that makes "flips-on-paper ≠ usable" visceral in 30s.
- **Beats:** (1) applicant card fades up; (2) "A typical tool says…" advice line (optional typewriter via `write_stream`, Smooth/Cinematic only, Instant = plain); (3) its three chips pop; the **failing** chip shakes red with the plain-language reason card; (4) "CAC says…" card slides in; its three chips pop all-green; verdict stamps **USABLE** (Cinematic+demo → tiny confetti); (5) one-line thesis punch; (6) **Next ▶**. Keep the existing cross-dataset hook borrowing (so Adult still shows a real failing-baseline story) and its caption.
- **Data:** `results/qualitative_cases.csv`; `theme.chips/verdict_badge`; `page_home` logic intact, now driven through `motion`.

### Act 2 — The Gap (thesis, 45s)
- **Student:** "Across everyone, everyone flips the model, but few plans are usable; CAC = 100%."
- **Teacher:** the money slide — bars grow, the gap draws itself, flip HELOC→Adult to prove honesty.
- **Beats:** (1) dataset `segmented_control` (snappier than radio), default HELOC; (2) three metrics count up: CAC usable, worst-baseline usable (named, live), gap; (3) **`gap_bars_html`** hero: faint validity tracks instant, solid usable fills grow `0→pct` staggered, worst-baseline row flagged, a gap bracket draws between it and CAC; (4) funnel grows left→right showing *which* test each baseline fails; (5) presenter flips to Adult → bars re-grow to the honest small gap, caption "we don't hide the easy cases." Whole page in a fragment → switching is instant.
- **Data:** `results/headline.csv`; worst baseline/gap computed live. Keep `funnel_figure` (matplotlib) and `gap_figure` as static fallback.

### Act 3 — Head-to-head (per applicant)
- **Student:** "Pick anyone; see each tool's plan pass/fail side by side."
- **Teacher:** CAC-first cards, each a crisp verdict; the failing baseline's reason is plain and specific.
- **Beats:** (1) applicant picker (curated case or any rejected) in a fragment; (2) recourse PNG fades up (solid=asked, hatched=SCM-added, ring=failed audit) with its legend caption; (3) per-method cards stagger-fade **CAC first**; each: chips pop, verdict stamp, effort/changes, and an expander `moves_table` (clean names — the `:immutable` suffix is already stripped); failing baselines show the red reason card; (4) optional effort bar ("cheaper is often cheaper *because* it's wrong"); (5) CAC narrative block for curated cases.
- **Data:** `components.per_record`, `recourse_png`, `moves_table`, `qualitative_cases.csv`.

### Act 4 — Live Sandbox (the star / real interactivity)
- **Student:** "I drive. I change something; the world reacts; the verdict updates for real."
- **Teacher:** the hidden causal knock-on effects appear on cue; every number is live on the real model; the skeptic can attack it and it holds.
- **Beats:**
  1. Pick rejected applicant; `st.status("loading the model once…")` first time only.
  2. Seed row (`Seed: CAC / DiCE / …`, `Reset`) — instant thanks to warm-up.
  3. **Live editing (fragment, instant, no animation):** sliders/select-sliders per feature; immutable features locked with a lock + tooltip; sensitive features flagged. Every move recomputes `eng.audit`; chips + effort meter update immediately; the P(target) metric `kf-pulse` with its delta.
  4. **The reveal (button "Reveal the world's reaction ▶", plays once per edit):** staged `st.empty()` — (a) "your point holds" if `valid`; (b) `consequence_frame` rows drop in staggered; (c) rows that break the graph (`off_features`) shake red; (d) probability counts `proba_target → proba_repaired_target`; (e) verdict re-stamps using the **existing `cascade_state`** truth (survives / breaks_survives / breaks_lost / no_survive) — caption and chips always agree (that invariant is already smoke-tested; keep it). Instant preset → final frame, no sleep.
  5. "Why not usable" expander: `off_features`, `violations`, over-budget/over-actions in plain sentences; **Download verdict (JSON)** via `Audit.as_dict()`.
  6. Caption: "one `predict_proba` + one SCM repair per change; the model is only queried, never trained."
- **Data:** live only. **Reduced-motion/Instant must skip all `time.sleep`.**

### Act 5 — Causal Map (mechanism)
- **Student:** "This is the world-model; change one thing, see what it drags."
- **Teacher:** the DAG + a visible ripple from an intervention; honesty about imperfect graphs, handed to Evidence.
- **Beats:** (1) SCM figure fades up (cached), role legend; (2) multiselect of `scm.manipulable` → chosen nodes ripple, a base→highlighted 2-frame fade shows downstream `plot_intervention_paths`; "Forced to move: …" lists descendants; (3) **"Try these in the sandbox →"** seeds the same features (existing `set_seed_features`); (4) honesty panel: `scm_validation_summary.csv` with the "we report imperfections" caption. Full topological wave = stretch.
- **Data:** `get_scm`, `plot_graphs`, `scm_validation_summary.csv`.

### Act 6 — Is this real? (skeptic killer)
- **Student:** "Not luck, not a lucky graph/model, and the causal part is what matters."
- **Teacher:** three tabs that each answer an attack; the ablation is the thesis isolated.
- **Beats:** `st.tabs` (fragment per tab):
  - **Significance:** CI bars grow; paired McNemar/Holm table; significant rows get a check-pop. Reads whatever `significance_aggregate.csv`/`significance_tests.csv` currently hold; **degrade gracefully** per-row if a cell is missing (a regeneration may be mid-flight — show the friendly caption, never crash).
  - **Robustness:** keep graph-perturbation + model-transfer + the k and budget sweeps (all four sub-arms); bars use shared `method_colour`; captions in plain words.
  - **Ablation (corrected hero):** animate **`flip_survival`** collapsing `cac → cac_no_causal` (adult 1.00→0.60, heloc 1.00→0.79, german 1.00→0.965, diabetes 1.00→0.977) with the lost delta counting; show `state_fidelity` beside it; keep `usable` only as a secondary row. Caption: "turn off causal scoring → plans stop *surviving the world model* (flip survival drops) → that isolated drop is the thesis." Fix the stale "usable falls" copy and the `full >= nocausal` smoke assertion so it targets `flip_survival`.
- **Data:** the significance/robustness/ablation CSVs.

### Act 7 — Takeaway (mic-drop)
- **Student:** "So what?" → one screen that recaps and closes.
- **Teacher:** memorable, honest, confident; one restart click.
- **Beats:** (1) thesis fades up; (2) four-dataset CAC-vs-worst-baseline bars grow (live); (3) three skeptic answers pop in one line each with links to Evidence tabs; (4) **Restart ↺** → `flow.reset_demo()` → Cover.

---

## 7. New / changed files (all additive)

**New:**
- `app/ui/motion.py` — tokens, `inject_motion_css(speed, reduced)`, pure HTML builders (`count_up`, `gap_bars_html`, `verdict_stamp_html`, `confetti_html`, `skeleton`). Pure + import-safe headless.
- `app/ui/flow.py` — story spine + presenter tooling: step registry (Acts 0–7), `current_step/goto/next_step/back`, `stepper()`, `presenter_hud()`, `speed_control()`, `reset_demo()`, standardized `loading_card/missing_card/safe_pyplot`. Owns new session keys.
- `app/ui/page_cover.py` (Act 0), `app/ui/page_takeaway.py` (Act 7).
- `scratch/smoke_flow.py` — headless driver for the flow/motion pure logic (see §8).
- `tests/test_motion_flow.py` — pytest for pure helpers.

**Changed (additive edits):**
- `app/app.py` — register 8 pages (Cover default → … → Takeaway); `motion.inject_motion_css`; render `flow.stepper()`; upgrade `_warm_up` (all methods + map figure).
- `app/ui/theme.py` — add animated variants of chips/verdict that call `motion`; keep old functions. Add keyframe CSS hook. Fix ablation-adjacent copy if any lives here.
- `app/ui/state.py` — add speed/presenter/step controls to the sidebar; wire `reset_demo`.
- `app/ui/page_home|gap|applicant|sandbox|map|evidence.py` — add the choreography above; wrap interactive regions in `@st.fragment`; route states through `flow` helpers. `page_evidence.py` ablation → flip_survival.
- `app/components.py` — add cached `plot_scm`/`plot_intervention_paths` wrappers + memoized `recourse_png`; (optionally) a `gap_rows(headline, dataset)` helper feeding `gap_bars_html`. Keep `gap_figure`/`funnel_figure`.
- `.streamlit/config.toml` — `runOnSave=false`, `toolbarMode="minimal"` (keep theme/headless).

---

## 8. Validation loop (the autonomous "test in a loop")

Keep all logic that can be tested without a browser **pure** (motion tokens, flow step order/guards, speed-preset resolution, reduced-motion resolution, count-up formatting, cascade staging order, gap-row/worst-baseline computation, ablation flip_survival selection). Run this loop after every change; iterate until green, then polish:

```powershell
python scratch/smoke_app.py ; if ($?) { python scratch/smoke_flow.py ; if ($?) { python -m pytest -q } }
```

- **`scratch/smoke_app.py` (extend):** import `motion`/`flow`; assert the step registry enumerates Acts 0–7 and each maps to a real render fn; assert `inject_motion_css` returns CSS containing each keyframe for each preset; assert reduced/Instant yields zero-duration CSS; assert HTML builders return strings for representative inputs; assert `count_up`/`gap_bars_html` embed the live value; keep the existing component/live/page checks green.
- **`scratch/smoke_flow.py` (new):** drive `flow` headlessly — `goto/next/back` clamp to [0,7]; `reset_demo` clears the right keys; `speed_control` resolves presets; `presenter_hud` reads a live number without error; `safe_pyplot` swallows a raising builder into a caption.
- **`tests/test_motion_flow.py` (new):** unit tests for the pure helpers above (fast, no Streamlit runtime; `st.*` no-op without a script run). Include: ablation hero selects `flip_survival` (and the corrected `cac.flip_survival >= cac_no_causal.flip_survival` monotonicity), worst-baseline computed from live headline rows, speed-preset table, reduced-motion → Instant.
- **Keep green:** full `pytest -q` (currently 320 pass / 1 skip) and `smoke_app.py`.
- **Manual projector pass (final, once loop is green):** `python -m streamlit run app/app.py` → run the golden path Cover→…→Takeaway in demo mode at projector resolution; verify each reveal plays once, Instant preset truly instant, sandbox edits <~300ms, no stack traces, all four datasets, missing-artifact pages show friendly cards.

---

## 9. Risks & mitigations

- **Re-animation on every rerun (the classic Streamlit trap).** → Entrance animations only on first render per navigation; reveals are button-triggered and guarded by `K_REVEALED`; live edits never animate. Fragments keep reruns local.
- **`time.sleep` blocking the session.** → Only inside presenter-triggered reveals, ≤1.6s, Cinematic/Smooth only; Instant/reduced-motion skip entirely.
- **iframe components (count-up/gap-bars/confetti) don't inherit theme.** → Inline fonts/colors from the palette; fixed heights; provide matplotlib fallbacks (`gap_figure`) behind a "static" toggle.
- **Fragment pitfalls (widgets/state scope).** → Keep each fragment self-contained; buttons that change seeds live outside and do a full `st.rerun()`.
- **Breaking the headless harness / contracts.** → Additive only; run the loop (§8) after each change; preserve the `cascade_state` invariant and `moves_table` clean-name invariant.
- **Significance data mid-regeneration.** → Evidence significance tab degrades per-row; never blocks the demo. (Out of scope to regenerate here.)
- **Number drift.** → No hardcoded result numbers; everything read live.
- **Motion overload / tackiness.** → Confetti only Cinematic+demo+USABLE; subtle durations; Instant always available for a serious skeptic.

---

## 10. Out of scope
- No new training, no new model queries beyond `predict_proba`, no `make all`.
- No new web framework, no new pip deps (lottie optional-only, not required).
- No edits to `src/`, `experiments/`, `paper/`, `main.tex`, notebooks, or significance regeneration.

---

## 11. Ordered task list (execute top-to-bottom; run §8 loop after each numbered block)

1. **Motion foundation.** Create `app/ui/motion.py` (tokens, `inject_motion_css`, `count_up`, `gap_bars_html`, `verdict_stamp_html`, `confetti_html`, `skeleton`); honor `prefers-reduced-motion`. Add keyframes. Loop.
2. **Flow spine.** Create `app/ui/flow.py` (step registry Acts 0–7, `stepper`, `goto/next/back`, `speed_control`, `presenter_hud`, `reset_demo`, `loading_card/missing_card/safe_pyplot`) + new session keys. Loop (`smoke_flow.py`, `test_motion_flow.py`).
3. **Shell wiring.** Edit `app/app.py`: 8 pages (Cover default → Takeaway), `inject_motion_css`, `stepper()`, upgraded `_warm_up`. Add `.streamlit/config.toml` demo hardening. Loop + first manual boot.
4. **Cover + Takeaway.** `page_cover.py` (Act 0) and `page_takeaway.py` (Act 7) with their beats (readiness ticks, live four-dataset recap, Restart). Loop.
5. **Speed everywhere.** Add `@st.fragment` to sandbox/gap/head-to-head/evidence; add cached `plot_scm`/`plot_intervention_paths` + memoized `recourse_png` in `components.py`. Loop; verify perf budget.
6. **Act 1 Home** choreography (contrast reveal, chip pop, fail shake, verdict stamp, thesis punch). Loop.
7. **Act 2 Gap** choreography (`gap_bars_html` hero, count-up metrics, funnel grow, instant dataset flip, live worst-baseline/gap). Loop.
8. **Act 3 Head-to-head** choreography (staggered CAC-first cards, PNG fade, reason cards). Loop.
9. **Act 4 Sandbox** choreography (instant fragment edits + the staged "Reveal the world's reaction" sequence honoring `cascade_state`; Instant skips sleeps). Loop + targeted manual (HELOC seed oracle→perturb).
10. **Act 5 Map** choreography (ripple + base→highlighted fade + sandbox handoff). Loop.
11. **Act 6 Evidence** — CI/robustness animations **and the ablation→flip_survival correction** (fix figure, copy, and smoke assertion). Loop.
12. **Surrounding polish.** Presenter HUD + stopwatch + timing budgets; per-page takeaway banners; standardized loading/missing/error states across all pages; microcopy pass; persistent legend; keep honesty/offline captions; optional keyboard ←/→. Loop.
13. **Final manual projector pass** (golden path, all datasets, demo mode, Instant vs Cinematic, perf, no traces). Fix anything, re-loop until clean.

**Handoff:** this plan requires source edits (new `app/ui/motion.py`, `flow.py`, two page modules, edits across `app/`, `.streamlit/config.toml`, new scratch/test files) — build it with an implementation-capable agent, following the task order; each block leaves the app runnable and demo-able.
