# Plan to turn CAC into a defensible workshop paper

Based on commit `5c8e9f370941f721ef2f28f1c50c6388f724b6a1`, reviewed 4 October 2026. This is a proposed plan; the audit did not implement these source changes.

**Target:** a computational workshop/student research paper about recourse evaluation under explicit assumptions. Approximate effort: 15–22 focused person-days plus experiment runtime and review. This is an effort estimate, not a promised deadline. A human-subject study or a new theoretical contribution would require a separate schedule.

Do not spend the first week polishing the manuscript or running the current large experiment grid: correct the endpoint and data pipeline first, because those changes invalidate downstream numbers.

## Stage 1 — Repair the evaluation contract (2–3 days; P0)

Files: `src/methods/base.py`, `src/evaluation/metrics.py`, `faithfulness.py`, `feasibility.py`, `src/scm/fcm.py`, `tests/test_metrics.py`, `tests/test_scm.py`.

1. Introduce an explicit action object carrying feature names and intervention values, separate from the predicted resulting state. Keep a deterministic point-to-action adapter for methods that return points.
2. Replay those exact declared actions through the audit SCM. All methods use this same path. Evaluate actionability and cost on those actions, and post-action success on the replayed state.
3. Separate four quantities: proposed-state validity, post-action success, action feasibility, and proposed-versus-replayed state fidelity. Select a primary endpoint before looking at new results. Recommended: action feasibility AND replayed success, while reporting proposed-state validity and fidelity separately. If retaining the old conjunction, define it precisely and justify it.
4. Add adversarial regression cases: empty declaration with changed final state; omitted expensive action; omitted immutable action; incorrect downstream state from a misspecified generator; no-action identity; intervention on a parent and child; no feasible solution.
5. Define categorical fidelity by equality/mismatch rather than distance between arbitrary category codes. If retaining a numerical state-gap gate, name its units and test sensitivity.

**Exit criteria:** the empty-plan probe is rejected; costs cannot be reduced by omitting necessary actions; a known toy SCM's replay exactly matches hand calculations; paper equations and program predicates agree. Existing tests plus the new behavioral regressions pass.

## Stage 2 — Establish clean data and statistical inference (3–4 days; P0)

Files: `src/data/load_*.py`, `src/data/preprocess.py`, `src/model/train.py`, `src/evaluation/significance.py`, `experiments/06_significance.py`.

1. Freeze each public dataset's source, version, row schema, target mapping, and SHA-256 checksum. Replace the modified diabetes copy with an original source preserving missing values. Do not reconstruct missingness from held-out labels.
2. Split raw data before learning preprocessing parameters. Fit imputation, scaling, and empirical bounds on training data; refit within each training CV fold during tuning.
3. Use a fixed tuning policy or training-only model selection. Consult the final test set only after the model and selection rule are fixed.
4. Fail research runs if a loader selects a synthetic fallback. Keep an explicitly labeled synthetic demo mode separately.
5. Choose and document the inferential unit. For fixed split studies, aggregate search seeds within applicant and bootstrap paired applicant clusters. If reporting McNemar, use a prespecified binary outcome per independent applicant, not flattened seed/applicant cells. Report seed variability separately from applicant uncertainty.
6. Add a statistical regression check showing that duplicating identical seed results does not fabricate additional independent evidence.

**Exit criteria:** no preprocessing/tuning decision depends on test labels or test statistics; fresh data hashes match the manifest; changing held-out labels cannot alter fitted preprocessing; repeated-row duplication does not tighten the applicant-level inference; raw per-applicant outcomes remain available for audit.

## Stage 3 — Make experiments reproducible (1–2 days; P0)

Files: `Makefile`, `run_all.sh`, dependency manifests, CI workflow, experiment artifact writers, the two manuscript entry points.

1. Add `prepare`, `test`, `smoke`, `research`, `significance`, and `paper` targets with explicit dependencies. Include experiment 06 in the research build.
2. Record commit, all effective settings, data hashes, split indices, package versions, hardware/thread settings, method parameters, and randomness for every run.
3. Key caches by those inputs; reject incompatible cached models and resumable result rows. Existing significance resume keys omit configuration identity and must not mix runs with different budgets or query sets.
4. Build paper tables/figures and numerical prose from the same artifacts. Choose one canonical manuscript; archive or generate the other automatically.
5. Separate generated smoke results from publication results. Add fixture setup so unit tests do not unexpectedly require the full research pipeline.
6. Document native runtime requirements, including OpenMP on macOS. Review the yanked NumPy pin and make any dependency change only with a validated rebuild.

**Exit criteria:** a clean clone produces tests, smoke outputs, significance outputs, and a complete paper input set with documented commands. Changing a source/config/data input invalidates affected caches. Two identical runs match discrete outcomes and documented numerical tolerances; runtime need not match.

## Stage 4 — Add the evidence missing from the research claim (5–7 days; P1)

Run a small pilot first to estimate compute and failure modes, freeze the protocol, then execute the final experiment matrix.

| Axis | Minimum useful design | What it resolves |
|---|---|---|
| Causal ground truth | Two known-SCM scenarios, including an unreachable-recourse case; train on observational samples and evaluate actions under the known generator | Whether success reflects actual intervention effects within a controlled world |
| Real datasets | At least two well-understood existing datasets; include the others if provenance and assumptions are defensible | Practical relevance without making every dataset a prerequisite |
| Classifiers | XGBoost plus one compatible differentiable model | Whether conclusions depend on the primary classifier/search mismatch |
| Methods | CAC; same search without propagation; one prior causal minimal-intervention method; strong applicable DiCE variant; random reference | Novelty and contribution of propagation versus optimizer strength |
| Constraints | Equal mutability, directions, action counts, and admissible domains for the controlled comparison | Whether the gap merely reflects giving CAC stronger constraints |
| Resources | At least three matched black-box query budgets, logging row evaluations and wall time separately | Whether success is bought by additional computation |
| Randomness | Several training splits and several search seeds, paired methods on the same applicants; actual counts disclosed | Training/sample variability versus search variability |
| Query population | Prefer all eligible held-out applicants where affordable; otherwise a prespecified random sample, up to 100 or more per dataset/split | Prevents undocumented convenient query selection; small datasets may have fewer rejected cases |
| Model uncertainty | Learn/perturb generator SCM; audit through independent known SCM for simulations; several alternative assumed SCMs on real data | Avoids treating the author's reference graph as truth |
| Costs | Alternative plausible weights, action limits 1/2/3, and paired cost among common successes | Dependence on handcrafted effort assumptions |

**Primary outcomes:** action-feasible post-action success, with paired effect sizes and applicant-aware intervals; query/time budget; cost on a clearly stated common-success set. Keep unconditional coverage beside conditional cost. Secondary outcomes: state fidelity, plausibility, sparsity, and group disparities where sample sizes permit reliable estimates.

For population-success claims, report appropriate uncertainty even when every sampled case succeeds. If group analysis is underpowered, say so; do not infer fairness from a small or empty gap.

**Exit criteria:** results include failures and unreachable cases, all methods' access assumptions are explicit, a prior causal method is compared, at least one independent known-SCM audit is complete, and claims survive or are narrowed in response to sensitivity results. Stronger baselines winning is an informative result, not a reason to discard an experiment.

## Stage 5 — Rewrite and independently reproduce (3–4 days; P1)

1. Reframe the contribution as a precise empirical evaluation study unless Stage 4 establishes a genuinely distinct algorithmic result.
2. Write the abstract last. Replace broad “first,” “100% everywhere,” “real person could do it,” and “causally impossible by construction” claims with their demonstrated and explicitly conditional versions.
3. Distinguish consistency with an assumed model from causal identification and real-world effectiveness. Explain why particular variables are actionable and why each assumed direction is credible.
4. Add related work on minimal causal interventions, imperfect causal knowledge, robustness, meaningful recourse, and existing benchmark libraries. Distinguish newly checked preprints from peer-reviewed results.
5. Report the oracle as a bounded-grid point reference; use a causal oracle on small known-SCM problems if claiming causal optimality. Separate capped/unproven reference searches.
6. Retain the user-study protocol as future work, or move the synthetic demonstration to supplementary software. A computational paper does not need invented human evidence. If human benefit is a central claim, conduct a properly designed real study separately.
7. Have a second person reproduce the repository from the documented commit and run a mock review using the checklist below.

**Exit criteria:** every abstract claim maps to an experiment/table, every numerical claim maps to a versioned artifact, all paper inputs exist, venue formatting is checked, and another person can reproduce the main results without undocumented local files.

## Submission decision

**Ready to consider submission only when:**

- The declared-action replay and statistics defects are corrected and their regressions pass.
- Preprocessing and model selection preserve a genuine held-out test set.
- Metrics in the paper exactly match those in the code.
- At least one prior causal baseline and one known-SCM evaluation are included.
- Reported rates, cost comparisons, and uncertainty have correct denominators and units.
- The claimed contribution is distinct, modest, and supported even if CAC is not universally best.
- A clean-clone build generates the reported artifacts and paper inputs.

If time is tight, reduce the number of datasets and robustness variants after fixing correctness. Do not trade correct evaluation or inference for additional polished plots. If the corrected study mainly confirms prior work, submit it as an honest replication/empirical study to a venue accepting that contribution type.
