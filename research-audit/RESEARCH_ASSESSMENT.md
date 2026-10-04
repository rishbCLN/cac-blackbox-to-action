# CAC research assessment

Assessment date: 4 October 2026, Asia/Kolkata. Target: workshop or student research paper.

Repository: https://github.com/rishbCLN/cac-blackbox-to-action

Audited commit: `5c8e9f370941f721ef2f28f1c50c6388f724b6a1`.

The project is at the repository root; this checkout has no folder named “cse blackbox to action.” The audit covers the library, experiment scripts, tests, data loaders, both manuscript entry points, and the study protocol. Source code was left unchanged. Audit scripts, logs, and separate smoke-test outputs are in this directory.

## Verdict

**Promising workshop/student research project; not ready to submit in its current form.** The repository contains a working research implementation, meaningful separation of generation and evaluation, and considerably more experimental infrastructure than a simple demonstration. However, the evaluation contract, statistical analysis, and data handling need correction before the current empirical claims are dependable.

The most credible publication route is **an empirical audit of counterfactual recourse under explicit causal and actionability assumptions**, with CAC presented as a practical reference implementation. Claiming a fundamentally new intervention-based recourse paradigm is difficult to defend against existing research. Acceptance will depend on the eventual venue and revised evidence; this assessment is not an acceptance prediction.

## What is already valuable

- Modular data, black-box model, structural causal model, cost, search, evaluation, and presentation layers.
- Real DiCE integration, an explicit bounded-grid reference search, action-space population search, and saved per-proposal records.
- Shared query sets; failures normally remain in denominators; record-level success requires at least one jointly usable suggestion rather than combining different successful suggestions.
- Ablation and graph/model/budget robustness infrastructure already exists. The graph perturbation experiment evaluates against a fixed reference graph, which is useful as a sensitivity study.
- SCM limitations and the absence of real user-study responses are disclosed. The synthetic study is described as an analysis demonstration, not collected human evidence.
- Extensive existing tests. Passing these tests establishes implementation consistency, but the independent probes below show that they do not yet establish evaluation validity.

## Executed checks and scope

See [TEST_RESULTS.md](TEST_RESULTS.md) for final counts, commands, environment, and smoke results. Evidence includes a fresh-clone test attempt, prepared-data full test suite with coverage, four XGBoost training runs, a four-dataset/five-method comparison on ten rejected records per dataset, and three targeted research-validity probes.

The comparison uses the default search settings and five requested suggestions, with `--no-tune` models. It is **not** a reproduction of the manuscript's 100-query, five-seed, tuned-model results. Full significance, ablation, robustness, MLP comparisons, and human studies were not rerun. The paper was checked statically; PDF compilation and visual layout were not validated.

## Submission-blocking findings

### 1. The evaluator does not verify the declared action plan

**Evidence: reproduced against a real German Credit CAC proposal.** The original suggestion is valid and usable, costs approximately 0.008605, and replaying its declared actions gives target probability 0.952385. Replacing only `changed_features` with an empty tuple leaves the evaluator's `valid`, `causal_valid`, `actionable`, and `usable` flags true, and changes the charged cost to zero. Replaying the empty action set gives target probability **0.390217**, below the 0.5 acceptance threshold.

The cause is a disagreement between audit inputs: cost and actionability trust the declared feature list, while causal repair infers an intervention on every changed coordinate in the proposed final state. It therefore does not verify whether the declared actions actually produce that state. Under a perturbed SCM this also risks treating incorrectly predicted downstream changes as additional interventions.

Locations: [metrics.py:203](</Users/shashwatkarn/Documents/ChatGPT/PROJECT AI/cac-blackbox-to-action/src/evaluation/metrics.py:203>), [feasibility.py:99](</Users/shashwatkarn/Documents/ChatGPT/PROJECT AI/cac-blackbox-to-action/src/evaluation/feasibility.py:99>), [fcm.py:327](</Users/shashwatkarn/Documents/ChatGPT/PROJECT AI/cac-blackbox-to-action/src/scm/fcm.py:327>), [faithfulness.py:78](</Users/shashwatkarn/Documents/ChatGPT/PROJECT AI/cac-blackbox-to-action/src/evaluation/faithfulness.py:78>). Reproducer: [probe_findings.py](probe_findings.py); numerical output: [probes.json](evidence/probes.json).

This controlled mutation demonstrates a missing evaluator guarantee. It does **not** establish that the unmodified generator intentionally produces empty-action claims or that every reported CAC result is false.

As a separate check, I replayed the actual declared actions of all **200 unmodified CAC suggestions** from the smoke run. Every replay exactly matched its proposed state and remained accepted. Thus the defect is demonstrated in the evaluator's contract; this sample does not show incorrect action declarations from the ordinary CAC generator. See [actual-cac-replay.json](evidence/actual-cac-replay.json).

**Required correction:** represent action names and values explicitly; replay that exact action set through the evaluator's SCM; query the classifier on the replayed state; price and validate precisely the same actions. Reject unexplained proposed-state differences. For a point baseline, use a documented adapter from changed coordinates to actions. Add empty-action, omitted-action, protected-feature, and wrong-SCM regression cases.

### 2. Repeated applicants are treated as independent statistical observations

**Evidence: reproduced using the project's `_paired_tests`.** For 20 applicants, with five discordant outcomes favoring CAC, exact McNemar gives **p = 0.0625**. Duplicating those identical outcomes over five seed labels produces **p = 5.96e-8**, although no new applicants or information were added. The effect remains 0.25 while the bootstrap interval narrows from [0.10, 0.45] to [0.17, 0.34].

`experiments/06_significance.py:195–222` pools `(seed, record)` cells and applies ordinary McNemar and an ordinary paired bootstrap. Rows sharing an applicant are correlated; the deterministic oracle is especially clear. Pairing methods within a cell does not make different cells independent. Holm correction cannot repair this dependence.

**Required correction:** define whether inference targets applicants, search randomness, or training samples. For fixed splits, average each applicant's outcomes over search seeds and bootstrap paired applicant clusters. Report per-seed rates separately. For a binary McNemar test use one prespecified seed per applicant, or another prespecified binary applicant-level outcome with a clear interpretation. Across training splits, use an analysis that respects repeated applicants and split/model dependence. Withdraw and recompute existing significance claims.

### 3. “Causal validity” means different things in the manuscript and headline code

`paper/sections/04_desiderata.tex` defines causal validity as the classifier remaining favorable after action propagation. The headline actually gates on `promise_kept`, which additionally requires the maximum range-normalized difference between the proposed and repaired states to be at most 0.10 (`faithfulness.py:107`; `feasibility.py:241`).

**Reproduced counterexample:** in a known toy SCM `b = 2a`, a proposal `(a,b)=(1,0)` from `(0,0)` replays to `(1,2)`. A classifier accepting `a >= 1` accepts both states. Thus the manuscript's causal-success test passes, but the implementation's promise gate fails because the normalized state gap is 0.20.

This extra criterion can reduce baseline headline rates even when their actions still achieve acceptance. It may be useful as a separate **state-fidelity** metric, but it is not the definition currently written in the paper. The config comment also describes the tolerance in standard deviations although the implementation normalizes by feature ranges. Categorical codes make such a continuous tolerance particularly hard to interpret.

**Observed impact in the smoke run:** removing only the extra fidelity gate changes German Random from 7/10 to 10/10 usable, German Wachter from 4/10 to 7/10, German DiCE from 9/10 to 10/10, and German Oracle from 8/10 to 10/10. CAC stays at 10/10. These small-sample numbers demonstrate that the mismatch materially changes reported comparisons; they are not estimates of the full-study effect. This alternative summary retains the current inferred-action replay and therefore does not itself repair finding 1. See [metric-definition-comparison.csv](evidence/metric-definition-comparison.csv).

**Required correction:** separately name and report proposed-state validity, replayed-action success, action feasibility, and state fidelity. Specify which conjunction is the primary endpoint; align equations, code, tables, and narrative. Recompute headlines and show tolerance sensitivity if fidelity remains a gate.

### 4. The evaluation is not fully isolated from test information

- `src/model/train.py:148–154` evaluates the test AUC and uses it to decide whether to launch a hyperparameter search. The subsequent tuned/default choice uses training CV, which is good, but the overall training procedure still depends on test performance.
- HELOC fills missing values using medians of the full data before the split (`src/data/load_heloc.py:161`). Diabetes does likewise (`load_diabetes.py:259`). The scaler is fitted only on training data; that does not undo earlier imputation leakage.
- Diabetes's `_leaked_values` uses labels from the whole dataset to decide which feature values to erase. The intent is to remove a redistributed dataset's label-dependent imputation, but a rule using held-out labels is not a clean evaluation protocol. Without the original missingness mask it also removes genuine observations sharing those values.

**Required correction:** use an original, provenance-checked diabetes source with original missingness; split before learning imputation, scaling, bounds, or preprocessing decisions; put these transformations inside CV folds for tuning. Choose tuning rules using training/validation data only and reserve test data for final evaluation. Rebuild models and every downstream recourse artifact.

### 5. The novelty claim and comparison set are too weak for the current framing

Optimizing interventions through an SCM is already central to [Karimi et al., FAccT 2021](https://arxiv.org/abs/2002.06278). The current manuscript cites that work, but still characterizes causal validity as a property of the search space as a contribution in a way that overlaps closely. Existing research also explicitly addresses recourse under causal uncertainty: [Karimi et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/02a3c7fb3f489288ae6942498498db20-Abstract.html). Joint attention to recourse constraints is discussed in the [recourse survey](https://arxiv.org/abs/2010.04050), so “the first evaluation to require all three” needs a much more precise supported claim.

The comparisons contain Random, Wachter, DiCE-random, and a project-specific oracle; no prior causal-recourse algorithm directly tests CAC against its closest conceptual competitor. Wachter uses finite differences with L-BFGS-B on a primary tree-based classifier. This is a documented adaptation, but its poor performance does not establish superiority to strong model-agnostic or causal alternatives. The population search is a reasonable engineering choice, not evidence of a uniquely necessary optimizer.

**Required correction:** add at least one prior SCM-based minimal-intervention method with matched information and constraints; compare identical search with/without propagation; add a stronger DiCE variant or other applicable gradient-free reference. Compare at matched query budgets, with success-versus-budget curves. If a method requires gradients, state that access assumption and compare on a compatible model rather than silently implying identical access.

For literature coverage, also consider [adversarial robustness of causal recourse, ICML 2022](https://proceedings.mlr.press/v162/dominguez-olmedo22a.html) and [meaningful versus merely predictive recourse](https://arxiv.org/abs/2107.07853). A July 2026 [preprint on causal performative recourse](https://arxiv.org/abs/2607.28497) studies label effects and adaptation; it is related context, not an interchangeable baseline. This was a targeted novelty check, not an exhaustive systematic review.

### 6. Consistency with an assumed SCM is being stretched into practical usability

The generator and the main evaluator share hand-built graphs and learned equations. Success under that shared model establishes an internal consistency property, not that an individual's intervention will work in the real world. The graph perturbation arm is worthwhile, but its reference “true” SCM is the project's original assumed graph, not independently known truth. Conditional independence tests on observational data cannot establish all causal directions or validate individual counterfactual effects.

FCM equation quality is reported on its fitting data (`src/scm/fcm.py:178` onward), which is not an out-of-sample causal validation. Categorical counterfactual noise is resolved to a particular coupling; uncertainty in that choice and in the learned structural equations is not propagated into headline success.

The feasibility rules, top-three-action budget, and qualitative cost multipliers are modeling assumptions. They do not measure a person's money, time, access, or treatment response. In particular, moving a medical measurement in a model is not evidence of an effective clinical intervention. The existing medical benchmark disclaimer is appropriate and should be reflected in the abstract's scope.

**Required correction:** add a benchmark with a known data-generating SCM, generate with learned/misspecified models, and evaluate through the independently known SCM using declared actions. On observational datasets say “usable under the specified SCM and constraints.” Validate or sensitivity-test costs, action limits, and measurement-to-action assumptions. A real user study is optional for a narrowly framed computational workshop paper; simulated responses cannot support human-benefit claims.

### 7. The advertised reproducible paper build is incomplete

Neither `make all` nor `run_all.sh` runs `experiments/06_significance.py`, yet `paper/sections/09_experiments.tex` inputs its two output tables. The generated results/models/tables are absent from the fresh clone. The fresh test command consequently stops at missing processed-data fixtures. The checked-in CSVs live in `datasets/`, but loaders first search `data/raw/<dataset>/`; the bundled German file additionally has an unsupported schema. Source fallback can change the data variant used, and HELOC/diabetes permit synthetic fallback.

There are two manuscript entry points: a modular `paper/main.tex` and a self-contained root `main.tex` with inline numerical tables and plots. Rebuilding experiment outputs cannot automatically update the root manuscript's numbers. Make's stage stamps also lack source/config/data prerequisites, allowing stale artifacts after edits.

**Required correction:** select one canonical generated paper; include significance in the build graph; make source/config/data changes invalidate artifacts; fail closed on synthetic data in research mode; add source checksums, dataset versions, run manifests, and fixture setup. Test a clean-clone build in CI. Record native macOS OpenMP setup and lock transitive dependencies. Separate an inexpensive smoke target from the full research target.

## Other issues to address before submission

1. **Oracle interpretation:** the oracle optimizes direct point changes on a finite grid, whereas CAC exploits downstream consequences. These are different feasible sets, so Cost/Oracle is a comparison to a grid reference, not a universal causal lower bound or optimality certificate. `metrics.cost_ratio` also does not filter `proven_optimal=False`; capped searches must be labeled as unproven references or excluded from certified-ratio claims.
2. **Uncertainty:** five search seeds over fixed people and fixed trained models describe optimizer variation only. A zero seed-variance interval at 100% does not imply zero uncertainty about new applicants. Report sample counts and applicant-level uncertainty, and separate variation over splits/models.
3. **Ablation interpretation:** actions are legalized independently of the soft feasibility weight. Removing that weight does not remove hard feasibility constraints, so an unchanged rate cannot establish that feasibility is unimportant. Name ablations by the mechanism actually removed.
4. **Cost and access fairness:** baseline constraints and computational budgets should match when testing the value of propagation. Cost comparisons on different sets of successful applicants need an explicitly paired common-success analysis plus unconditional coverage.
5. **Scope of guarantees:** SCM-consistent generation is not a guarantee that feasible recourse exists, that search finds it, or that a perturbed black box accepts it. Include genuinely unreachable cases and allow an honest failure outcome.

## Recommended paper direction

Working title: **“When Valid Counterfactuals Fail as Actions: An Empirical Audit under Explicit Causal Assumptions.”**

Suggested research questions:

1. How often does a valid proposed state correspond to an action whose replay still receives acceptance?
2. How much do hard action constraints versus causal propagation account for differences between methods under equal resources?
3. How sensitive are those conclusions to errors in the assumed SCM, search randomness, training splits, and action costs?

Treat CAC as the reference implementation supporting these questions. Keep the empirical contribution even if it does not dominate a prior causal method. Replace blanket “100% usable everywhere” language with measured rates, denominators, intervals, and the explicit model assumptions. A credible workshop paper does not need universal superiority; it needs a clear question and trustworthy evidence.

The prioritized implementation and experiment schedule is in [GAP_CLOSURE_PLAN.md](GAP_CLOSURE_PLAN.md).
