# Recourse usefulness questionnaire

This is the instrument for the study described in [`consent.md`](consent.md). It is
paper-and-pencil ready: a researcher can administer it as written, or wire it into
a survey platform. It is designed so its results feed directly into
[`analysis.py`](analysis.py), which expects one CSV row per (respondent, scenario,
plan) rating.

**No responses have been collected.** This file defines the questions; the analysis
script ships with a small synthetic response set so the pipeline runs end to end.

## Design

- **Within-subject.** Each respondent sees several *scenarios*; within a scenario
  they rate several *plans* (one per method), presented in randomised order and
  **unlabelled by method**, so nobody rates "the CAC plan" — they rate a plan.
- **Scenarios** are drawn from the curated cases in
  `results/qualitative_cases.csv`, which already pair a rejected applicant with
  each method's leading plan. The most informative are the contrast cases:
  `cheaper_but_wrong` (a baseline offers a near-free plan that fails once its
  consequences play out; CAC's costs more but works) and `side_effect` (a plan
  that disturbs something it should not).
- **Blinding.** Method identity is stored with the stimulus but hidden from the
  respondent until debrief. Plan order is randomised per respondent.

## Per-plan questions

For each plan shown (phrased for a lay respondent; the applicant's situation and
the plan's actions are shown above the questions):

1. **Feasibility.** *"How realistic is it that a person in this situation could
   actually do what this advice asks?"*
   `1 Not at all realistic … 5 Completely realistic`
2. **Effort.** *"How much effort or difficulty would following this advice take?"*
   `1 Very little … 5 A very large amount`
3. **Trust.** *"If you followed this advice, how confident are you it would
   actually get the decision changed?"*
   `1 Not at all confident … 5 Completely confident`
4. **Willingness.** *"Would you personally be willing to follow this advice?"*
   `1 Definitely not … 5 Definitely yes`
5. **Free text (optional).** *"Anything that struck you as odd, unfair, or
   impossible about this advice?"*

## Per-scenario question

After rating every plan in a scenario, with the plans now shown side by side:

6. **Forced choice.** *"Which single piece of advice would you actually follow?"*
   (choose exactly one of the plans, or "none of them")

## Optional closing questions (once, at the end)

7. **Familiarity (optional).** *"How familiar are you with how lending or credit
   decisions are made?"* `1 Not at all … 5 Very familiar`
8. **General free text (optional).** *"Any overall thoughts on the advice you saw?"*

## Hypotheses the study tests

- **H1 (feasibility tracks the audit).** Plans our audit marks `usable` receive
  higher feasibility (Q1) and trust (Q3) ratings than plans it marks not usable.
- **H2 (causal failure is felt).** In `cheaper_but_wrong` scenarios, the cheap
  baseline plan is rated *no less feasible* than CAC's plan (people cannot see the
  causal flaw from the advice alone), yet loses the forced choice (Q6) once
  outcomes are explained in debrief — motivating why the audit, not the reader,
  must catch causal invalidity.
- **H3 (effort is real).** CAC's plans are rated higher-effort (Q2) than the
  cheapest baseline on Adult, consistent with the measured cost premium — the study
  is expected to confirm the cost is real, not to deny it.

## Response schema (what `analysis.py` reads)

One row per rating, CSV with header:

```
respondent_id,scenario_id,dataset,case,method,plan_rank,feasibility,effort,trust,willingness,chosen
```

- `respondent_id` — random per-session token (never a name).
- `scenario_id` — stable id for the (dataset, case) scenario.
- `method`, `plan_rank`, `dataset`, `case` — provenance of the stimulus (hidden
  from respondent; used only in analysis).
- `feasibility, effort, trust, willingness` — integer 1–5 from Q1–Q4.
- `chosen` — 1 if this plan won the scenario's forced choice (Q6) for this
  respondent, else 0.
