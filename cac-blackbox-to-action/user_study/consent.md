# Consent and ethics checklist

This document governs the optional human-subjects component of the Causal
Actionable Counterfactuals (CAC) project: a small survey in which respondents
judge the *usefulness* of recourse plans produced by the methods we compare. It is
written to be honest about scope. **No study has been run.** This is the protocol
and the consent form a researcher would use before collecting any data, shipped so
that the human-feasibility claims in the paper can be tested rather than merely
asserted.

## What the study measures

The paper audits whether a recourse plan is *actionable* using a hand-crafted
model of what people can and cannot do (Section 7). That model encodes our
assumptions. The purpose of this study is to check those assumptions against real
people: shown a rejected applicant and a set of plans, do respondents agree that
the plan our audit calls "usable" is the one they would actually be willing and
able to follow? The study measures perceived feasibility, effort, and
trustworthiness of plans. It does **not** collect any decision about a real person,
and it uses only the public benchmark records already in this repository.

## Ethics checklist

This is an informal checklist, appropriate to a low-risk opinion survey. A study
run under an institution should additionally seek that institution's IRB / ethics
board approval before collection.

- [x] **Voluntary.** Participation is entirely optional. No one is required,
  incentivised into coercion, or disadvantaged by declining.
- [x] **Informed.** Every respondent sees the consent text below before any
  question and must actively agree to proceed.
- [x] **Anonymous.** No name, email, IP address, or device identifier is stored.
  Responses carry only a random per-session token. We do not ask for demographic
  detail beyond what is strictly needed for analysis (self-reported familiarity
  with lending or credit decisions, on a coarse scale), and even that is optional.
- [x] **Withdrawable.** A respondent may stop at any time; partial responses are
  discarded, not analysed.
- [x] **No sensitive data.** The survey shows only public benchmark records
  (Adult, German Credit, HELOC, Pima diabetes). It never shows, collects, or asks
  about the respondent's own financial or health information.
- [x] **No deception.** Respondents are told plainly that the plans were generated
  by different automated methods and that the goal is to compare those methods.
  Which plan came from which method is withheld during rating (to avoid bias) and
  disclosed in the debrief.
- [x] **Minimal risk.** The task is reading short hypothetical scenarios and giving
  opinions. There is no foreseeable risk beyond mild boredom.
- [x] **Data retention.** Aggregated, anonymous responses only; retained for the
  lifetime of the research and then deletable on request of the hosting
  institution. No raw response is ever tied to a person.

## Consent form (shown to every respondent)

> **Study:** Judging the usefulness of automated "how to get approved" advice.
>
> You are invited to take part in a short, anonymous research survey. You will be
> shown several hypothetical scenarios in which a person was declined for a loan,
> a credit line, or flagged for a health risk, together with a few different pieces
> of advice on what that person could change. For each, we will ask your opinion on
> how realistic, how difficult, and how trustworthy the advice is.
>
> - Taking part is **voluntary**. You may stop at any time, for any reason, without
>   giving a reason and without any consequence.
> - The survey is **anonymous**. We do not collect your name or any information
>   that could identify you. We store only your answers, under a random code.
> - The scenarios use **public research datasets**. We never ask about your own
>   money, health, or personal circumstances.
> - The survey takes roughly **10–15 minutes**.
> - There are no right or wrong answers; we want your honest judgement.
>
> If you understand the above and agree to take part, select **"I agree"** to
> begin. If not, simply close this page.
>
> [ ] I have read and understood this information, and I agree to take part.

## Debrief (shown on completion)

> Thank you. The pieces of advice you rated were produced by five different
> automated methods for explaining a rejection. One of them, called CAC, is the
> method this research proposes; the others are existing baselines. We are testing
> whether advice that our automated audit judges "usable" is also the advice people
> find realistic and worth following. Your anonymous answers help us check that our
> assumptions about what is feasible match human judgement. If you have questions
> about the research, contact the study team at the address provided by the hosting
> institution.
