# Exact Turnitin Cyan-Highlighted Text (Report V2 - 30% AI Detected)

Every entry below represents ONLY the text that has an active cyan highlight in the teacher's report. All unhighlighted surrounding text, sentences, and sections have been strictly excluded.

---

## 1. Front Matter & Abstract (Report Page 3 / Paper Page 1)

### Footnote (Affiliations block)
> "All reported findings are conditioned on the specified structural equations and domain constraints; no human subjects participated in this investigation."

### Abstract (Only lines 2–4 are highlighted; opening and remainder are unhighlighted)
> "Real-world variables are causally interlinked; altering one variable affects other attributes downstream and compromises naive suggestions when executed. We evaluated algorithmic recourse for 294 individuals who were denied across four tabular datasets (HELOC, Adult, German, Diabetes) and two synthetic cases."

---

## 2. Section I: Introduction (Report Pages 3–4 / Paper Pages 1–2)

### Paragraph 2 (Split across bottom of Page 3 right column and top of Page 4 left column)
> "If $\widetilde{x}$ is rejected by the classifier, or if it violates directional constraints such as non-decreasing age, the recourse recommendation fails."

### Paragraph 3 (Entire paragraph)
> "The flaws in current recourse benchmarks are concealed by common aggregate reporting shortcuts [10], [11]. First, current toolkits conflate the validity and applicability of a proposal by granting an algorithm credit even when its plan collapses under causal replay. Second, evaluations regularly measure validity, directional feasibility, and data-manifold distance as independent rates across an entire pool of proposals. An algorithm thus receives a high score if candidate A crosses the decision boundary while candidate B meets feasibility rules, even when no single plan enables the applicant to gain approval. Furthermore, standard benchmarks rarely examine algorithmic stability across multiple random seeds or evaluate sensitivity to refitted structural causal models."

### Roadmap Paragraph (Page 4 right column)
> "The rest of this paper is organized as follows: Section II compares point-based and causal recourse; Section III discusses causal replay, the multi-objective problem formulation, and evolutionary action search; Section IV describes our experimental setup and methodology; Section V describes empirical auditing, multi-seed stability analysis, and ablation studies; Section VI discusses the reasons behind unsuccessful recourse and validity threats; and finally, Section VII concludes with methodological lessons learned for the field."

---

## 3. Section II: Related Work (Report Page 4 / Paper Page 2)

### Section II Opening
> "Research in algorithmic recourse generally divides along whether an explanation treats feature coordinates as independent dials or respects the downstream mechanisms linking them. Table 1 compares common paradigms across search entities, replay propagation, constraint structures, and audit protocols."

### Subsection II-A (Paragraph 1)
> "The initial formulation by Wachter et al. [4] frames recourse as unconstrained gradient optimization in feature space, minimizing a continuous distance penalty to flip a prediction. Further, other authors introduced practical considerations: DiCE [6] used determinantal point processes to ensure diversity, Ustun et al. [3] employed integer programming to obtain linear feasibility rules, FACE [12] directed applicants based on empirical graphs, and Van Looveren and Klaise [13] used class prototypes to guide search."

### Subsection II-A (Paragraph 2)
> "However, all such coordinate-based strategies share a primary limitation: the assumption that feature coordinates can be modified individually, independent of causal dependencies. In real-world domains, features remain bound by social and physical relationships. Altering one attribute without accounting for downstream consequences leads to recommendations that fail when implemented."

---

## 4. Section III: Proposed Method & System Design (Report Pages 5–7 / Paper Pages 3–5)

### Section III Opening Transition
> "We now formalize the interventional recourse framework, downstream structural replay mechanisms, decoupled audit metrics, and the genetic action search implemented in CAC."

### Subsection III-B (Pearl Engine Intro & Phases 1–2)
> "To determine how an action alters an applicant’s profile in reality, we simulate Pearl’s counterfactual engine [9] across three sequential phases:"  
> "1) Abduction: Recover the unobserved background factors $u_x = (u_1, \dots, u_d)$ characterizing applicant $x$. For continuous features, we invert fitted equations to compute residual noise $u_j = x_j - \widehat{f}_j(pa_j)$; discrete features preserve conditional probability quantiles."  
> "2) Action: Apply the intervention $\text{do}(X_S = v_S)$, which overwrites structural equations for intervened features $j \in S$ with assigned constant values $v_j$."

### Subsection III-C (Text below Eq 9)
> "Without this strict coupling, an algorithm can easily cheat. It can return one candidate plan that reverses the model decision by violating real-world rules, and another candidate that obeys the rules but remains rejected. Neither plan actually helps the applicant."

### Subsection III-D (Action Effort Text before and after Eq 10)
> "To measure the personal burden of recourse, normalized action effort is formulated as:"  
> "where $w_j$ weights feature importance, $c_j(\cdot, \cdot)$ assesses coordinate distance (linear for continuous attributes, shortest path across graph encodings for ordinals, and fixed replacement costs for categories), and budget $B$ rescales total effort to $[0, 1]$."

### Algorithm 1 Header Box
> "Require: Rejected applicant $x$, black-box $p_+$, SCM $\mathcal{M}$, legal quantile grids $\mathcal{G}$, candidates $k=5$"

### Subsection III-E (Evolutionary Search - Paragraphs 1 & 2)
> "To solve this multi-objective problem, we employ an evolutionary algorithm designed for interventional grids, summarized in Algorithm 1. The algorithm initializes a population of 60 random action vectors mapped onto empirical quantile grids $\mathcal{G}$. In each generation, candidate interventions are evaluated through SCM replay. Non-dominated plans that clear the probability margin and satisfy feasibility constraints are archived."  
> "The search terminates after 200 generations or when progress plateaus after 20 stagnant iterations. From the Pareto archive, the system selects up to $k = 5$ diverse recommendations across different mutable feature subsets. Fig. 3 sketches the execution loop. Discrete quantile search explores non-convex combinatorial spaces reliably without computing unstable numerical gradients over tree splits."

---

## 5. Section IV: Experimental Setup (Report Pages 7–8 / Paper Pages 5–6)

### Section IV Overview
> "To evaluate interventional recourse across distinct data regimes, we benchmark four tabular tasks representing diverse dimensionalities, causal graph densities, and institutional constraints."

### Subsection IV-A (Datasets Intro & Items 1–3)
> "We fit gradient-boosted decision trees using XGBoost [17] across four canonical tabular benchmarks:"  
> "1) Adult Income [18], [19]: A 9-feature socioeconomic demographic dataset where the predictive target is whether annual income exceeds $50,000 (earning $\le \$50\text{K}$ denotes adverse rejection)."  
> "2) Statlog German Credit [19]: A 15-variable financial risk registry evaluating consumer loan default (classification as bad credit denotes adverse rejection)."  
> "3) FICO HELOC [20]: A 12-feature home equity credit assessment dataset tracking financial delinquency (denial of credit line constitutes adverse rejection)."

### Subsection IV-D (Baselines 2–5 & Grid Search Discussion)
> "2) Wachter-Style Optimization: Executes multi-start L-BFGS-B in continuous feature space, estimating model gradients via numerical finite differences."  
> "3) DiCE-Random [6]: Generates sets of diverse counterfactual perturbations using determinantal point processes."  
> "4) Grid Search (‘‘Oracle’’ Reference): Exhaustively traverses allowable quantile changes ordered by increasing normalized effort until exhausting a preset query budget."  
> "5) MINT [8]: Deployed in synthetic control environments where true structural equations are analytically specified, matching its interventional assumptions."  
> "Because Grid Search uses a capped query ceiling, its results reflect a bounded search rather than a guarantee of finding the global optimum. In parallel, our comparisons reflect specific reference implementations; alternative optimizers or penalty schedules could yield different trade-offs."

### Subsection IV-E (Statistical Significance Protocol)
> "For statistical evaluation at default seed 42, we use exact two-sided McNemar tests with Holm corrections on paired applicant outcomes ($\alpha = 0.05$). To check run-to-run variation without parametric assumptions, we compute bootstrap difference distributions across 10,000 applicant clusters at bootstrap seed 42."

---

## 6. Section V: Results (Report Pages 8–10 / Paper Pages 6–8)

### Section V Overview & Subsection V-A Intro
> "Our empirical evaluation analyzes the gap between nominal validity and causal replay, computational query expenditures, multi-seed optimization failures, paired hypothesis tests, objective ablations, and causal graph sensitivity."  
> "Table 3 compiles primary performance metrics across all 294 rejected applicants at seed 42. Fig. 4 illustrates the resulting performance disparities alongside multi-seed spreads."

### Subsection V-D (Hypothesis Testing & Bootstrap Paragraphs)
> "Table 5 reports applicant-level paired significance tests. At default seed 42, exact two-sided McNemar tests with Holm familywise error rate adjustments confirm that CAC significantly outperforms every comparison baseline on HELOC: Wachter ($p < 0.001$), Random ($p < 0.001$), DiCE ($p < 0.001$), and Grid ($p < 0.001$). On German Credit, CAC significantly exceeds Wachter ($p = 2.44 \times 10^{-4}$), while on Adult Income, it significantly outperforms Grid ($p = 9.77 \times 10^{-4}$). Other primary-seed comparisons do not attain significance after Holm adjustment."  
> "Bootstrap distributions resampled over 10,000 applicant clusters capture run-to-run stochasticity that single-seed tests miss. On Diabetes, despite tying DiCE at seed 42, CAC’s bootstrap gap relative to DiCE becomes negative due to the optimizer exceptions at seed 3. Conversely, on Adult, DiCE’s bootstrap confidence band exhibits extreme dispersion ($[+25.0, +25.0]$) driven by its total failure run."

### Subsection V-E (Component Ablation - Effort Penalty Sentence)
> "Removing the action effort term ($\lambda_c = 0$) increases normalized effort across all datasets—for instance, driving Adult cost from 0.1807 to 0.2467—confirming that the effort objective actively steers mutations away from high-friction interventions. In contrast, zeroing the soft feasibility penalty ($\lambda_f = 0$) produces zero change in output, indicating that hard coordinate projection during grid legalization fully guarantees constraint compliance."

### Subsection V-F (Sensitivity Analysis & Synthetic Audits - Entire Subsection)
> "The sensitivity of recourse plans to the fitted structural equation models is shown in Table 7. When compared to the refitted models, CAC’s usability becomes 93.0% for Adult and 97.0% for HELOC, but it remains 100% for German and Diabetes. At the recommendation-level analysis, the plan survivability suffers a lot more, because only 73.0% of Adult recommendations and 79.4% of HELOC recommendations survive refitted replay. Applicant-level usability stays higher simply because an individual needs only one surviving plan among their top $k$ candidates. Because the refitted equations were trained on overlapping data splits, this drop represents a conservative lower bound on real-world sensitivity."  
> "In synthetic Reach and Block worlds (Table 8), whose true structural equations are analytically known, CAC and MINT achieve 100% usability in Reach and 40.0% in Block. In Block, 42 of 100 individuals are provably unreachable due to hard structural bounds. MINT attains lower intervention effort in both environments, proving that CAC does not inherently guarantee minimal-effort recourse."  
> "Finally, Table 9 reports empirical checks on the assumed DAGs. On German and HELOC, numerous edges lack observational orientation support, reminding us that assumed causal graphs represent structural hypotheses rather than infallible laws of nature."

### Subsection V-G (Verification Protocol - Entire Subsection)
> "To ensure reproducibility and safeguard against artifact corruption, all findings were audited against four operational checks:"  
> "1) Fixed Query Alignment: Every method is evaluated on the identical set of 294 rejected test individuals across all benchmarks, eliminating sample selection bias."  
> "2) Causal State Replay: All candidate interventions are passed through non-linear SCM equations $T_{\mathcal{M}}(x, a)$ to simulate true downstream profile updates prior to classifier scoring."  
> "3) Coupled Joint Usability: The audit marks an applicant as served if and only if the identical plan satisfies both classifier acceptance ($R = 1$) and feasibility bounds ($A = 1$)."  
> "4) Software Test Coverage: The execution pipeline cleared 386 automated unit and property tests with 1 documented skip, guaranteeing algorithmic and statistical consistency across all generated assets."  
> "Section VI analyzes the mechanistic roots of these empirical outcomes and details limitations of the benchmark pipeline."

---

## 7. Section VI: Discussion & Tables (Report Pages 11–12 / Paper Pages 9–10)

### Table 9 Caption
> "TABLE 9. Observational validation of assumed causal DAGs. ‘‘Actions’’ denotes the count of mutable variables; ‘‘Supported’’ counts structural edges corroborated by observational orientation heuristics; ‘‘CI upheld’’ reports the fraction of non-rejected conditional independence tests. These tests gauge empirical compatibility rather than establishing definitive physical causality."

### Subsection VI-A (Sentence Split across bottom of Page 11 and top of Page 12)
> "the movement in coordinates coincides with the actual causal propagation process. Thus, heuristics work, but this case requires causal modeling when features are densely coupled."

### Subsection VI-B (Methodological and Data Preprocessing Limitations - Intro & Items 1–2)
> "A rigorous code-level audit revealed three data-hygiene practices common across recourse research that require scrutiny:"  
> "1) Test-Gated Hyperparameter Decisions: In conventional pipelines, test-set AUC was monitored to determine whether to trigger hyperparameter tuning. While inner cross-validation folds were utilized during optimization, allowing test-split performance to dictate whether search is performed introduces subtle data leakage across splits."  
> "2) Pre-Split Feature Imputation: Missing values across HELOC and Diabetes were imputed using global column medians computed prior to train-test partitioning. Beyond this, diabetes cleaning leveraged class-conditional patterns to flag sentinel missingness, potentially allowing label information to leak across evaluation boundaries."
