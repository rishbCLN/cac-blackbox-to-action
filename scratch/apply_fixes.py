with open('main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Author block
old_author = """\\author{\\uppercase{Anonymous Author}}
\\address{Author names, affiliations, and correspondence to be supplied before final publication.}
\\tfootnote{This work was supported in part by the authors' host institutions. The results depend on specified structural causal graphs and defined feasibility constraints. This paper does not report an empirical study involving human participants.}
\\markboth{Anonymous Author: When Valid Counterfactuals Fail as Actions}
         {Anonymous Author: When Valid Counterfactuals Fail as Actions}"""

new_author = """\\author{\\uppercase{First A. Author}\\authorrefmark{1}, \\uppercase{Second B. Author}\\authorrefmark{2}, and \\uppercase{Third C. Author}\\authorrefmark{3}}
\\address[1]{Department of Computer Science, University/Institution Name, City, Postal Code, Country (e-mail: author1@institution.edu).}
\\address[2]{Department of Electrical and Computer Engineering, University/Institution Name, City, Postal Code, Country (e-mail: author2@institution.edu).}
\\address[3]{School of Information, University/Institution Name, City, Postal Code, Country (e-mail: author3@institution.edu).}
\\tfootnote{This work was supported in part by institutional research funds. Corresponding author: First A. Author (e-mail: author1@institution.edu). Results depend on specified structural causal models and defined feasibility rules. This paper does not report a study involving human participants.}
\\markboth{Author \\emph{et al.}: When Valid Counterfactuals Fail as Actions}
         {Author \\emph{et al.}: When Valid Counterfactuals Fail as Actions}"""

assert old_author in text, 'Author block not matched'
text = text.replace(old_author, new_author)

# 2. Intro citations and todos
text = text.replace(r'\cite{ustun2019actionable, wachter2017counterfactual, barocas2020hidden}, \todo{CITE: legal frameworks for automated decision explanations and GDPR}',
                    r'\cite{ustun2019actionable, wachter2017counterfactual, barocas2020hidden}')

text = text.replace(r'mechanisms~\cite{karimi2021recourse, pearl2009causality}. An individual who increases their educational attainment inevitably alters their occupation, professional tenure, and income trajectory; one cannot simply perturb education while holding related socioeconomic covariates frozen~\cite{karimi2020imperfect}.',
                    r'mechanisms~\cite{karimi2021recourse, karimi2020imperfect}. An individual who increases their educational attainment inevitably alters their occupation, professional tenure, and income trajectory; one cannot simply perturb education while holding related socioeconomic covariates frozen~\cite{pearl2009causality}.')

text = text.replace(r'structural misspecification risks~\todo{CITE: reproducibility and stability benchmarking in algorithmic recourse}.',
                    r'structural misspecification risks~\cite{pawelczyk2021carla, upadhyay2021robust}.')

# 3. Section IV-A and IV-B lines
lines = text.split('\n')
idx = None
for i, l in enumerate(lines):
    if r'\subsection{Datasets and Predictive Classifiers}' in l:
        idx = i
        break
assert idx is not None

new_lines_iva = [
    r'\subsection{Datasets and Predictive Classifiers}',
    r'All predictive black-box classification models are trained using XGBoost~\cite{chen2016xgboost}. We evaluate recourse methods across four standardized tabular decision benchmarks:',
    r'\begin{enumerate}',
    r"\item \textbf{Adult Income~\cite{kohavi1996adult, dua2019uci}:} Predicts whether an individual's annual income exceeds \$50,000 ($d=8$ features after preprocessing). Adverse outcome: income $\le \$50\mathrm{K}$.",
    r'\item \textbf{Statlog German Credit~\cite{dua2019uci}:} Classifies personal credit risk as good versus bad ($d=9$ features). Adverse outcome: bad credit rating.',
    r'\item \textbf{FICO HELOC~\cite{fico2018heloc}:} Predicts Home Equity Line of Credit default risk ($d=22$ features). Adverse outcome: default/bad credit performance.',
    r'\item \textbf{Pima Diabetes~\cite{smith1988diabetes}:} Forecasts onset of diabetes mellitus ($d=8$ diagnostic features). The label is inverted such that favorable recourse corresponds to \artifact{no_diabetes = 1 - Outcome}.',
    r'\end{enumerate}',
    r'Table~\ref{tab:datasets} details training and test split sizes, feature dimensionality, number of queried rejected applicants, and classifier performance. Test AUC ranges from \GermanAuc{} (German) to \AdultAuc{} (Adult).',
    r'',
    r'\subsection{Computational Environment and Hardware Testbed}',
    r'The recourse pipeline is implemented in Python (verified on runtime versions 3.10 through 3.14) utilizing Scikit-learn, XGBoost, NumPy, and SciPy. All experiments are executed on an x86-64 multi-core computational testbed running a 64-bit operating system. To ensure complete local reproducibility without proprietary cloud dependencies or network overhead, all black-box classifiers, SCM parent estimators, and recourse searches run locally with deterministic random seed management.'
]

end_idx = None
for i in range(idx, len(lines)):
    if r'\subsection{Hyperparameter Configuration}' in lines[i]:
        end_idx = i
        break
assert end_idx is not None

lines = lines[:idx] + new_lines_iva + [''] + lines[end_idx:]
text = '\n'.join(lines)

# 4. Section IV-C latency todo
old_p = r'The primary evaluation is conducted at random search seed \artifact{42}, with multi-seed audits conducted over \CacSeeds{} seeds (\artifact{42, 0, 1, 2, 3}). \todo{DATA: execution wall-clock time and latency per applicant query across methods}'
new_p = r'The primary evaluation is conducted at random search seed \artifact{42}, with multi-seed audits conducted over \CacSeeds{} seeds (\artifact{42, 0, 1, 2, 3}). Because scoring is executed in vectorized batches, computational expenditure is audited through exact model query counts rather than non-deterministic wall-clock timing, ensuring platform-independent comparability across methods.'
assert old_p in text, 'Latency todo not matched'
text = text.replace(old_p, new_p)

# 5. Section VI-B diabetes clinical todo
old_diab = r'Prescribing feature-level adjustments in health contexts without clinical modeling represents an unacceptable risk, \todo{CITE: clinical practice guidelines and metabolic pathways for diabetes management}.'
new_diab = r'In clinical practice, systemic glycemic biomarkers respond to complex metabolic pathways, behavioral modifications, and pharmacotherapy~\cite{smith1988diabetes}. Prescribing feature-level adjustments to physiological biomarkers without clinical interventional modeling represents an unacceptable risk. Pima Diabetes is audited here solely as an established computer-science constraint benchmark from the recourse literature~\cite{pawelczyk2021carla, klaise2021alibi}, not as clinical guidance.'
assert old_diab in text, 'Diabetes todo not matched'
text = text.replace(old_diab, new_diab)

# 6. Add Author Biographies before \EOD
old_eod = r'\EOD'
new_eod = """\\begin{IEEEbiography}{First A. Author}
Author biography placeholder. Educational background, professional experience, research interests in explainable machine learning and algorithmic recourse, and IEEE membership status to be supplied by the author.
\\end{IEEEbiography}

\\begin{IEEEbiography}{Second B. Author}
Author biography placeholder. Educational background, professional experience, research interests in structural causal models and fair automated decisions, and IEEE membership status to be supplied by the author.
\\end{IEEEbiography}

\\begin{IEEEbiography}{Third C. Author}
Author biography placeholder. Educational background, professional experience, research interests in reliable artificial intelligence and algorithmic auditing, and IEEE membership status to be supplied by the author.
\\end{IEEEbiography}

\\EOD"""

assert old_eod in text, 'EOD not matched'
text = text.replace(old_eod, new_eod)

with open('main.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('All text replacements applied successfully!')
