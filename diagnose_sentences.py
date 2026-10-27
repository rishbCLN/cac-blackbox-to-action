import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

abstract_sentences = [
    "A counterfactual vector that flips a classifier's decision does not guarantee a workable real-world action.",
    "When features interact causally, modifying an input triggers downstream side effects that standard point-based recourse methods ignore entirely, resulting in recommendations that fail once executed.",
    "In this paper, we audit causal actionable recourse across 294 adverse loan, employment, and medical decisions spanning four tabular benchmarks and two synthetic domains.",
    "By formalizing recourse as an explicit interventional operator within a structural causal model rather than an unconstrained coordinate perturbation, we decouple raw proposal validity from causal replay validity, physical actionability, and manifold fidelity.",
    "Our central criterion is joint usability, which requires the identical candidate plan to simultaneously survive structural propagation and directional feasibility bounds.",
    "While conventional counterfactual generators report near-perfect nominal validity, joint usability drops to just 18% on credit risk under Wachter-style gradient search because secondary credit attributes shift unfavorably during replay.",
    "In contrast, Causal Actionable Counterfactuals achieve 100% usability at the default seed by searching directly over the space of admissible actions.",
    "Multi-seed evaluations over 5 seeds show clear limits: optimizer convergence failures drop mean diabetes usability to 85%, while refitting structural equations degrades applicant usability to 38% on income data and 52% on credit records.",
    "These findings show that nominal validity is an unreliable proxy for viable recourse: safe real-world deployment demands decoupled interventional audits integrated directly into automated decision systems."
]

print("=== SENTENCE-LEVEL AI PROBABILITIES (ABSTRACT) ===")
for s in abstract_sentences:
    inputs = tokenizer(s, return_tensors='pt', truncation=True)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        ai_score = p[1].item() * 100
        print(f"AI: {ai_score:5.1f}% | {s}")
