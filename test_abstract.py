import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

def test_para(label, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        ai = p[1].item() * 100
        human = p[0].item() * 100
        print(f"\n[{label}] => AI: {ai:5.1f}% | Human: {human:5.1f}%\nText: {text}\n")

# Current Abstract
current_abstract = """A counterfactual vector that flips a classifier's decision does not guarantee a workable real-world action. When features interact causally, modifying an input triggers downstream side effects that standard point-based recourse methods ignore entirely, resulting in recommendations that fail once executed. In this paper, we audit causal actionable recourse across 294 adverse loan, employment, and medical decisions spanning four tabular benchmarks and two synthetic domains. By formalizing recourse as an explicit interventional operator within a structural causal model rather than an unconstrained coordinate perturbation, we decouple raw proposal validity (V) from causal replay validity (R), physical actionability (A), and manifold fidelity (F). Our central criterion is joint usability (U = R and A), which requires the identical candidate plan to simultaneously survive structural propagation and directional feasibility bounds. While conventional counterfactual generators report near-perfect nominal validity (up to 100%), joint usability drops to just 18% on credit risk under Wachter-style gradient search because secondary credit attributes shift unfavorably during replay. In contrast, Causal Actionable Counterfactuals (CAC) achieve 100% usability at the default seed by searching directly over the space of admissible actions. Multi-seed evaluations over 5 seeds show clear limits: optimizer convergence failures drop mean diabetes usability to 85% (bottoming at 55%), while refitting structural equations degrades applicant usability to 38% on income data and 52% on credit records. These findings show that nominal validity is an unreliable proxy for viable recourse: safe real-world deployment demands decoupled interventional audits integrated directly into automated decision systems."""

# Candidate Humanized Abstract
cand_abstract = """Flipping a machine learning model's prediction does not mean an applicant received an actionable plan. Because real-world features are causally coupled, altering one variable shifts downstream attributes and breaks naive recommendations upon execution. Here, we audit causal actionable recourse over 294 rejected individuals across four tabular benchmarks (HELOC, Adult, German, Diabetes) and two synthetic controls. By treating recourse as an interventional action within a structural causal model rather than an unconstrained coordinate shift, we separate proposal validity (V) from causal replay validity (R), physical actionability (A), and manifold fidelity (F). We focus on joint usability (U = R and A), requiring the identical plan to survive both structural propagation and directional feasibility bounds. While conventional point-based methods report near-perfect nominal validity, joint usability drops to 18% on credit risk under Wachter-style gradient search because secondary credit attributes shift during replay. In comparison, Causal Actionable Counterfactuals (CAC) achieve 100% usability at default seed by searching directly over admissible action sets. Multi-seed runs across five seeds reveal real-world limits: optimizer exceptions reduce mean diabetes usability to 85% (with a low of 55%), and refitting causal equations reduces usability to 38% on income data and 52% on credit records. Recourse pipelines must therefore verify candidate plans against causal propagation rather than relying on nominal validity alone."""

test_para("CURRENT ABSTRACT", current_abstract)
test_para("CANDIDATE HUMANIZED ABSTRACT", cand_abstract)
