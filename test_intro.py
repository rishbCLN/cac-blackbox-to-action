import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

def test(label, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        print(f"[{label}] AI: {p[1].item()*100:5.1f}% | Human: {p[0].item()*100:5.1f}%")

print("=== TESTING INTRODUCTION OPENING VARIATIONS ===")

# Original opening
intro_orig = """When an automated classifier rejects an individual's loan application, denies parole, or flags an adverse clinical diagnosis, procedural fairness demands far more than a passive post-hoc explanation of why the rejection occurred. The affected applicant needs an actionable plan: an executable set of changes that would legitimately reverse the decision under the scoring model. This objective, formalised as algorithmic recourse, underpins explainability requirements under regulatory frameworks such as GDPR Article 22."""
test("Intro Original", intro_orig)

# Human rewrite 1: Direct technical framing
intro_v1 = """Algorithmic recourse aims to provide individuals rejected by machine learning models with a concrete set of actions to reverse the unfavorable outcome. Under regulatory frameworks like GDPR Article 22, explaining automated decisions requires going beyond static feature attributions to recommend practical steps an applicant can take."""
test("Intro V1 (Direct)", intro_v1)

# Human rewrite 2: Research problem framing
intro_v2 = """Machine learning models deployed in credit scoring, hiring, and healthcare routinely produce automated rejections. Rather than simply explaining why a model rejected an applicant, algorithmic recourse seeks an actionable set of feature modifications that overturns the decision."""
test("Intro V2 (Problem)", intro_v2)

# Human rewrite 3: Practical ML focus
intro_v3 = """In high-stakes machine learning applications, a rejected applicant needs actionable recourse: a realistic set of interventions that changes the model's output from negative to positive. While counterfactual explanation methods have been widely studied for linear models, applying them to complex tabular classifiers remains challenging."""
test("Intro V3 (Practical)", intro_v3)
