import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

def test(label, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        ai = p[1].item() * 100
        human = p[0].item() * 100
        print(f"[{label}] AI: {ai:5.1f}% (Human: {human:5.1f}%) | {text}")

print("=== TESTING HUMANIZED SENTENCE VARIATIONS ===")
test("S1-var1", "Machine learning classifiers often accept counterfactual changes that cannot be executed in practice.")
test("S1-var2", "Flipping a machine learning model's prediction does not mean an applicant received an actionable plan.")
test("S1-var3", "Reversing a model prediction via counterfactual explanation often yields instructions that applicants cannot actually carry out.")
test("S1-var4", "Most counterfactual explanation methods find points that reverse a classifier's decision on paper, but fail when individuals try to execute them.")

test("S2-var1", "Because real-world features are causally coupled, altering one variable shifts downstream attributes and breaks naive recommendations upon execution.")
test("S2-var2", "When attributes share physical or institutional dependencies, intervening on an actionable feature sets off downstream shifts that coordinate-based algorithms overlook.")

test("S3-var1", "We evaluate causal actionable recourse across 294 loan, job, and health decisions using four tabular datasets and two synthetic environments.")
test("S3-var2", "Here, we audit causal actionable recourse over 294 rejected individuals across four tabular benchmarks (HELOC, Adult, German, Diabetes) and two synthetic controls.")

test("S9-var1", "Our results indicate that nominal counterfactual validity cannot guarantee viable recourse without direct causal auditing.")
test("S9-var2", "Nominal validity alone is therefore insufficient; recourse pipelines must explicitly verify candidate plans against causal propagation.")
test("S9-var3", "Relying on nominal validity alone leaves decision systems vulnerable to silent recourse failures during actual deployment.")
