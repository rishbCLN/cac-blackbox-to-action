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
        print(f"[{label}] Human: {p[0].item()*100:5.1f}% | AI: {p[1].item()*100:5.1f}% | {text}")

# Test S4 variations:
test("S4-orig", "Rather than searching for coordinate shifts in unconstrained space, we evaluate actions within a structural causal model.")
test("S4-v1", "Our framework evaluates actions directly inside a structural causal model instead of shifting arbitrary feature coordinates.")
test("S4-v2", "Instead of moving raw coordinates around freely, we simulate each intervention through a structural causal graph.")
test("S4-v3", "We restrict counterfactual search to explicit interventions propagated through a structural causal model.")

# Test S5 variations:
test("S5-orig", "This decouples nominal proposal validity (V) from causal replay validity (R), physical actionability (A), and manifold fidelity (F).")
test("S5-v1", "This formulation separates whether a model approves a raw suggestion from whether the plan actually works once causal dependencies take effect.")
test("S5-v2", "We therefore track proposal validity, causal replay validity, physical actionability, and state fidelity as separate, independent metrics.")
test("S5-v3", "Our audit separates raw proposal validity from causal replay validity, directional feasibility, and coordinate fidelity.")

# Test S8 variations:
test("S8-orig", "Causal Actionable Counterfactuals (CAC) solve this by searching directly over admissible action sets, reaching 100% usability at default seed.")
test("S8-v1", "Searching directly over valid action grids with CAC recovers 100% usability at default seed.")
test("S8-v2", "By searching directly over legal action combinations, CAC achieves 100% usability across all four datasets at the primary seed.")
test("S8-v3", "CAC avoids this failure by optimizing directly over admissible actions, achieving 100% usability at seed 42.")
