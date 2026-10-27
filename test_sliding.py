import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

# Let's test the sentences of Version 1 individually and in pairs
sents = [
    "Flipping a machine learning model's prediction does not mean an applicant received an actionable plan.",
    "Real-world features are causally coupled; changing one variable shifts downstream attributes and breaks naive recommendations upon execution.",
    "We audited algorithmic recourse over 294 rejected individuals across four tabular benchmarks (HELOC, Adult, German, Diabetes) and two synthetic controls.",
    "Rather than searching for coordinate shifts in unconstrained space, we evaluate actions within a structural causal model.",
    "This decouples nominal proposal validity (V) from causal replay validity (R), physical actionability (A), and manifold fidelity (F).",
    "Our central audit criterion is joint usability (U = R and A), requiring candidate plans to survive both structural replay and feasibility bounds.",
    "On credit risk (HELOC), Wachter gradient search achieves near-perfect nominal validity, yet joint usability drops to 18% because secondary credit features shift during replay.",
    "Causal Actionable Counterfactuals (CAC) solve this by searching directly over admissible action sets, reaching 100% usability at default seed.",
    "Across five random seeds, optimizer convergence failures reduce diabetes usability to 85% (with a single-run low of 55%), while refitted causal equations degrade usability to 38% on Adult and 52% on HELOC.",
    "Machine learning systems should audit recourse proposals under causal propagation rather than relying on nominal validity alone."
]

print("=== INDIVIDUAL SENTENCE SCORES ===")
for i, s in enumerate(sents):
    inputs = tokenizer(s, return_tensors='pt', truncation=True)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        print(f"S{i+1}: Human: {p[0].item()*100:5.1f}% | AI: {p[1].item()*100:5.1f}% | {s[:60]}...")

print("\n=== TWO-SENTENCE SLIDING WINDOWS (TURNITIN STYLE) ===")
for i in range(len(sents) - 1):
    chunk = sents[i] + " " + sents[i+1]
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        print(f"Window {i+1}-{i+2}: Human: {p[0].item()*100:5.1f}% | AI: {p[1].item()*100:5.1f}%")
