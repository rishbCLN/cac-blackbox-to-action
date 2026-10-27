import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

def test(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        print(f"Human: {p[0].item()*100:5.1f}% | AI: {p[1].item()*100:5.1f}% | {text}")

test("We test whether proposed interventions survive downstream causal propagation while respecting physical constraints like non-decreasing age.")
test("To address this, we simulate interventions through structural causal equations, checking whether candidate recommendations survive downstream changes.")
test("Our audit checks whether recommended actions survive causal propagation and obey practical limits like non-decreasing age.")
