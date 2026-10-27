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

test("When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision. Most existing algorithms treat this as a simple distance minimization problem in continuous feature space.")
test("When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision. Standard methods formulate this as finding a nearest point across the decision boundary.")
test("When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision. Given a rejected feature vector, standard algorithms minimize distance to find a nearby point that gets approved.")
