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
        print(f"Human: {p[0].item()*100:5.1f}% | AI: {p[1].item()*100:5.1f}% | {text[:80]}...")

para1 = "When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision. Rather than simply reporting static feature importances, the goal is to provide an applicant with practical interventions that lead to model approval."
test(para1)

para2 = "Most existing recourse algorithms frame this problem as finding a nearest point across the decision boundary. Given a rejected feature vector and a fixed model, standard methods minimize continuous distance to identify an approved state. The core limitation of this approach is coordinate independence: real-world features are bound by physical and economic mechanisms."
test(para2)
