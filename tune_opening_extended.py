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

test("When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision.")
test("When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision. Under regulatory rules like GDPR Article 22, applicants need concrete steps to achieve approval rather than a static score.")
test("When a predictive model denies a loan, algorithmic recourse calculates an actionable set of feature changes to reverse the decision.")
test("When a predictive model denies a loan or credit application, algorithmic recourse calculates a set of feature changes to reverse the decision.")
