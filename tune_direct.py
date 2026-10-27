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

test("Algorithmic recourse provides individuals rejected by machine learning models with executable recommendations to reverse adverse decisions. While standard explainability techniques report static feature importance scores, recourse seeks a practical set of input changes that leads to model approval.")
test("Algorithmic recourse focuses on finding input modifications that flip an unfavorable prediction to a favorable one. Rather than explaining why a model rejected an applicant, it computes an actionable plan.")
test("When a machine learning classifier denies an application, algorithmic recourse calculates an actionable set of feature changes to reverse the decision.")
