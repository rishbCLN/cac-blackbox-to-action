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
        print(f"Human: {p[0].item()*100:5.1f}% | AI: {p[1].item()*100:5.1f}%")

test("Algorithmic recourse provides individuals rejected by machine learning models with executable recommendations to reverse adverse decisions. Under regulations like GDPR Article 22, explaining an automated loan denial or clinical risk prediction requires more than static feature attributions; applicants need practical steps to achieve approval.")
test("When automated models reject loan applicants or medical patients, explainability frameworks must provide an executable path to approval rather than a static description of the rejection.")
test("Providing actionable recourse to individuals rejected by predictive models is a central goal in trustworthy machine learning.")
