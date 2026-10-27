import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok_openai = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
mod_openai = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
mod_openai.eval()

tok_chatgpt = AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')
mod_chatgpt = AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')
mod_chatgpt.eval()

def score_text(text):
    clean = ' '.join(text.split())
    # OpenAI: 0 is Fake (AI), 1 is Real (Human)
    inputs1 = tok_openai(clean, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p1 = torch.softmax(mod_openai(**inputs1).logits, dim=-1)[0]
    openai_ai = p1[0].item() * 100.0

    # ChatGPT: 0 is Human, 1 is ChatGPT (AI)
    inputs2 = tok_chatgpt(clean, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p2 = torch.softmax(mod_chatgpt(**inputs2).logits, dim=-1)[0]
    chatgpt_ai = p2[1].item() * 100.0

    return openai_ai, chatgpt_ai

if __name__ == '__main__':
    text = sys.stdin.read().strip()
    if text:
        o, c = score_text(text)
        print(f"OpenAI AI: {o:6.2f}% | ChatGPT AI: {c:6.2f}%")
