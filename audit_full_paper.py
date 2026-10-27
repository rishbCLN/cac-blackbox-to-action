import re, sys, pypdf, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

print("Loading neural models...")
tok_openai = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
mod_openai = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
mod_openai.eval()

tok_chatgpt = AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')
mod_chatgpt = AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')
mod_chatgpt.eval()

reader = pypdf.PdfReader('output/pdf/main.pdf')
full_text = []
for i, page in enumerate(reader.pages):
    t = page.extract_text()
    if 'REFERENCES' in t:
        t = t[:t.find('REFERENCES')]
    full_text.append(t)

all_text = '\n'.join(full_text)
ligatures = {'\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff', '\ufb03': 'ffi', '\ufb04': 'ffl', '–': '-', '—': '--', '’': "'", '“': '"', '”': '"'}
for k, v in ligatures.items():
    all_text = all_text.replace(k, v)

all_text = re.sub(r'Kumar et al\.: When Valid Counterfactuals Fail as Actions', '', all_text)
all_text = re.sub(r'VOLUME [–-], 2026 \d+', '', all_text)

lines = all_text.split('\n')
paras, curr = [], []
for line in lines:
    l = line.strip()
    if not l:
        if curr:
            paras.append(' '.join(curr))
            curr = []
    elif any(l.startswith(h) for h in ['I. ', 'II. ', 'III. ', 'IV. ', 'V. ', 'VI. ', 'VII. ']):
        if curr:
            paras.append(' '.join(curr))
            curr = []
        curr.append(l)
    else:
        curr.append(l)
if curr:
    paras.append(' '.join(curr))

clean_paras = [p for p in paras if len(p.split()) > 25 and not p.startswith('TABLE ') and not p.startswith('FIGURE ') and not p.startswith('PLACE PHOTO')]

print(f"\nTotal extracted paragraphs: {len(clean_paras)}")
print(f"{'#':<3} | {'Words':<5} | {'OpenAI AI%':<11} | {'ChatGPT AI%':<12} | {'Snippet'}")
print("-" * 80)

flagged_openai = 0
flagged_chatgpt = 0

for idx, p in enumerate(clean_paras):
    clean = ' '.join(p.split())
    in1 = tok_openai(clean, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p1 = torch.softmax(mod_openai(**in1).logits, dim=-1)[0]
    o_ai = p1[0].item() * 100.0

    in2 = tok_chatgpt(clean, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p2 = torch.softmax(mod_chatgpt(**in2).logits, dim=-1)[0]
    c_ai = p2[1].item() * 100.0

    if o_ai >= 50.0: flagged_openai += 1
    if c_ai >= 50.0: flagged_chatgpt += 1

    snip = clean[:45] + '...'
    print(f"{idx+1:<3} | {len(clean.split()):<5} | {o_ai:>9.2f}% | {c_ai:>10.2f}% | {snip}")

print("-" * 80)
print(f"OpenAI Flagged (>=50%): {flagged_openai} / {len(clean_paras)}")
print(f"ChatGPT Flagged (>=50%): {flagged_chatgpt} / {len(clean_paras)}")
