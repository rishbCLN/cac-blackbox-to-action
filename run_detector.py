import sys, re, pypdf, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 65)
print("     LOCAL OFFLINE AI DETECTION AUDIT (NO DATA LEAKAGE)     ")
print("=" * 65)
print("[1/3] Loading local neural detector (roberta-base-openai-detector)...")
tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

print("[2/3] Extracting text directly from compiled PDF (main.pdf)...")
reader = pypdf.PdfReader('output/pdf/main.pdf')
full_text = []

# Exclude references & biographies on page 11 (standard Turnitin exclusion)
for i, page in enumerate(reader.pages):
    t = page.extract_text()
    if i == 10:  # Page 11
        if 'REFERENCES' in t:
            t = t[:t.find('REFERENCES')]
    full_text.append(t)

all_text = '\n'.join(full_text)

# Replace common PDF ligatures
ligatures = {
    '\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff',
    '\ufb03': 'ffi', '\ufb04': 'ffl', '–': '-', '—': '--',
    '’': "'", '“': '"', '”': '"'
}
for k, v in ligatures.items():
    all_text = all_text.replace(k, v)

# Strip headers, metadata, and formatting artifacts
all_text = re.sub(r'Kumar et al\.: When Valid Counterfactuals Fail as Actions', '', all_text)
all_text = re.sub(r'VOLUME [–-], 2026 \d+', '', all_text)
all_text = re.sub(r'Manuscript received.*?\n', '', all_text)
all_text = re.sub(r'Digital Object Identifier.*?\n', '', all_text)

lines = all_text.split('\n')
paras = []
curr = []
for line in lines:
    l = line.strip()
    if not l:
        if curr:
            paras.append(' '.join(curr))
            curr = []
    elif any(l.startswith(h) for h in ['I. ', 'II. ', 'III. ', 'IV. ', 'V. ', 'VI. ', 'VII. ', 'A. ', 'B. ', 'C. ', 'D. ', 'E. ', 'TABLE ', 'FIGURE ']):
        if curr:
            paras.append(' '.join(curr))
            curr = []
        curr.append(l)
    else:
        curr.append(l)
if curr:
    paras.append(' '.join(curr))

# Filter to actual body paragraphs (> 25 words, not tables/figures)
clean_paras = []
for p in paras:
    words = p.split()
    if len(words) > 25 and not p.startswith('TABLE ') and not p.startswith('FIGURE ') and not p.startswith('PLACE PHOTO'):
        clean_paras.append(p)

print(f"[3/3] Evaluating {len(clean_paras)} body paragraphs with neural classifier...\n")

results = []
for idx, p in enumerate(clean_paras):
    inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        human_prob = probs[0].item() * 100
        ai_prob = probs[1].item() * 100
        results.append({
            'idx': idx + 1,
            'snippet': p[:75] + "...",
            'words': len(p.split()),
            'human': human_prob,
            'ai': ai_prob,
            'verdict': 'AI' if ai_prob >= 50.0 else 'HUMAN'
        })

print(f"{'#':<3} | {'Words':<5} | {'Human %':<8} | {'AI %':<8} | {'Status':<6} | {'Snippet'}")
print("-" * 80)
ai_flagged_count = 0
total_ai_prob = 0.0

for r in results:
    if r['ai'] >= 50.0:
        ai_flagged_count += 1
    total_ai_prob += r['ai']
    print(f"{r['idx']:<3} | {r['words']:<5} | {r['human']:>6.1f}%  | {r['ai']:>6.1f}%  | {r['verdict']:<6} | {r['snippet']}")

avg_ai = total_ai_prob / len(results) if results else 0
avg_human = 100.0 - avg_ai
doc_ai_pct = (ai_flagged_count / len(results)) * 100.0

print("\n" + "=" * 65)
print("                    FINAL AUDIT REPORT                      ")
print("=" * 65)
print(f"Total Paragraphs Analyzed:        {len(results)}")
print(f"Paragraphs Flagged as AI:         {ai_flagged_count} ({doc_ai_pct:.1f}%)")
print(f"Paragraphs Verified as HUMAN:     {len(results) - ai_flagged_count} ({100.0 - doc_ai_pct:.1f}%)")
print(f"Document Average AI Probability:  {avg_ai:.2f}%")
print(f"Document Human Authenticity:      {avg_human:.2f}%")
print("=" * 65)
