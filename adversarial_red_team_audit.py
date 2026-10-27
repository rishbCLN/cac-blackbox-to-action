import sys, os, re, math, statistics, pypdf, torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("             ADVERSARIAL RED-TEAM DEEP VERIFICATION SUITE             ")
print("          (Hunting for Vulnerabilities, Outliers, and AI Tells)       ")
print("=" * 80)

# 1. Extract pure sentences
reader = pypdf.PdfReader('output/pdf/main.pdf')
raw_text_pages = []
for i, page in enumerate(reader.pages):
    t = page.extract_text()
    if i == 10 and 'REFERENCES' in t:
        t = t[:t.find('REFERENCES')]
    raw_text_pages.append(t)

all_text = '\n'.join(raw_text_pages)
ligatures = {'\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff', '\ufb03': 'ffi', '\ufb04': 'ffl', '–': '-', '—': '--', '’': "'", '“': '"', '”': '"'}
for k, v in ligatures.items():
    all_text = all_text.replace(k, v)

# Strip headers and footers
all_text = re.sub(r'Kumar et al\.: When Valid Counterfactuals Fail as Actions', '', all_text)
all_text = re.sub(r'VOLUME [–-], 2026 \d+', '', all_text)
all_text = re.sub(r'Manuscript received.*?\n', '', all_text)
all_text = re.sub(r'Digital Object Identifier.*?\n', '', all_text)

raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', all_text)]
# Filter out tables, figures, metadata
sentences = [s for s in raw_sentences if len(s.split()) >= 6 and not s.startswith('TABLE ') and not s.startswith('FIGURE ') and not s.startswith('PLACE PHOTO') and not s.startswith('0 20 40 60')]

print(f"Total Substantive Sentences Audited: {len(sentences)}")

# -----------------------------------------------------------------------------
# ADVERSARIAL TEST 1: SENTENCE-BY-SENTENCE NEURAL PROFILING
# Hunt down any individual sentence where ChatGPT probability > 10%
# -----------------------------------------------------------------------------
print("\n[ADVERSARIAL 1] Scanning EVERY sentence individually with chatgpt-detector-roberta...")
tok1 = AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')
mod1 = AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')
mod1.eval()

ai_risk_sentences = []
for idx, s in enumerate(sentences):
    inputs = tok1(s, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        logits = mod1(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        # id2label: {0: 'Human', 1: 'ChatGPT'}
        h_prob = probs[0].item() * 100
        ai_prob = probs[1].item() * 100
        ai_risk_sentences.append((ai_prob, h_prob, idx + 1, s))

# Sort by highest AI risk
ai_risk_sentences.sort(key=lambda x: x[0], reverse=True)

print(f"-> Worst-performing sentences (Highest AI probability across the entire paper):")
flagged_high_risk = 0
for ai_prob, h_prob, s_idx, s_text in ai_risk_sentences[:10]:
    if ai_prob > 10.0:
        flagged_high_risk += 1
    status = "ALERT" if ai_prob > 30.0 else ("WATCH" if ai_prob > 10.0 else "SAFE")
    print(f"[{status}] Sent #{s_idx:3d} | AI Prob: {ai_prob:5.1f}% | Human: {h_prob:5.1f}% | \"{s_text[:75]}...\"")

# -----------------------------------------------------------------------------
# ADVERSARIAL TEST 2: LOW-PERPLEXITY (ROBOTIC) SENTENCES (GPT-2)
# AI models generate predictable sequences. Find sentences with PPL < 20.
# -----------------------------------------------------------------------------
print("\n[ADVERSARIAL 2] Scanning for unnaturally low perplexity (robotic) sentences (GPT-2)...")
gpt_tok = GPT2Tokenizer.from_pretrained('gpt2')
gpt_mod = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_mod.eval()

ppl_scores = []
for idx, s in enumerate(sentences):
    enc = gpt_tok(s, return_tensors='pt', truncation=True, max_length=128)
    if enc['input_ids'].size(1) < 4:
        continue
    with torch.no_grad():
        out = gpt_mod(**enc, labels=enc['input_ids'])
        loss = out.loss.item()
        ppl = math.exp(loss)
        ppl_scores.append((ppl, idx + 1, s))

ppl_scores.sort(key=lambda x: x[0]) # lowest perplexity first

print("-> Lowest Perplexity Sentences (Most predictable/generic phrases):")
for ppl, s_idx, s_text in ppl_scores[:8]:
    status = "LOW_PPL" if ppl < 20.0 else "NORMAL"
    print(f"[{status}] Sent #{s_idx:3d} | PPL: {ppl:6.1f} | \"{s_text[:75]}...\"")

# -----------------------------------------------------------------------------
# ADVERSARIAL TEST 3: REPETITIVE OPENER & SYNTACTIC MONOTONY SCAN
# -----------------------------------------------------------------------------
print("\n[ADVERSARIAL 3] Scanning for consecutive repetitive sentence starters...")
repetitive_streaks = []
for i in range(len(sentences) - 2):
    first_w0 = sentences[i].split()[0].lower()
    first_w1 = sentences[i+1].split()[0].lower()
    first_w2 = sentences[i+2].split()[0].lower()
    if first_w0 == first_w1 == first_w2 and first_w0 in ['the', 'we', 'this', 'in', 'as', 'to', 'for']:
        repetitive_streaks.append((i+1, first_w0, [sentences[i][:50], sentences[i+1][:50], sentences[i+2][:50]]))

print(f"-> 3+ Consecutive Identical Starters Found: {len(repetitive_streaks)}")
for s_idx, word, streak in repetitive_streaks:
    print(f"   Streak starting at sentence #{s_idx} ('{word}'):")
    for st in streak:
        print(f"     - \"{st}...\"")

# -----------------------------------------------------------------------------
# ADVERSARIAL TEST 4: SUBTLE AI CLICHÉ & IDIOM INVENTORY
# -----------------------------------------------------------------------------
print("\n[ADVERSARIAL 4] Scanning for 40 subtle secondary AI cliches...")
subtle_cliches = [
    'in today', 'at the forefront', 'wide range of', 'plays a crucial role',
    'serves as a', 'it is worth noting', 'shed light', 'sheds light',
    'pave the way', 'game changer', 'vital role', 'testament', 'cornerstone',
    'intricate dance', 'double-edged sword', 'grapple with', 'harness the power',
    'a testament to', 'a myriad of', 'in essence', 'notably absent',
    'ever-evolving', 'fast-paced', 'beacon of', 'pivotal moment',
    'delve deeper', 'delving into', 'rich tapestry', 'seamless integration',
    'holistic view', 'paramount importance', 'treasure trove', 'spearhead'
]

cliche_hits = {}
for c in subtle_cliches:
    matches = re.findall(r'\b' + re.escape(c) + r'\b', all_text.lower())
    if matches:
        cliche_hits[c] = len(matches)

print(f"-> Subtle Cliche Hits: {len(cliche_hits)} found {cliche_hits if cliche_hits else '(Zero hits - clean!)'}")

# -----------------------------------------------------------------------------
# ADVERSARIAL TEST 5: LATEX COMPILATION OVERFULL BOX AUDIT
# -----------------------------------------------------------------------------
print("\n[ADVERSARIAL 5] Inspecting compilation log for formatting defects...")
with open('output/pdf/main.log', 'r', encoding='latin-1') as f:
    log_content = f.read()

overfull_boxes = re.findall(r'Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)', log_content)
print(f"-> Overfull \\hbox warnings in text paragraphs: {len(overfull_boxes)}")
for pts, l1, l2 in overfull_boxes:
    print(f"   Line {l1}-{l2}: {pts}pt overfull")

unresolved_cites = re.findall(r'Citation `([^`]+)\' on page \d+ undefined', log_content)
unresolved_refs = re.findall(r'Reference `([^`]+)\' on page \d+ undefined', log_content)
print(f"-> Unresolved Citations: {len(unresolved_cites)} {unresolved_cites}")
print(f"-> Unresolved References: {len(unresolved_refs)} {unresolved_refs}")

print("\n" + "=" * 80)
print("                   RED-TEAM ADVERSARIAL AUDIT COMPLETE                ")
print("=" * 80)
