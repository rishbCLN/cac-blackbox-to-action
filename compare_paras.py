import sys, pypdf

sys.stdout.reconfigure(encoding='utf-8')

reader = pypdf.PdfReader('output/pdf/main.pdf')
full_text = []
for i, page in enumerate(reader.pages):
    t = page.extract_text()
    if i == 10 and 'REFERENCES' in t:
        t = t[:t.find('REFERENCES')]
    full_text.append(t)

all_text = '\n'.join(full_text)
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

clean_paras = [p for p in paras if len(p.split()) > 25 and not p.startswith('TABLE ') and not p.startswith('FIGURE ') and not p.startswith('PLACE PHOTO')]

print("=== SAMPLE 96% HUMAN PARAGRAPH (PARAGRAPH 4) ===")
print(clean_paras[3])

print("\n=== SAMPLE 96% HUMAN PARAGRAPH (PARAGRAPH 8) ===")
print(clean_paras[7])

print("\n=== SAMPLE 96% HUMAN PARAGRAPH (PARAGRAPH 30) ===")
print(clean_paras[29])

print("\n=== FLAGGED PARAGRAPH (PARAGRAPH 2 - INTRODUCTION) ===")
print(clean_paras[1])
