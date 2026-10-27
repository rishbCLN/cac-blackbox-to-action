import sys, os, re, math, statistics, pypdf, torch
from collections import Counter
import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
import textstat
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("     MULTI-DIMENSIONAL DEEP OFFLINE AUDIT SUITE (LEVEL 2)     ")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. LOAD EXTRACTED PROSE FROM COMPILED PDF
# -----------------------------------------------------------------------------
print("\n[PHASE 1/6] Extracting Pure Text from Compiled PDF (output/pdf/main.pdf)...")
reader = pypdf.PdfReader('output/pdf/main.pdf')
raw_pages = []
for i, page in enumerate(reader.pages):
    t = page.extract_text()
    if i == 10 and 'REFERENCES' in t:
        t = t[:t.find('REFERENCES')]
    raw_pages.append(t)

all_text = '\n'.join(raw_pages)
ligatures = {'\ufb01': 'fi', '\ufb02': 'fl', '\ufb00': 'ff', '\ufb03': 'ffi', '\ufb04': 'ffl', '–': '-', '—': '--', '’': "'", '“': '"', '”': '"'}
for k, v in ligatures.items():
    all_text = all_text.replace(k, v)

# Clean out headers/footers
all_text = re.sub(r'Kumar et al\.: When Valid Counterfactuals Fail as Actions', '', all_text)
all_text = re.sub(r'VOLUME [–-], 2026 \d+', '', all_text)
all_text = re.sub(r'Manuscript received.*?\n', '', all_text)
all_text = re.sub(r'Digital Object Identifier.*?\n', '', all_text)

# Sentences
raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', all_text)]
sentences = [s for s in raw_sentences if len(s.split()) >= 4 and not s.startswith('TABLE ') and not s.startswith('FIGURE ') and not s.startswith('PLACE PHOTO')]
total_words = sum(len(s.split()) for s in sentences)
print(f"-> Extracted {len(sentences)} prose sentences ({total_words:,} words total).")

# -----------------------------------------------------------------------------
# 2. TEST SUITE 1: AUTOREGRESSIVE PERPLEXITY & BURSTINESS (GPT-2 ENGINE)
# -----------------------------------------------------------------------------
print("\n[PHASE 2/6] Evaluating Autoregressive Perplexity & Perplexity Burstiness (GPT-2)...")
gpt_tok = GPT2Tokenizer.from_pretrained('gpt2')
gpt_mod = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_mod.eval()

sentence_ppls = []
# Evaluate on sample of sentences across the paper
eval_sentences = sentences[::2]  # sample every second sentence for speed & full-paper spread

for s in eval_sentences:
    enc = gpt_tok(s, return_tensors='pt', truncation=True, max_length=128)
    if enc['input_ids'].size(1) < 4:
        continue
    with torch.no_grad():
        out = gpt_mod(**enc, labels=enc['input_ids'])
        loss = out.loss.item()
        ppl = math.exp(loss)
        # Cap extreme values to prevent arithmetic skew
        sentence_ppls.append(min(ppl, 10000.0))

mean_ppl = statistics.mean(sentence_ppls)
median_ppl = statistics.median(sentence_ppls)
stdev_ppl = statistics.stdev(sentence_ppls)
min_ppl = min(sentence_ppls)
max_ppl = max(sentence_ppls)

# GPTZero classification heuristic:
# Synthetic AI: low mean perplexity (< 35) and low burstiness (stdev < 25)
# Authentic Human: high perplexity spikes on technical terms (stdev > 45, max > 200)
burstiness_ratio = stdev_ppl / mean_ppl

print("-" * 65)
print(f"Sentences Evaluated:            {len(sentence_ppls)}")
print(f"Mean Perplexity (PPL):          {mean_ppl:.2f}")
print(f"Median Perplexity:              {median_ppl:.2f}")
print(f"Perplexity Burstiness (Std Dev):{stdev_ppl:.2f} (Human threshold >= 35.0)")
print(f"Burstiness Coefficient (CV):    {burstiness_ratio:.2f} (Human >= 0.50)")
print(f"Perplexity Dynamic Range:       {min_ppl:.2f} to {max_ppl:.2f}")
print("-" * 65)
if stdev_ppl >= 35.0:
    print(">> GPTZero Perplexity Verdict: AUTHENTIC HUMAN (High burstiness & lexical variation)")
else:
    print(">> GPTZero Perplexity Verdict: BORDERLINE/UNIFORM")

# -----------------------------------------------------------------------------
# 3. TEST SUITE 2: COMPLETE READABILITY & COGNITIVE COMPLEXITY (TEXTSTAT)
# -----------------------------------------------------------------------------
print("\n[PHASE 3/6] Running Complete Textstat Academic Readability Battery...")
sample_prose = ' '.join(sentences[:100]) # 100 prose sentences

fre = textstat.flesch_reading_ease(sample_prose)
fkgl = textstat.flesch_kincaid_grade(sample_prose)
fog = textstat.gunning_fog(sample_prose)
smog = textstat.smog_index(sample_prose)
ari = textstat.automated_readability_index(sample_prose)
coleman = textstat.coleman_liau_index(sample_prose)
linsear = textstat.linsear_write_formula(sample_prose)
dale_chall = textstat.dale_chall_readability_score(sample_prose)
consensus = textstat.text_standard(sample_prose)

print("-" * 65)
print(f"1. Flesch Reading Ease:         {fre:.1f} (Technical journal target: 15 - 35)")
print(f"2. Flesch-Kincaid Grade:        {fkgl:.1f} (Graduate / Doctoral >= 14)")
print(f"3. Gunning Fog Index:           {fog:.1f} (Advanced academic research >= 15)")
print(f"4. SMOG Index:                  {smog:.1f} (Formal scientific prose >= 14)")
print(f"5. Automated Readability (ARI): {ari:.1f} (College/Doctoral graduate >= 14)")
print(f"6. Coleman-Liau Index:          {coleman:.1f} (High technical terminology density)")
print(f"7. Linsear Write Formula:       {linsear:.1f} (Complex sentence structure)")
print(f"8. Dale-Chall Readability:      {dale_chall:.2f} (Specialized domain vocabulary > 9.0)")
print(f"9. Text Standard Consensus:     {consensus}")
print("-" * 65)

# -----------------------------------------------------------------------------
# 4. TEST SUITE 3: ZIPF'S LAW & VOCABULARY POWER-LAW DISTRIBUTION
# -----------------------------------------------------------------------------
print("\n[PHASE 4/6] Auditing Zipf's Law Conformity & Hapax Legomena...")
tokens = re.findall(r'\b[a-z]{2,}\b', all_text.lower())
token_freq = Counter(tokens)

# Ranks vs Frequencies
sorted_freqs = sorted(token_freq.values(), reverse=True)
ranks = np.arange(1, len(sorted_freqs) + 1)
log_ranks = np.log(ranks)
log_freqs = np.log(sorted_freqs)

# Linear regression for Zipf exponent alpha: log(f) = -alpha * log(r) + C
slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
alpha = -slope
# R^2 calculation
corr_matrix = np.corrcoef(log_ranks, log_freqs)
r_squared = corr_matrix[0, 1] ** 2

# Hapax Legomena (words occurring exactly once)
hapax = [w for w, c in token_freq.items() if c == 1]
dis_legomena = [w for w, c in token_freq.items() if c == 2]
hapax_ratio = len(hapax) / len(token_freq)

print("-" * 65)
print(f"Total Word Tokens Analyzed:     {len(tokens):,}")
print(f"Unique Vocabulary Size:         {len(token_freq):,} words")
print(f"Zipf's Law Exponent (Alpha):    {alpha:.3f} (Human natural language = 0.8 to 1.2)")
print(f"Zipf's Law Fit Quality (R^2):   {r_squared:.4f} (Ideal human power-law fit > 0.95)")
print(f"Hapax Legomena (Single Use):    {len(hapax):,} words ({hapax_ratio*100:.1f}% of vocabulary)")
print(f"Dis Legomena (Double Use):      {len(dis_legomena):,} words")
print("-" * 65)
if 0.8 <= alpha <= 1.2 and r_squared > 0.95:
    print(">> Zipf's Law Verdict: AUTHENTIC NATURAL HUMAN LANGUAGE")
else:
    print(">> Zipf's Law Verdict: DEVIATION FROM NATURAL DISTRIBUTION")

# -----------------------------------------------------------------------------
# 5. TEST SUITE 4: N-GRAM DIVERSITY & REPETITION DIAGNOSTICS (DISTINCT-N)
# -----------------------------------------------------------------------------
print("\n[PHASE 5/6] Measuring Distinct-N Lexical Diversity & Repetition...")
unigrams = tokens
bigrams = list(zip(tokens[:-1], tokens[1:]))
trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
fourgrams = list(zip(tokens[:-3], tokens[1:-2], tokens[2:-1], tokens[3:]))

distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0
distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0
distinct_4 = len(set(fourgrams)) / len(fourgrams) if fourgrams else 0

print("-" * 65)
print(f"Distinct-1 (Unigram Variety):   {distinct_1:.3f}")
print(f"Distinct-2 (Bigram Variety):    {distinct_2:.3f}")
print(f"Distinct-3 (Trigram Variety):   {distinct_3:.3f} (High variety > 0.85)")
print(f"Distinct-4 (4-gram Variety):    {distinct_4:.3f} (High variety > 0.90)")
print("-" * 65)

# -----------------------------------------------------------------------------
# 6. TEST SUITE 5: SYNTACTIC VARIETY, POS DISTRIBUTION & EMPIRICAL DENSITY
# -----------------------------------------------------------------------------
print("\n[PHASE 6/6] Auditing Syntactic Voice, Opener Diversity & Empirical Density...")

# Sentence openers POS tagging
openers = []
passive_count = 0
for s in sentences:
    toks = word_tokenize(s)
    if not toks:
        continue
    tags = pos_tag(toks)
    openers.append(tags[0][1]) # tag of first word
    
    # Passive voice check (to be + VBN)
    words_lower = [w.lower() for w in toks]
    has_be = any(b in words_lower for b in ['is', 'are', 'was', 'were', 'been', 'be', 'being'])
    has_vbn = any(tag == 'VBN' for _, tag in tags)
    if has_be and has_vbn:
        passive_count += 1

opener_dist = Counter(openers)
pct_passive = (passive_count / len(sentences)) * 100

# Empirical precision metrics
# Count occurrences of specific numbers, percentages, p-values, tables, figures, citations
num_matches = len(re.findall(r'\b\d+(?:\.\d+)?%?\b', all_text))
cite_matches = len(re.findall(r'\[\d+\]', all_text))
table_ref_matches = len(re.findall(r'\bTable\s+\d+\b', all_text, re.IGNORECASE))
fig_ref_matches = len(re.findall(r'\b(?:Figure|Fig\.)\s+\d+\b', all_text, re.IGNORECASE))
p_val_matches = len(re.findall(r'p\s*[<>=]\s*[\d.]+', all_text, re.IGNORECASE))

density_numbers = (num_matches / total_words) * 1000
density_cites = (cite_matches / total_words) * 1000

print("-" * 65)
print(f"Passive Voice Construction:     {pct_passive:.1f}% (Healthy technical range: 15% - 30%)")
print(f"Unique Sentence Opener POS:     {len(opener_dist)} distinct syntactic structures")
print(f"Top Sentence Opener POS:        {dict(opener_dist.most_common(5))}")
print(f"Specific Quantitative Metrics:  {num_matches} numeric data points ({density_numbers:.1f} per 1k words)")
print(f"Explicit In-Text Citations:     {cite_matches} citations ({density_cites:.1f} per 1k words)")
print(f"Table & Figure Cross-References:{table_ref_matches + fig_ref_matches} explicit pointers")
print(f"Statistical p-value References: {p_val_matches} formal hypothesis pointers")
print("-" * 65)

print("\n" + "=" * 80)
print("             DEEP MULTI-DIMENSIONAL AUDIT COMPLETE             ")
print("=" * 80)
