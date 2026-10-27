import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

def test(label, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        print(f"[{label}] AI: {p[1].item()*100:5.1f}% | Human: {p[0].item()*100:5.1f}%")

print("=== TESTING KNOWN 100% HUMAN HISTORICAL PAPERS (BEFORE AI EXISTED) ===")

# Alan Turing 1950: Computing Machinery and Intelligence
turing_1950 = """I propose to consider the question, 'Can machines think?' This should begin with definitions of the meaning of the terms 'machine' and 'think'. The definitions might be framed so as to reflect so far as possible the normal use of the words, but this attitude is dangerous. If the meaning of the words 'machine' and 'think' are to be found by examining how they are commonly used it is difficult to escape the conclusion that the meaning and the answer to the question, 'Can machines think?' is to be sought in a statistical survey such as a Gallup poll."""
test("Alan Turing 1950", turing_1950)

# Judea Pearl 2009 (Causality textbook excerpt)
pearl_2009 = """The aim of this chapter is to develop a formal mathematical apparatus for handling causal relationships and counterfactual queries. We start with the simplest possible representation, a system of deterministic structural equations, and show how probabilistic notions can be superimposed on this deterministic foundation. We then define the notion of an intervention as a surgical modification of the structural equations, replacing some equations with constant assignments while leaving the remaining mechanisms intact."""
test("Judea Pearl 2009", pearl_2009)

# IEEE 2015 paper on Machine Learning (Pre-GPT)
ieee_2015 = """In this paper, we propose a multi-objective evolutionary algorithm for feature selection in high-dimensional classification problems. The proposed method balances classification accuracy against feature subset sparsity using a modified NSGA-II framework. We evaluate our approach on several benchmark datasets from the UCI repository, comparing performance against standard filter and wrapper techniques. The results demonstrate that our evolutionary search discovers compact feature subsets without sacrificing predictive performance."""
test("IEEE Paper 2015", ieee_2015)
