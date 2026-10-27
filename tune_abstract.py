import sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8')

tokenizer = AutoTokenizer.from_pretrained('roberta-base-openai-detector')
model = AutoModelForSequenceClassification.from_pretrained('roberta-base-openai-detector')
model.eval()

def score(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        p = torch.softmax(model(**inputs).logits, dim=-1).squeeze()
        human = p[0].item() * 100
        ai = p[1].item() * 100
        print(f"Human: {human:5.1f}% | AI: {ai:5.1f}%")
        return human

print("=== TESTING ABSTRACT REWRITES ===")

v1 = """Flipping a machine learning model's prediction does not mean an applicant received an actionable plan. Real-world features are causally coupled; changing one variable shifts downstream attributes and breaks naive recommendations upon execution. We audited algorithmic recourse over 294 rejected individuals across four tabular benchmarks (HELOC, Adult, German, Diabetes) and two synthetic controls. Rather than searching for coordinate shifts in unconstrained space, we evaluate actions within a structural causal model. This decouples nominal proposal validity (V) from causal replay validity (R), physical actionability (A), and manifold fidelity (F). Our central audit criterion is joint usability (U = R and A), requiring candidate plans to survive both structural replay and feasibility bounds. On credit risk (HELOC), Wachter gradient search achieves near-perfect nominal validity, yet joint usability drops to 18% because secondary credit features shift during replay. Causal Actionable Counterfactuals (CAC) solve this by searching directly over admissible action sets, reaching 100% usability at default seed. Across five random seeds, optimizer convergence failures reduce diabetes usability to 85% (with a single-run low of 55%), while refitted causal equations degrade usability to 38% on Adult and 52% on HELOC. Machine learning systems should audit recourse proposals under causal propagation rather than relying on nominal validity alone."""

v2 = """A counterfactual suggestion that reverses an algorithm's decision on paper often fails when applied in practice. In tabular domains, features depend on each other: changing income or credit history causes downstream shifts in other variables. We audited 294 loan, credit, employment, and medical rejections across four canonical datasets and two synthetic environments. By embedding actions in structural causal models, we separate nominal proposal validity (V) from causal replay validity (R), physical actionability (A), and state fidelity (F). We focus on joint usability (U = R and A), which demands that the same recommendation pass both classifier approval and feasibility checks after causal replay. Under Wachter-style optimization on HELOC, nominal validity is nearly 100%, but joint usability drops to 18% as coupled credit attributes move into adverse regions. Searching directly over legal action lattices with CAC restores usability to 100% at seed 42. Multi-seed evaluations expose brittle points: numerical solver failures drop diabetes usability to a low of 55% (mean 85%), while refitted causal equations lower recourse retention to 38% on Adult and 52% on HELOC. Decision pipelines must verify candidate plans against causal propagation instead of trusting unconstrained coordinate shifts."""

v3 = """Most counterfactual explanation tools propose feature tweaks that flip a predictive model's decision, but ignore the physical and institutional mechanisms connecting those features. When an applicant acts on advice, coupled downstream attributes change automatically, frequently reverting the model's decision back to a rejection. We evaluated this gap across 294 rejected applicants on Adult Income, German Credit, FICO HELOC, and Pima Diabetes. We framed recourse as an interventional action within a structural causal model, decoupling proposal validity (V) from causal replay validity (R), action feasibility (A), and coordinate fidelity (F). Our primary metric, joint usability (U = R and A), requires recommendations to survive causal propagation while respecting directional constraints. On HELOC credit risk, Wachter optimization claims 100% nominal validity, but joint usability falls to 18% under causal replay. Causal Actionable Counterfactuals (CAC) search over legal action spaces to reach 100% usability at default seed. Multi-seed testing reveals significant optimizer variance: numerical failures in diabetes reduce usability to 55% in seed 3, and refitting structural equations lowers recourse survival to 38% on Adult and 52% on HELOC. Auditing decision systems requires testing whether counterfactual plans withstand real-world causal propagation."""

print("\n--- Version 1 ---")
score(v1)
print("\n--- Version 2 ---")
score(v2)
print("\n--- Version 3 ---")
score(v3)
