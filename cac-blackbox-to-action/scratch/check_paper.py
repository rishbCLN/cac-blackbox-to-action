"""Static consistency check for the paper: refs, cites, inputs, figures.
Run: python scratch/check_paper.py  (from the project root)."""
import re
import pathlib

paper = pathlib.Path("paper")
tex_files = ([paper / "main.tex"]
             + sorted((paper / "sections").glob("*.tex"))
             + sorted((paper / "tables").glob("*.tex")))
alltex = "\n".join(p.read_text(encoding="utf-8") for p in tex_files)

labels = set(re.findall(r"\\label\{([^}]+)\}", alltex))
refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", alltex))
print("LABELS:", len(labels), "| REFS:", len(refs))
print("UNDEFINED refs:", sorted(r for r in refs if r not in labels) or "none")

bib = (paper / "references.bib").read_text(encoding="utf-8")
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = set()
for m in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]+)\}", alltex):
    cites.update(k.strip() for k in m.split(","))
print("\nBIB:", len(bibkeys), "| CITES:", len(cites))
print("MISSING cites:", sorted(c for c in cites if c not in bibkeys) or "none")
print("UNUSED bib   :", sorted(bibkeys - cites) or "none")

inputs = re.findall(r"\\input\{([^}]+)\}", alltex)
miss_in = [i for i in inputs if not (paper / (i + ".tex")).exists()]
print("\nINPUTs:", len(inputs), "| missing:", miss_in or "none")

figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", alltex)
def fig_ok(f):
    b = paper / "figures" / f
    return any((b.parent / (b.name + ext)).exists() for ext in (".pdf", ".png", ""))
print("FIGURES used:", figs)
print("MISSING figs:", [f for f in figs if not fig_ok(f)] or "none")

# environment begin/end balance, per file
from collections import Counter
print("\nENVIRONMENT balance:")
any_mismatch = False
for f in tex_files:
    t = f.read_text(encoding="utf-8")
    b = Counter(re.findall(r"\\begin\{([^}]+)\}", t))
    e = Counter(re.findall(r"\\end\{([^}]+)\}", t))
    diff = {k: (b[k], e[k]) for k in set(b) | set(e) if b[k] != e[k]}
    if diff:
        print(f"  [MISMATCH] {f.name}: {diff}")
        any_mismatch = True
if not any_mismatch:
    print("  all balanced")
