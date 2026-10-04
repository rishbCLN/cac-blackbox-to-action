"""Heuristic LaTeX lint for the hand-written sections (no toolchain here).
Flags brace imbalance, odd '$' counts, and underscores that look unescaped
outside math mode. Run from project root."""
import re
import pathlib

paper = pathlib.Path("paper")
files = [paper / "main.tex"] + sorted((paper / "sections").glob("*.tex"))

def strip_math(s: str) -> str:
    # remove $...$ and \(...\) and \[...\] spans so we only see text mode
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\\([^)]*?\\\)", " ", s)
    s = re.sub(r"\\\[[^]]*?\\\]", " ", s)
    return s

problems = 0
for f in files:
    t = f.read_text(encoding="utf-8")
    nl, nr = t.count("{"), t.count("}")
    if nl != nr:
        print(f"[BRACE] {f.name}: {{={nl} }}={nr}")
        problems += 1
    # $ balance (each line's cumulative parity across file)
    if t.count("$") % 2 != 0:
        print(f"[MATH$] {f.name}: odd number of '$' ({t.count('$')})")
        problems += 1
    # unescaped underscores in text mode (ignore \_ and _ inside math)
    text = strip_math(t)
    for i, line in enumerate(text.splitlines(), 1):
        # drop escaped underscores, then any remaining bare _ is suspect
        bare = re.findall(r"(?<!\\)_", line)
        if bare:
            print(f"[USCORE] {f.name}:{i}: {line.strip()[:90]}")
            problems += 1
    # \text{...} spans containing a raw underscore
    for m in re.findall(r"\\text\{([^}]*)\}", t):
        if re.search(r"(?<!\\)_", m):
            print(f"[TEXT_]  {f.name}: \\text{{{m}}}")
            problems += 1

print("PROBLEMS:", problems)
