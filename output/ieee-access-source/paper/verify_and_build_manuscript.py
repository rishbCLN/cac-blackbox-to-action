"""Check a prose-only manuscript revision and optionally compile/package it.

This is a factual-preservation tool, not a plagiarism detector or a text
spinner. It cannot access Turnitin, predict a similarity percentage, establish
originality, or replace an author's review of attribution and scientific claims.

Run once BEFORE editing: python paper/verify_and_build_manuscript.py --capture
After editing:          python paper/verify_and_build_manuscript.py --build --bundle
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.tex"
MANIFEST = ROOT / "paper/verification/wording_lock.json"
PLAIN_MANIFEST = ROOT / "paper/verification/plain_language_lock.json"
REPORT = ROOT / "paper/verification/wording_preservation.json"
GENERATED = ROOT / "paper/generated/access"


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def without_comments(source: str) -> str:
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def without_captions(source: str) -> str:
    """Allow caption wording changes, but retain the enclosing float code."""
    result = []
    cursor = 0
    for match in re.finditer(r"\\caption\{", source):
        if match.start() < cursor:
            continue
        depth, end = 1, match.end()
        while end < len(source) and depth:
            if source[end] == "{" and source[end - 1] != "\\":
                depth += 1
            elif source[end] == "}" and source[end - 1] != "\\":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("Unbalanced caption braces.")
        result.extend((source[cursor:match.start()], r"\caption{}"))
        cursor = end
    result.append(source[cursor:])
    return "".join(result)


def document_fingerprint(source: str, value_source: str, editable_captions: bool = False) -> dict:
    """Lock facts and technical structures while leaving prose editable.

    These are conservative syntactic checks, not a proof of unchanged meaning.
    Exact numerical and macro multisets allow prose to move but not values to
    be silently dropped, added, or substituted. Semantic author review is still
    required: equal numbers alone cannot detect swapping their interpretations.
    """
    text = without_comments(source)
    macro_names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", value_source)
    all_commands = Counter(re.findall(r"\\([A-Za-z]+)\b", text))
    protected_blocks = {}
    for environment in (
        "equation", "align", "figure", "figure*", "table", "table*",
        "algorithm", "thebibliography",
    ):
        escaped = re.escape(environment)
        blocks = re.findall(
            rf"\\begin\{{{escaped}\}}.*?\\end\{{{escaped}\}}", text, re.S
        )
        protected_blocks[environment] = [
            digest((without_captions(b) if editable_captions else b).encode())
            for b in blocks
        ]
    return {
        "numeric_literals": dict(sorted(Counter(re.findall(
            r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])", text
        )).items())),
        "measurement_macros": {
            name: all_commands[name] for name in sorted(macro_names)
        },
        "inline_math": dict(sorted(Counter(re.findall(
            r"(?<![\\$])\$(?!\$).*?(?<!\\)\$", text, re.S
        )).items())),
        "citations": dict(sorted(Counter(re.findall(
            r"\\cite(?:\[[^]]*\])?\{[^}]+\}", text
        )).items())),
        "inputs": re.findall(r"\\input\{[^}]+\}", text),
        "labels": re.findall(r"\\label\{[^}]+\}", text),
        "protected_blocks": protected_blocks,
        "preamble_sha256": digest(text.split(r"\begin{document}", 1)[0].encode()),
    }


def compare_document(expected: dict, actual: dict) -> list[str]:
    return [name for name in expected if expected[name] != actual.get(name)]


def locked_paths() -> list[Path]:
    paths = list(GENERATED.glob("*"))
    evidence = ROOT / "paper/verification/access_evidence.json"
    if evidence.exists():
        paths.append(evidence)
        inputs = json.loads(evidence.read_text(encoding="utf-8"))["source_sha256"]
        for name, expected in inputs.items():
            path = ROOT / name
            if not path.is_file() or digest(path.read_bytes()) != expected:
                raise ValueError(f"Evidence input no longer matches its audit: {name}")
            paths.append(path)
    paths.extend((ROOT / "paper/ieee-access").glob("*"))
    return sorted({p for p in paths if p.is_file()})


def capture() -> None:
    if MANIFEST.exists():
        raise ValueError("A wording lock already exists; refusing to replace the baseline.")
    source = MAIN.read_text(encoding="utf-8")
    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_main_sha256": digest(MAIN.read_bytes()),
        "fingerprint": document_fingerprint(
            source, (GENERATED / "values.tex").read_text(encoding="utf-8")
        ),
        "immutable_files_sha256": {
            p.relative_to(ROOT).as_posix(): digest(p.read_bytes())
            for p in locked_paths()
        },
        "scope": "Numerical, citation, equation and data preservation; not similarity screening.",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Captured pre-edit lock: {MANIFEST.relative_to(ROOT)}")


def capture_plain_language() -> None:
    """Add a caption-editing lock only after the original strict check passes."""
    if PLAIN_MANIFEST.exists():
        raise ValueError("A plain-language lock already exists; refusing to replace it.")
    verify(original=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["baseline_main_sha256"] = digest(MAIN.read_bytes())
    manifest["fingerprint"] = document_fingerprint(
        MAIN.read_text(encoding="utf-8"),
        (GENERATED / "values.tex").read_text(encoding="utf-8"),
        editable_captions=True,
    )
    manifest["scope"] = (
        "User-requested plain-language prose and captions. Numeric literals, "
        "math, measurement macros, citations, data and all non-caption float code stay locked. "
        "The original strict wording_lock.json is preserved."
    )
    PLAIN_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Captured caption-editing lock after passing the original preservation check.")


def verify(original: bool = False) -> dict:
    lock = PLAIN_MANIFEST if PLAIN_MANIFEST.exists() and not original else MANIFEST
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    actual = document_fingerprint(
        MAIN.read_text(encoding="utf-8"),
        (GENERATED / "values.tex").read_text(encoding="utf-8"),
        editable_captions=lock == PLAIN_MANIFEST,
    )
    differences = compare_document(manifest["fingerprint"], actual)
    changed = [
        name for name, expected in manifest["immutable_files_sha256"].items()
        if not (ROOT / name).is_file() or digest((ROOT / name).read_bytes()) != expected
    ]
    if differences or changed:
        raise ValueError(f"Preservation check failed: document={differences}, files={changed}")
    return {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "lock_file": lock.relative_to(ROOT).as_posix(),
        "caption_wording_editable": lock == PLAIN_MANIFEST,
        "baseline_main_sha256": manifest["baseline_main_sha256"],
        "revised_main_sha256": digest(MAIN.read_bytes()),
        "prose_source_changed": digest(MAIN.read_bytes()) != manifest["baseline_main_sha256"],
        "preserved_document_groups": list(actual),
        "unchanged_locked_files": len(manifest["immutable_files_sha256"]),
        "turnitin_checked": False,
        "similarity_percentage": None,
        "qualification": (
            "No similarity score or plagiarism-free guarantee is claimed. "
            "Syntactic preservation checks do not replace semantic or attribution review."
        ),
    }


def compile_pdf() -> Path:
    out = ROOT / "output/pdf"
    out.mkdir(parents=True, exist_ok=True)
    portable = ROOT / "tmp/paper-tools/tectonic/tectonic.exe"
    tectonic = str(portable) if portable.is_file() else shutil.which("tectonic")
    env = os.environ.copy()
    support = ROOT / "paper/ieee-access"
    if tectonic:
        command = [tectonic, "-Z", f"search-path={support.as_posix()}",
                   "--keep-logs", "--outdir", str(out), str(MAIN)]
    elif shutil.which("latexmk"):
        for variable in ("TEXINPUTS", "TFMFONTS", "T1FONTS", "TEXFONTMAPS"):
            env[variable] = str(support) + "//" + os.pathsep + env.get(variable, "")
        command = [shutil.which("latexmk"), "-pdf", "-interaction=nonstopmode",
                   "-halt-on-error", f"-outdir={out}", str(MAIN)]
    else:
        raise ValueError("Install Tectonic or latexmk with a TeX distribution to compile.")
    print("Compiling with", Path(command[0]).name, flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    result = out / "main.pdf"
    if not result.is_file() or not result.read_bytes().startswith(b"%PDF-"):
        raise ValueError("The compiler did not produce a valid PDF header.")
    # Convenient requested filename, in addition to the established output path.
    shutil.copyfile(result, ROOT / "main.pdf")
    return result


def bundle() -> Path:
    target = ROOT / "output/ieee-access-source.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = {"main.tex": MAIN, ".latexmkrc": ROOT / ".latexmkrc"}
    # Flat official assets avoid compiler-specific nested font search settings.
    entries.update({p.name: p for p in (ROOT / "paper/ieee-access").glob("*") if p.is_file()})
    for folder in (GENERATED, ROOT / "paper/verification"):
        entries.update({p.relative_to(ROOT).as_posix(): p for p in folder.glob("*") if p.is_file()})
    for name in ("ACCESS_README.md", "build_access_evidence.py", "verify_and_build_manuscript.py"):
        entries[f"paper/{name}"] = ROOT / "paper" / name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source bundle failed its CRC check.")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Lock the pre-edit source once.")
    parser.add_argument("--capture-plain-language", action="store_true",
                        help="Before editing, allow caption prose while retaining the original lock.")
    parser.add_argument("--build", action="store_true", help="Verify, compile, and copy main.pdf.")
    parser.add_argument("--bundle", action="store_true", help="Rebuild the portable source ZIP.")
    args = parser.parse_args()
    try:
        if args.capture_plain_language:
            if args.capture or args.build or args.bundle:
                raise ValueError("Capture must be a separate pre-edit operation.")
            capture_plain_language()
            return 0
        if args.capture:
            if args.build or args.bundle:
                raise ValueError("Capture must be a separate pre-edit operation.")
            capture()
            return 0
        report = verify()
        if args.build:
            pdf = compile_pdf()
            report["compiled_pdf"] = pdf.relative_to(ROOT).as_posix()
            report["compiled_pdf_sha256"] = digest(pdf.read_bytes())
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.bundle:
            print("Source bundle:", bundle().relative_to(ROOT))
        print(f"PASS: {report['unchanged_locked_files']} locked files unchanged; "
              "numbers, data, math, float structure and citations preserved.")
        print("No Turnitin score has been measured or predicted.")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
