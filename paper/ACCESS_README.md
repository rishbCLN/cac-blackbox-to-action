# Verified IEEE Access manuscript

The new entry point is the repository-root `main.tex`. It uses the official
IEEE Access class, blue journal headings, white pages, and two columns. Its
writing and workflow diagram are new. The existing modular `paper/main.tex`
and the untracked `rhino.tex` were not modified.

## Deliverables

- `main.tex`: manuscript, equations, workflow diagram, and inline bibliography.
- `output/pdf/main.pdf`: compiled manuscript after layout verification.
- `output/ieee-access-source.zip`: complete LaTeX source bundle for upload.
- `paper/generated/access/`: generated values, tables, and vector/result plots.
- `paper/verification/`: source hashes, fresh pytest XML, verified and replayed
  headline CSVs, corrected applicant statistics, and discrepancy reports.
- `paper/build_access_evidence.py`: rebuilds the paper's measured values from
  the existing experiment artifacts. It does not overwrite those artifacts.
- `paper/ieee-access/`: unmodified official class, font files, and journal assets.

## Rebuild the evidence

Run these commands from the repository root with the project dependencies
already installed:

```powershell
python -m pytest -q -ra -p no:cacheprovider --basetemp tmp/pytest-access-new --junitxml paper/verification/pytest.xml
python paper/build_access_evidence.py --replay
```

Use a fresh `--basetemp` name if needed; pytest owns and clears its specified
temporary directory. The replay audits the saved main-study plans, not newly
generated searches. The synthetic experiments retain only aggregate outcomes,
so their tables are checked against the saved summaries rather than replayed
from individual plans. Omit `--replay` for a fast CSV consistency rebuild; the
JSON report explicitly records whether fresh model replay was performed.

The generator stops on inconsistent reported primary metrics, missing audit
rows, mismatched applicants, inconsistent seed denominators, failed pytest
results, or predictive metrics that disagree with the model reports.

## Build the manuscript

With a standard TeX Live/MiKTeX installation:

```powershell
latexmk -pdf main.tex
```

The included `.latexmkrc` supplies search paths for the official class and its
fonts. The bibliography is inline; BibTeX is unnecessary. With the portable
Tectonic executable downloaded for this task:

```powershell
tmp/paper-tools/tectonic/tectonic.exe -Z search-path=paper/ieee-access --keep-logs --keep-intermediates --outdir output/pdf main.tex
```

Tectonic downloads its TeX resource bundle on the first build. A completed
build subsequently uses the cache. The manuscript includes a XeTeX-only
adapter because the official spot-color package is specific to pdfLaTeX.
The template palette is also registered with xcolor in CMYK for consistent
TikZ color mixing. pdfLaTeX loads the unmodified official spot-color package.
The template's publication-year setter
is also renamed locally so TikZ can use the normal TeX year register.

For Overleaf, create a new project by uploading `output/ieee-access-source.zip`.
This upload bundle puts the unmodified official class, fonts, and image assets
beside `main.tex`, so their discovery does not depend on custom search paths.
Do not upload only `main.tex`: `ieeeaccess.cls` is not a standard class supplied
by every TeX installation, and the generated tables and plot are also needed.

Alternatively, upload `main.tex`, `.latexmkrc`, and the complete
`paper/ieee-access/` and `paper/generated/access/` directories, retaining their
paths. Set the root `main.tex` as the main document and use pdfLaTeX. The
generated inputs are included, so a Python run is not needed on Overleaf.

## Evidence corrections reflected in the manuscript

1. Usability is replay success AND feasibility on the **same** suggestion.
   Proposed-state validity and state fidelity are separate measurements.
2. Headline rates use all queried applicants, including generation failures.
3. Several baselines tie CAC; the observational comparison does not contain
   MINT. No universal superiority or novelty claim is made.
4. The saved significance summary pools applicants across repeated seeds.
   New tables recompute exact McNemar tests at seed 42, apply Holm correction
   per dataset, and bootstrap applicant clusters for across-seed effect sizes.
   The original experiment summary is preserved.
5. Adult has only four saved seeds for three baselines. Diabetes CAC seed 3
   records twelve generation failures. Adult DiCE seed 2 has zero usability
   without recorded generation exceptions. These observations are retained.
6. The saved refitted-FCM results contain only the `rescore` arm, whose fitting
   data overlap with the generator's. They are not a disjoint cross-fit audit.
7. The synthetic Block world's intrinsic blocked count is distinct from the
   number of applicants for whom a finite search finds no usable plan.
8. Saved and default-recomputed cost-ratio conventions differ. The new paper
   reports absolute effort and logs the ratio discrepancy for inspection.
9. Test-triggered tuning and pre-split preprocessing remain limitations of
   the measured experiments. Artifact agreement does not remedy them.
10. Synthetic user-study responses are excluded from empirical claims.

## Template provenance and submission details

The official template was downloaded on 19 October 2026 from IEEE's
[Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/)
page. The archive is
[ACCESS_latex_template_20260513-1-1.zip](https://ieeeaccess.ieee.org/wp-content/uploads/2026/05/ACCESS_latex_template_20260513-1-1.zip).
Only its class, font support files, and required journal image assets are
bundled; sample author text and example figures are excluded.

Author identity, affiliation, correspondence, funding, and any repository
archive identifier still need to be supplied by the research authors. The
manuscript intentionally has no fabricated DOI or publication date. Its
acknowledgment discloses AI assistance, consistent with the
[IEEE Access submission guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/).
Author review remains necessary before submission. No originality/similarity
screening or external peer review has been performed.
