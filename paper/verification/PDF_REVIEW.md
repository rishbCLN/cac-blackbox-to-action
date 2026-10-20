# Final manuscript review

Reviewed on 19 October 2026 after the final source compilation.

- Output: `output/pdf/main.pdf`, eight pages, official IEEE Access class.
- Tectonic compilation exited successfully (code 0).
- The missing-class repair was checked in both layouts: the repository's
  nested class directory and the flat upload bundle. The latter compiled
  without any custom search-path arguments. Its class, fonts, and journal
  images sit beside `main.tex`; generated tables and plot retain their paths.
- After the plain-language revision, every page was rendered at 1.5x resolution and
  inspected in a contact sheet;
  dense results pages and revised small tables were inspected at full size.
- The final blue/white charts, table rules, two columns, headers, and footers
  show no clipping or overlapping content. Small audit tables precede the
  reference list rather than appearing on a separate final float page.
- Text-boundary checks found no blocks outside the page and no unresolved
  `??` reference markers.
- Compiler warnings remain, including the official class's zero-width footer
  boxes and title spacing. Their rendered output was checked visually;
  a clean compilation does not mean the log is warning-free.
- All 51 hashed evidence inputs still match `access_evidence.json`.
- The pre-edit wording lock confirms unchanged numerical literals,
  measurement macros, inline mathematics, equations, citations, float structure,
  input paths, labels, and preamble, plus 110 locked files. The preservation
  script also passed seven positive/negative regression checks covering
  permitted prose edits and rejected changes to technical content.
- The plain-language pass simplifies the body, headings, captions, and
  acknowledgment. The original lock is retained. The second lock allows
  caption text edits, but no changes to numbers, mathematics, data, citations,
  or other figure/table code. Formal method names, equations, and algorithm
  steps remain intact.
- The numerical audit passed 284 checks, including fresh replay of the 7,025
  saved main-study plans. The fresh pytest report contains 386 passing tests,
  one skipped test, and no failures or errors.

This is a technical artifact and layout review, not peer review, a similarity
screen, or an independent replication of every experimental search. No
Turnitin result or numerical similarity guarantee is available. Author,
affiliation, correspondence, and archival details remain for the authors.
