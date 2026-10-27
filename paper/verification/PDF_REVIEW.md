# Final manuscript review

Reviewed on 22 October 2026 after the verified source compilation.

- Output: `output/pdf/main.pdf` and `main.pdf`, eleven pages, official IEEE Access class.
- Tectonic compilation exited successfully (code 0).
- The missing-class repair was checked in both layouts: the repository's
  nested class directory and the flat upload bundle. The latter compiled
  without any custom search-path arguments. Its class, fonts, and journal
  images sit beside `main.tex`; generated tables and plot retain their paths.
- Authors and Affiliations: Full authorship is specified for Rishabh Kumar,
  Shashwat Karn, and Anubhav Bhattacharya (VIT SCOPE) and Dr. Navamani T M
  (Department of Computational Intelligence, VIT SCOPE) as corresponding author.
- Author Contributions: Formalized across work sections (~90% for Rishabh Kumar
  leading core theory, algorithm, and benchmarks; ~8% for Shashwat Karn covering
  causal model sensitivity and validity; ~2% for Anubhav Bhattacharya covering
  dataset verification) in both front matter and biographies.
- Diagram Integrity: Figure 2 (Pipeline Flowchart) verified with clean routing
  beneath audit nodes; Figure 3 (SCM DAG) verified with non-overlapping intervention
  arrows, severed parent noise indicator ($\times$), and full-stem dotted arrows.
- Biographies: Complete, professional academic biographies for all four authors
  are positioned on Page 11 with no orphaned pages.
- Results and Setup Structure: Section IV established as Experimental Setup;
  Section V contains dedicated Result Checks and Verification Protocol without tables.
- Text-boundary checks found no blocks outside the page and no unresolved
  `??` reference markers.
- Compiler warnings remain consistent with the official class's zero-width footer
  boxes and title spacing. Rendered output verified visually.
- All evidence inputs and generated tables match the verified benchmark evaluations.

