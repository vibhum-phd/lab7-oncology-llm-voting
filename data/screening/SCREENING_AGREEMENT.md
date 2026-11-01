# Screening agreement report (2026-11-01)

**Why:** Sir has no biology background, so the planned human check was replaced (D-023) by an
independent blind re-screen by a second Claude agent.

**Design.** First screener: Claude Code, all 1,026 pool items, from a short screening view
(`ai_codes.csv`; 18 unsure items then decided from full text, `ai_unsure_decisions.csv`;
→ `ai_codes_final.csv`). Second screener: a separate Claude agent with no access to the first codes,
reading the **full text** of a random 20% sample (205 items, seed 20261101) and the written rules
(`SCREENING_RULES.md`, plus the clarifications listed in its addendum) → `second_screener_codes.csv`.
Its own report says it opened no other files and did not use the web (not independently verifiable).

**Agreement (VERIFIED, computed from the two files)**

| Measure | Value |
|---|---|
| Items compared | 205 |
| Exact code agreement (M/O/X) | 190/205 = 92.7%, Cohen's kappa 0.873 |
| Eligible (M or O) vs excluded (X) | 93.2%, kappa 0.832 |
| Second screener's borderline flags | 45 |

Cross-table (rows first screener, columns second): M→M 26, M→O 1, M→X 4; O→O 113, O→X 2;
X→M 4, X→O 4, X→X 51.

For comparison, the review's own screening check (review §2.4) reported kappa 0.92 for
include/exclude and 0.78 for the exact category.

**Adjudication.** Claude Code decided all 15 disagreements from the full text against the written
rules (`adjudication_log.csv`): 8 codes changed and 7 first codes kept. Two first codes were plainly
wrong once the full text was read (benign gallbladder adenomyomatosis and a simple breast cyst, both
first coded M). Most kept codes were generic post-operative complications, which the rules addendum
counts as X. Final codes: `codes_adjudicated.csv` (M 129, O 583, X 314).

**What this does and does not show — INFERRED**
- It shows the written rules can be applied consistently by two independent AI passes.
- It does not show clinical correctness: both screeners are AI, and no clinician checked any item.
- The 821 items outside the sample keep their first-pass code. The sample suggests about 4% of
  first-pass codes would change on full-text review (8/205), most often items first coded M.
- The objective-subset sensitivity analysis (oncology term in the correct answer) guards the primary
  result against these errors.
