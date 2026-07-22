# AAAI-27 Format Audit

Audit date: 2026-07-21

## Submission Policy

- Official main-track instructions: https://aaai.org/conference/aaai/aaai-27/submission-instructions/
- Official supplementary-material instructions: https://aaai.org/conference/aaai/aaai-27/supplementary-material/
- The main paper allows seven pages of non-reference content and nine pages total. Pages 8-9 may contain references only.
- The reproducibility checklist must be uploaded separately from the main paper.
- A technical supplement PDF and a code/data ZIP are optional separate uploads.

## Main PDF

File: `output/pdf/aaai27_anonymous_submission.pdf`

- PASS: 8 pages total; page 8 contains references only.
- PASS: US Letter, two-column AAAI review layout.
- PASS: exact `aaai2027.sty` and `aaai2027.bst` from AuthorKit27; SHA-256 hashes match.
- PASS: anonymous submission mode, anonymous author line, no acknowledgements, and no identifying PDF metadata or extracted text.
- PASS: PDFLaTeX output with embedded, subset Type 1 fonts and no Type 3 fonts.
- PASS: no unresolved references, citation warnings, overfull boxes, page numbers, hyperlinks, JavaScript, or encryption.
- PASS: the two raster figures are embedded at 600 and 632 effective DPI.
- PASS: figure labels were regenerated for the AAAI column width and are at least 9 pt after placement.
- PASS: captions use the kit's 10 pt roman style and references are the final section.
- PASS: all pages were rendered to PNG and inspected for clipping, overlap, unreadable text, or broken figures.

## Separate Uploads

- `output/pdf/aaai27_reproducibility_checklist.pdf`: completed two-page kit checklist for the designated separate upload field.
- `output/pdf/aaai27_technical_supplement.pdf`: anonymous eight-page supplementary document with secondary results and methodological details.
- `output/aaai27_code_data_package.zip`: anonymous cloze-only code, scenario data, result tables, and protocols; legacy MCF code and machine-specific logs are excluded.
- Source ZIPs are provided for local rebuilds. Only the main PDF is required for initial paper submission.

The technical supplement compiles with one page-1 `Overfull \\vbox` warning caused by the submission-style review footer output routine. The rendered page was inspected at full resolution and has no clipping or visible collision. The main paper has no such warning.

## Scientific Work Still Pending

These are not formatting failures and are already disclosed in the manuscript and checklist:

- complete-set checkpoint-appropriate reruns with matched-prompt sensitivities;
- final prior/baseline implementation in the released analysis code;
- exact model and tokenizer revision capture;
- expanded open-ended validation;
- independent geopolitical and native-speaker translation review;
- final dataset/code provenance and redistribution license.

The displayed pilot estimates remain explicitly marked provisional until those steps are complete.
