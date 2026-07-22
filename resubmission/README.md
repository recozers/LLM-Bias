# Resubmission Work Package

This directory converts the AIES reviews into an executable revision plan. The manuscript's central claim has already been narrowed from a causal claim about bias originating in RLHF to an observational claim about behavioural differences between released base and post-trained checkpoints.

## Completed Without New Inference

- Replaced the title, abstract, introduction, discussion, limitations, and conclusion with checkpoint-localised claims.
- Defined `expressed geopolitical preference` separately from normative bias.
- Reclassified maker alignment and population-level Chinese-language amplification as exploratory.
- Added UNQOVER, BBQ, length bias, format bias, sycophancy, and shallow-alignment literature.
- Reclassified the model-dependent coherence filter as a sensitivity analysis.
- Specified a complete-set primary analysis and documented the required rerun.
- Added a statistical analysis plan, reviewer-response matrix, dataset-review protocol, translation-review protocol, and open-ended validation protocol.
- Audited the cloze estimator, answer priors, prompt matching, and existing inference code.
- Added pilot-variance power calculations and revised the open-ended recommendation to 50 principal scenarios plus a 16-scenario factorial subset.
- Drafted a point-by-point response letter with explicit completion placeholders.
- Added a dataset card and reviewer/annotation templates.

## Remaining Compute-Dependent Work

1. Rerun the prespecified checkpoint-appropriate protocol on the complete set, with semantically matched prompt conditions as sensitivity analyses.
2. Recompute every headline estimate and figure on the complete human-approved scenario set using both question polarities.
3. Fit the prespecified scenario-level models and multiplicity-adjusted family contrasts.
4. Run scoring sensitivities: content-free prior controls, full-sequence answer scoring, tokenizer/prefill variants, low-compliance exclusions, and leave-one-family-out analyses.
5. Run the expanded open-ended validation conditions.
6. Replace every `RESUBMISSION BLOCKER` comment and provisional number in the manuscript.

## Remaining Human-Dependent Work

1. Confirm and record scenario authorship, tools used during construction, and revision history.
2. Obtain blinded geopolitical review of every English scenario.
3. Obtain independent native-speaker review of every French and Chinese translation.
4. Human-annotate the expanded open-ended outputs with two annotators per item.

## Submission Gate

Do not submit until:

- the primary results use the complete human-approved set;
- human-review fields in the dataset card are complete;
- all visible future-tense language and internal blocker comments are removed;
- every numeric claim is regenerated from a frozen analysis output;
- the final source compiles without undefined citations or references.
