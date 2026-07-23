# Cloze Preference Computation Audit

## Bottom Line

The current score correctly removes a constant A-versus-B label preference by reversing option order, and it reduces narrative-role effects by reversing country roles. It does not, however, fully account for prompt-dependent priors. The base-versus-post-trained contrast intentionally uses stage-appropriate demonstrations and instructions, so it estimates a bundled released-interface difference rather than a prompt-held-constant causal effect. The headline analysis also uses only the justified question, and some prefill corrections change the estimand. These design choices must be stated explicitly and accompanied by the prespecified sensitivity analyses.

## Current Estimator

For country 1 versus country 2, one narrative role, and one question polarity, the code computes

```text
d_ab = log P(A | country1 is option A) - log P(B | country1 is option A)
d_ba = log P(B | country1 is option B) - log P(A | country1 is option B)
position-symmetrised score = (d_ab + d_ba) / 2
```

It then averages the position-symmetrised score across the two country-role assignments. Positive values favour country 1 on the justified question. The unjustified score has the opposite semantic direction.

This is an average of log odds, equivalent to the log of a geometric mean of odds. It is not the log odds of an averaged probability and should not be labelled as such.

## Prior Accounting

### What Is Already Controlled

- A constant answer-label prior cancels under option reversal. If the model adds a stable log-odds preference `lambda` for A, `lambda` enters the first ordering and `-lambda` enters the country-aligned reverse ordering.
- A constant aggressor/defender role effect is reduced by country-role reversal.
- Country ordering is retained in the raw data and can be tested directly.

These cancellations require additivity. They do not remove label-by-wording, label-by-country, label-by-language, or label-by-chat-template interactions.

### Priors That Remain

1. **Checkpoint-interface prior:** base and post-trained checkpoints intentionally do not receive text-identical prompts. In `gpu_bias.py`, bases receive two fictional geopolitical demonstrations, while post-trained models receive two arithmetic demonstrations, a hedge, a different answer instruction, and a chat template. The checkpoint coefficient therefore estimates the bundled difference between the released systems under stage-appropriate elicitation; it is not a prompt-held-constant causal coefficient.
2. **Country-name prior:** the score intentionally contains unconditional country associations. That is defensible if the estimand is total expressed preference, but not if the claim concerns how a scenario changes judgment.
3. **Question-polarity prior:** headline plots use only the justified question. The unjustified question is currently used mainly to select scenarios, rather than as a second measurement of the same direction-corrected construct.
4. **Answer-format prior:** prefill conditions estimate `P(A versus B | supplied prefix)`. This is close to the unconditional answer ratio only when the supplied prefix is a shared, near-deterministic opening. It is not generally valid when the model distributes mass over several response formats.
5. **Tokenizer-event prior:** summing token IDs obtained from `A`, ` A`, `(A`, and newline-plus-A at one answer position mixes surface continuations that are not all compatible with the actual prompt suffix. This can improve coverage but does not define one consistent completion event.

## Required Estimator Revision

### Direction-Correct Both Questions

For each scenario, pair, and checkpoint, define

```text
preference = 0.5 * (justified_score - unjustified_score)
polarity_residual = 0.5 * (justified_score + unjustified_score)
```

The first quantity is the primary preference outcome. The residual is a diagnostic for wording, position, or question-following effects. Do not select scenarios using the residual or sign agreement.

### Separate Total Preference From Narrative Update

Report two estimands rather than silently subtracting a prior:

- **Total expressed preference:** the direction-corrected score under the full scenario.
- **Narrative-conditioned update:** full-scenario score minus a prespecified content-free control that keeps the same question, options, answer format, and country order.

The content-free control must be frozen before viewing its results. Because removing the narrative can make the question unnatural, use at least two controls, such as a neutral shared-action narrative and an options-only prompt, and treat disagreement between them as baseline sensitivity. The total score remains primary unless the paper's claim is explicitly about narrative updating.

### Separate the Primary Interface Estimand From Prompt-Matched Sensitivities

Retain the prespecified checkpoint-appropriate protocol as primary and run a factorial that tests sensitivity to prompt protocol:

1. zero-shot, no hedge, semantically identical question and answer instruction;
2. matched geopolitical demonstrations for both checkpoints;
3. matched generic task demonstrations for both checkpoints;
4. official chat template where required, with the plain-text rendering archived for comparison.

The primary checkpoint contrast compares stage-appropriate interfaces and must be described as a released-system behavioural difference. The matched semantic conditions estimate a narrower prompt-held-constant contrast and test whether particular demonstrations, instructions, or wrappers explain the primary result. Neither estimand identifies a specific training intervention.

### Replace Prefix Conditioning With Sequence Scoring

Score complete answer strings from the unmodified model context, for example `A`, `(A)`, and newline-plus-`(A)`, with matched B strings. Sum probabilities only across a prespecified set of complete, parseable answer forms. Report prefix-conditioned scores separately as diagnostics. This is especially important for Yi, where the supplied prefix is not a deterministic model output.

## Statistical Problems in the Existing Analysis

- Scenario-type means are treated as independent observations in t-tests. There are only 13 purposively chosen types, and country pairs share countries and templates.
- Existing headline plots select scenarios using outputs from the evaluated models and languages.
- Country-level tests reuse overlapping pairs without modelling that dependence.
- The code uses justified-question estimates for headline figures while discussing dual-polarity validation.
- Multiple-testing policies differ across pair, country, family, and language analyses.
- Low A/B compliance makes a conditional A-versus-B ratio a weak description of model behaviour even when the ratio is numerically stable.

The frozen statistical plan addresses these issues with a complete human-approved set, direction-corrected outcomes, family-specific contrasts, scenario and country-pair effects, multiplicity control, and explicit compliance sensitivity analyses.

## Power Consequences

- Cloze logit extraction is deterministic. Repeating an identical prompt adds no power; use one run.
- Distinct human-approved scenarios, not duplicate runs, increase precision for cloze contrasts.
- More scenarios do not increase the independent sample of model developers. No number of prompt templates can turn seven purposively sampled labs into a powered population-level maker-country test.
- Existing pilot variance is suitable only for planning. Re-estimate variance under the prespecified checkpoint-appropriate protocol and update the frozen calculation without selecting the target effect from significance; matched-prompt conditions remain sensitivity analyses.

## Submission Gate

Do not present the current checkpoint contrast as confirmatory until the checkpoint-appropriate estimand is stated explicitly, matched-prompt sensitivity analyses are reported, both question polarities are included in the primary outcome, full-sequence scoring is reported, and complete-set mixed-model results replace output-filtered t-tests.
