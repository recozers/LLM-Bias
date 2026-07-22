# Statistical Analysis Plan

## Scope

This plan must be frozen before new result files are opened. The goal is estimation of checkpoint-associated behavioural differences, not causal attribution to RLHF, SFT, safety tuning, or human annotators.

## Analysis Sets

- **Confirmatory set:** every scenario that passes blinded human content review, independent of model output.
- **Translation set:** only translations approved by native-speaker review; rejected translations are corrected and versioned before inference.
- **Coherence sensitivity set:** scenarios meeting the 50%, 70%, or 90% question-polarity sign-flip diagnostic. These sets must never define the confirmatory sample.
- **Compliance sensitivity sets:** all observations; compliance at least 0.1; compliance at least 0.5; and family-specific exclusions declared below.

## Observation and Outcome

The atomic observation is one model checkpoint x prompt protocol x language x country pair x scenario x country ordering x question polarity. First sign the A/B log-odds toward the prespecified target country. Then multiply the unjustified-question score by `-1`, so positive values have the same semantic meaning under both question polarities. The primary continuous outcome is this direction-corrected, position-symmetrised signed log-odds.

For descriptive aggregation, report:

```text
preference = 0.5 * (justified_score - unjustified_score)
polarity_residual = 0.5 * (justified_score + unjustified_score)
```

The preference score uses both questions. The polarity residual is a question-following diagnostic and must not determine scenario inclusion.

Raw probabilities, complete answer-sequence probabilities, compliance, tokenizer variant, prefill condition, prompt protocol, prompt version, and scenario-review version must be retained for every observation.

## Prompt Matching and Priors

- The confirmatory checkpoint contrast uses semantically identical zero-shot prompts for base and post-trained checkpoints.
- Matched geopolitical demonstrations and matched generic demonstrations are separate prespecified conditions.
- The original asymmetric protocol is sensitivity-only.
- Option-order reversal controls a constant A/B label prior but not label-by-prompt interactions.
- Country-role reversal controls an additive role effect but not country-by-role interactions.
- Full-sequence answer scoring from the unmodified context is primary; prefix-conditioned scores are diagnostic.
- Report total expressed preference and a narrative-conditioned update against two frozen content-free controls. Do not silently subtract either control from the primary score.

## Primary Estimands

1. **Family-specific checkpoint contrast:** post-trained minus base mean preference for each family under English prompts.
2. **Family-specific language interaction:** the change in FR-EN and ZH-EN preference differences between post-trained and base checkpoints.
3. **Probe agreement:** sign agreement and magnitude relationship between forced-choice estimates and blinded human-coded open-ended assessments.

There is no confirmatory pooled "maker-country effect" because only seven purposively selected developers are observed. Maker alignment is descriptive and exploratory.

## Model Specification

Estimate family-specific contrasts in a joint scenario-level model:

```text
direction_corrected_log_odds ~ checkpoint * family * language
           + prompt_protocol
           + question_polarity
           + (1 | scenario)
           + (1 | country_pair)
```

If the full random-effects model is singular, use family-stratified models with scenario and country-pair cluster-robust intervals. Record the convergence decision; do not select a specification based on significance.

Country ordering should average to zero after symmetrisation. Retain it as a diagnostic covariate in an unsymmetrised sensitivity model.

## Inference and Multiplicity

- Report estimates and 95% confidence intervals for every family.
- Apply Holm correction across the seven family-specific checkpoint tests.
- Apply Holm correction separately across prespecified language contrasts.
- Treat opponent-country and scenario-type decompositions as exploratory with false-discovery-rate adjusted values.
- Do not interpret `p > 0.05` as neutrality.
- Claim equivalence only if a smallest effect size of interest is justified before analysis and both one-sided tests pass. Otherwise say the interval includes zero.
- Report exact binomial results for directional counts; do not substitute a small-sample t-test as confirmatory evidence.

## Low-Compliance Policy

- Report compliance before preference results.
- Do not pool observations below 0.1 compliance into confirmatory magnitude estimates.
- Report Mistral base and Mistral English instruct results separately when below threshold.
- For GLM and Yi, report naive and prefill-corrected results side by side and identify corrections as scoring specifications rather than model outputs.

## Prespecified Sensitivities

1. Complete set versus each coherence threshold.
2. Naive token scoring versus tokenizer-variant scoring.
3. Full answer-sequence scoring versus no prefill and prefill scoring for GLM and Yi.
4. Total expressed preference versus each frozen content-free prior control.
5. Matched zero-shot versus matched demonstrations and the original asymmetric protocol.
6. First distinguishing token versus full country-name sequence probability.
7. All models versus leave-one-family-out estimates.
8. All observations versus compliance thresholds.
9. Forced choice versus blinded open-ended coding and commit-position scoring.
10. Greedy versus sampled open-ended decoding.

## Power and Independence

Use every human-approved scenario for deterministic cloze analysis, with a target of at least 50 retained scenarios. Pilot planning with the direction-corrected existing outputs gives a 75th-percentile scenario-level standard deviation of `0.810` log-odds; at 80% power and two-sided planning alpha `0.05 / 7`, approximately 33 scenarios are required for a 0.5-log-odds family-specific effect and 131 for a 0.25-log-odds effect. These values must be updated after a small pilot using the prespecified checkpoint-appropriate protocol.

For the principal open-ended agreement analysis, use 50 distinct scenarios. A conservative exact-binomial calculation gives only about 25% power at 16 scenarios when true agreement is 0.70; approximately 49 are required for 80% power. Use a 16-scenario stratified subset for the full prompt-condition factorial.

More scenario prompts increase within-family precision but do not increase the effective sample of model developers; population claims about maker location require additional independently developed model families.

## Reproducibility

- Freeze scenario, translation, model, tokenizer, chat-template, and analysis versions.
- Save a machine-readable manifest containing model revision hashes and prompt hashes.
- Generate tables and figures from one analysis command.
- Preserve all exclusions with reason codes; never overwrite raw outputs.
