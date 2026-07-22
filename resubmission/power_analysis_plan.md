# Power Analysis and Sampling Recommendation

## Independent Units

- **Deterministic cloze:** one run per unique prompt is sufficient. The precision unit is the distinct scenario template, with repeated country pairs, orderings, and question polarities modelled within template.
- **Open-ended validation:** decoding seeds are repeated measurements within a model-template-condition cell. They estimate generation variability but do not replace distinct templates.
- **Developer-location claim:** the independent unit is the model developer or family. The current `n=7` cannot support a confirmatory population claim regardless of prompt count.

## Cloze Planning Calculation

For planning only, the existing 79-scenario English outputs were recoded as

```text
0.5 * (justified score - unjustified score)
```

and averaged across China-involving country pairs before taking the post-trained-minus-base contrast per scenario. Across the seven families, the scenario-level standard deviation ranged from `0.202` to `1.531` log-odds; the median was `0.572` and the 75th percentile was `0.810`.

Using 80% power, a two-sided familywise planning alpha of `0.05 / 7`, and the 75th-percentile standard deviation:

| Smallest effect of interest | Required distinct scenarios |
|---:|---:|
| 0.25 log-odds | 131 |
| 0.50 log-odds | 33 |
| 1.00 log-odds | 9 |

These are normal-approximation planning values, not final mixed-model power estimates. The current 79 templates are adequate for a 0.5-log-odds family-specific effect under this conservative variance choice, assuming enough survive blinded human review. They are not adequate for a 0.25-log-odds target under the same assumptions.

## Open-Ended Agreement Calculation

A conservative exact-binomial calculation treats each distinct scenario as one agreement unit and tests agreement above 0.5. If true direction agreement is 0.70, 16 scenarios provide only about 25% power and approximately 49 are required for 80% power. If true agreement is 0.75, approximately 30 are required.

Accordingly:

- Use **50 distinct human-approved scenarios** for the principal open-ended, no-forced-letter validation.
- Use **16 of those scenarios**, selected by prespecified strata, for the full prompt-condition factorial.
- Use one greedy and three sampled generations per cell. More sampled runs should be added only if the pilot shows seed variance dominates template variance.
- Reverse country order in every cell.

The principal 50-template analysis should estimate agreement across the seven observed model families with template clustering. Do not treat all generations as independent trials.

## Recommended Design

### Deterministic Cloze

- Retain every scenario that passes blinded review, targeting at least 50 and preferably the full current bank.
- Run once per exact prompt/checkpoint/language/order/polarity/scoring condition.
- Use all approved scenarios for primary estimates; never power or filter on model-output coherence.

### Open-Ended Validation

- Principal set: 50 scenarios across scenario type, severity, and country-pair strata.
- Factorial subset: 16 scenarios across the same strata.
- Principal elicitation: default system behaviour, two-to-four sentence assessment, no A/B or forced country commitment.
- Factorial conditions: neutral instruction, no-forced-letter paraphrase, and reason-then-country commitment.
- Decoding: one greedy plus three sampled generations with frozen seeds and parameters.

## Recalculation Before Launch

After human review and a small pilot using the prespecified checkpoint-appropriate protocol, estimate scenario, country-pair, and seed variance components without testing the substantive hypotheses. Recalculate power by simulation from the frozen mixed model. Increase templates if needed; do not reduce the sample because the pilot effect happens to be large.
