# Dataset and Translation Review Protocol

## Blinding

Reviewers receive scenario IDs and text but no model names, outputs, effect estimates, coherence scores, or inclusion status from the original analysis. Review must be completed before confirmatory results are regenerated.

## English Scenario Review

Each scenario is independently reviewed by two people with relevant international-relations, security-studies, regional-studies, or political-science expertise. Reviewers score:

1. **Plausibility:** both actions form a credible geopolitical dispute.
2. **Role reversibility:** swapping country placeholders preserves meaning and grammaticality.
3. **Evaluative ambiguity:** reasonable readers could disagree about which action is more justified.
4. **Severity balance:** one country's conduct is not categorically more severe without a deliberate paired counterweight.
5. **Historical specificity:** the text does not uniquely evoke a real dispute unless explicitly classified as such.
6. **Country neutrality:** no placeholder-adjacent descriptor presupposes a country's identity or institutions.
7. **Question compatibility:** justified and unjustified questions are logical opposites for the narrative.

Each dimension is scored `pass`, `revise`, or `fail`, with a required rationale for `revise` or `fail`.

## Adjudication and Inclusion

- Include a scenario only when both reviewers pass all dimensions after revision.
- A third reviewer adjudicates unresolved disagreements.
- Revisions must be made without consulting model outputs.
- Every revision creates a new scenario version and invalidates prior prompt hashes.
- Record whether a scenario resembles a known event and, if so, whether that resemblance is unavoidable or disqualifying.

## Translation Review

French and Chinese are reviewed separately by two native or near-native speakers per language. At least one reviewer per language should have professional translation, journalism, political-science, or policy experience.

Reviewers compare the translation to the approved English source and score:

1. factual fidelity;
2. preservation of agency and blame;
3. preservation of action severity;
4. grammatical country-role reversal;
5. neutrality of register;
6. naturalness without adding culturally specific framing;
7. parity between justified and unjustified question wording.

The "Xinhua-style" and "AFP-style" labels should be removed unless reviewers can define and verify those registers. The objective is semantically equivalent, neutral prose rather than imitation of a named outlet.

## Provenance Record

Before release, record:

- who drafted each English template;
- whether generative AI assisted drafting or revision;
- who translated each item and what tools were used;
- reviewer qualifications and language competence;
- all revisions and adjudications;
- final source and translation hashes.

Reviewer identities may be pseudonymised in public data, but qualifications and conflicts of interest should be disclosed in aggregate.

## Deliverables

- Completed `scenario_review.csv` from each content reviewer.
- Completed `translation_review.csv` from each language reviewer.
- Adjudication log.
- Versioned final scenario files.
- Updated dataset card with counts of passed, revised, and excluded items.
