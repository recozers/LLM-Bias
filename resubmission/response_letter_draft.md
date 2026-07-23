# Draft Response to the Metareview and Reviewers

We thank the metareviewer and reviewers for identifying a mismatch between the strongest evidence in the submission and its original causal framing. The reviews converged on three central problems: the paper attributed behavioural differences to humans or RLHF without isolating those mechanisms; the forced-choice and scenario design required stronger construct validation; and several statistical patterns were described more generally than the model and developer sample supported.

We have therefore restructured the paper around a narrower claim: geopolitical preferences differ between matched released base and post-trained checkpoints and can vary across prompt languages. We no longer claim that bias originates in RLHF, that humans rather than data are responsible, that base models are neutral, or that measured preferences are normatively biased. The revised title is **"Geopolitical Preferences Differ Between Matched Base and Post-Trained Language Models and Across Prompt Languages."**

The revision also clarified two design issues that were not stated clearly enough in the original submission. First, the original coherence subset was selected using evaluated-model outputs; it is now sensitivity-only, with the complete independently reviewed scenario bank designated as primary. Second, base and post-trained checkpoints intentionally received stage-appropriate few-shot content and instructions. The resubmission now defines this released-interface comparison as the primary estimand and uses semantically matched prompt conditions to test its sensitivity to specific elicitation choices.

## Response to the Metareview

**Concern: the topic is valuable, but the causal framing and forced-choice evidence are not sufficiently supported.**

We agree. The title, abstract, introduction, results headings, discussion, limitations, and conclusion have been rewritten to make released-checkpoint behavioural differences the contribution and to leave the post-training mechanism unresolved. The original open-ended experiment is now labelled exploratory because it used one template, deterministic decoding, and a forced A/B commitment. The resubmission protocol adds a 50-scenario principal open-ended assessment with no forced letter, plus a 16-scenario prompt-condition factorial, country-order reversal, sampled generations, and blinded human annotation.

## Response to Reviewer 1

**Concern: a forced-choice answer may not predict what a model says in a realistic open-ended setting.**

We agree. The revised manuscript no longer presents the original free-generation check as independent validation. It now states its limitations explicitly. For the resubmission, the principal open-ended condition requests a short assessment without naming A/B or requiring a final choice. We will evaluate 50 independently reviewed scenarios across all seven post-trained families, reverse country order, include greedy and sampled decoding, and use two annotators blinded to model identity and forced-choice scores. `[TO COMPLETE: insert agreement estimates, uncertainty intervals, and annotation reliability after execution.]`

**Concern: the scenario bank appears to lack deep country and political expertise.**

We have added a dataset card and a blinded review protocol. Every English scenario must be independently approved by two reviewers with relevant international-relations, security-studies, regional-studies, or political-science expertise. The review covers plausibility, reversibility, severity balance, historical specificity, country neutrality, and question compatibility. Revisions are versioned and made without access to model outputs. `[TO COMPLETE: report reviewer qualifications, pass/revise/fail counts, and adjudication outcomes.]`

**Concern: engage with "The Neutral Mask."**

The related-work section now cites this work and uses it to sharpen the paper's limits: output neutrality does not imply neutral representations, and a base/post-trained output contrast cannot establish that a representation was created during post-training.

## Response to Reviewer 2

**Concern: the paper attacks a strawman by treating data and post-training as mutually exclusive explanations.**

We agree. The revised introduction states that pre-training data and post-training interventions both shape observable behaviour. The paper now asks where released-checkpoint outputs differ, not whether data or humans are exclusively responsible.

**Concern: the European or French conclusion rests on one Mistral family.**

We have removed the population-level interpretation. Mistral is presented as one family-specific French-prompt association, not evidence about European models or a general maker-language activation mechanism.

**Concern: model-generated translations may be unreliable.**

The revised manuscript identifies the translations as model-generated and unvalidated. Before resubmission, each French and Chinese item will be independently reviewed by two native or near-native speakers for factual fidelity, agency, severity, reversibility, register, naturalness, and question-polarity parity. The "AFP-style" and "Xinhua-style" labels have been removed. `[TO COMPLETE: report review and revision counts.]`

## Response to Reviewer 3

**Concern: the central "humans, not data" and RLHF-origin claims are unsupported.**

We agree and have removed them. The released checkpoints do not reveal whether supervised fine-tuning, preference optimisation, safety training, data curation, or another intervention produced an observed difference. The revised paper explicitly states that staged checkpoints or controlled interventions would be required for that attribution.

**Concern: the benchmark contribution overlooks UNQOVER and BBQ.**

The related-work section now treats UNQOVER and BBQ as direct methodological predecessors and narrows the contribution to a multilingual geopolitical application with matched checkpoints and scoring diagnostics.

**Concern: the paper does not engage with known post-training artefacts.**

We added work on length bias, format bias, sycophancy, and output/representation divergence. These studies now motivate a probe-sensitivity interpretation rather than a mechanism claim.

**Concern: dataset authorship and review are unclear.**

The dataset card now records required provenance fields for authorship, generative-AI assistance, translation tools, reviewer qualifications, revisions, and hashes. These fields are submission blockers rather than optional documentation. `[TO COMPLETE: fill provenance and review fields.]`

**Concern: "whose bias?" is unresolved without a normative baseline.**

We agree. The revised paper uses **expressed geopolitical preference** as the descriptive construct. Zero is a role- and position-symmetric probe null, not a claim of political neutrality or correctness. The study does not adjudicate which country's action is normatively justified.

## Response to Reviewer 4

**Concern: base-versus-chat differences cannot isolate RLHF, SFT, or safety tuning.**

We agree. All component-level attribution has been removed. The base and post-trained prompts intentionally differ in demonstrations, instructions, and interface because the primary estimand compares released checkpoints under stage-appropriate elicitation. We now state that design explicitly and use matched semantic prompt conditions to test whether specific prompt choices explain the contrast. `[TO COMPLETE: report the primary released-interface contrasts and matched-prompt sensitivity results.]`

**Concern: maker alignment and Chinese-language amplification are statistically mixed.**

The 6/7 maker-direction count is now explicitly descriptive and nonsignificant under the exact two-sided binomial test. No confirmatory maker-country population claim is made from seven purposively selected families. The Chinese-prompt population claim has been removed; family-specific language contrasts will be multiplicity-adjusted.

**Concern: tokenizer and prefill corrections appear ad hoc.**

We now distinguish unconditional full-sequence answer scoring from prefix-conditioned diagnostics. The primary rerun will score prespecified complete answer strings from the unmodified context and report naive, token-variant, and prefill-conditioned results side by side. `[TO COMPLETE: insert full-sequence sensitivity table.]`

**Concern: the coherence filter may create selection bias.**

We agree. The complete human-approved scenario set is now primary. The 50%, 70%, and 90% coherence thresholds are sensitivity analyses only and cannot determine inclusion.

**Concern: the open-ended check is narrow and uses greedy decoding.**

The expanded protocol uses one greedy and three frozen-seed sampled generations, no-forced-letter assessment as the principal condition, a prespecified prompt factorial, and blinded annotation. Repeated generations are modelled within template rather than treated as independent observations.

**Concern: the sample is limited to small open models and three languages.**

We have narrowed the scope accordingly. The conclusions apply to the seven observed 7--9B open-weight families and the three tested prompt languages. We do not generalise to larger, closed, or nationally representative model populations.

## Summary of New Submission Evidence

Before resubmission, this letter will be updated with:

1. blinded geopolitical-review outcomes and complete provenance;
2. native-speaker French and Chinese review outcomes;
3. complete-set checkpoint-appropriate cloze estimates and matched-prompt sensitivities from the frozen mixed model;
4. full-sequence, compliance, prefill, and coherence sensitivity analyses;
5. expanded open-ended agreement and annotation-reliability results;
6. regenerated figures and multiplicity-adjusted intervals.

We will not submit while any bracketed placeholder remains.
