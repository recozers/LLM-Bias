# Follow-up experiments

## Hedge-prefix ablation (priority — methodological validity)

The current pipeline applies a neutral hedging prefix (*"This is a
difficult situation with valid perspectives on both sides, but if you
had to choose one answer:"*) **only to instruct-model prompts**, not
to base-model prompts. The base-vs-instruct delta we report is
therefore measured across slightly different prompt contexts. The
hedge text itself is country-neutral, so it cannot introduce direct
country-specific priming, and the observed deltas go in opposite
directions across models (Qwen +3.06 pro-China, Mistral −2.23
anti-China) — inconsistent with a hedge-driven confound. But the
asymmetric prompt is a real limitation and reviewers will ask.

**Ablation to run (cheap, ~25 min / model / condition):**

1. **Qwen 2.5 7B-inst, EN real, no hedge.** If China favourability
   remains strongly positive (> +1.5) the hedge is not the driver; if
   it collapses to near neutral, the hedge is a major confound.
2. **Mistral 7B-inst, EN real, no hedge.** Same, for the
   opposite-direction case. If still strongly negative, hedge is not
   driver; if it attenuates, same caveat.
3. **Optional:** base models *with* the hedge prepended, for at least
   Qwen base and Mistral base. Confirms the base-vs-inst delta is
   robust when prompts are matched.

**Expected outcome.** The hedge is country-neutral and the cross-model
sign pattern is incompatible with a single hedge-driven artefact, so I
expect minimal shift. If that's what we see, the result is a small
paragraph in §2.3 reporting the ablation. If not, we have to
restructure the base-vs-inst comparison.

**Ideal longer-term fix:** apply the hedge to base models too, so the
base-vs-instruct Δ is measured across strictly identical prompts.
Would require re-running all 7 base models × 6 conditions = 42 runs,
~16 hrs on the 3090.

---

## GLM-4-9B-chat: deterministic refusal template

**Finding to follow up.** In a single-prompt diagnostic (USA-vs-China airspace
scenario, EN, standard instruct prompt with hedge prefix), GLM-4-9B-chat
places **P(next token = '\n') = 1.0000** — the model emits a newline
deterministically as the first response token. This is a signature of a
hard-coded RLHF refusal template.

The variant-sum tokenizer fix does NOT rescue measurement here (compliance
stays at ~2×10⁻¹⁰ under the new scoring), because the refusal is
categorical, not tokenization-driven.

## Probing experiments to run later

1. **Prefill `\n` and score the next token.** If GLM's refusal template is
   `\n\nI cannot take sides...`, prefilling `\n` should shift the next-token
   distribution onto the refusal-prefix words (`I`, `Based on`, `This is`,
   etc.). Measuring what comes after gives us the *content* of the refusal.

2. **Free-generation for a few scenarios.** Generate 20–50 tokens greedily
   for a handful of scenarios × languages. Decode and look at the actual
   refusal text. Questions:
   - Is the refusal in English even when the prompt is Chinese?
   - Does the refusal mention country names at all?
   - Does it sometimes break into a latent preference under specific
     phrasings (e.g. historical scenarios vs. contemporary)?

3. **Scoring conditional on refusal commitment.** After prefilling the
   refusal template's first few tokens, does GLM show a residual pro-China
   preference in its continuation? If yes, Zhipu has trained a model that
   *refuses but still biases the refusal content*. If no, the refusal is
   cleanly neutralizing.

4. **Compare to other Chinese labs' refusal formats.** Yi's top-25 tokens
   showed 56% probability mass on verbose-prefix tokens (`Given`, `I`, `Ne`,
   `Based on`). Decode Yi's actual 20-token continuations to see whether
   those are also refusals or just preamble to a real answer.

5. **English-language-only probe.** Does the same refusal template trigger
   when GLM is asked in Chinese? If yes, the refusal is semantic not
   language-gated. If no, there's a language-conditional refusal mechanism
   worth characterizing.

## Why it's deferred

The current paper's headline — post-training amplifies geopolitical bias,
direction follows the maker, language triggers it — is cleanly supported by
the existing data. The refusal-content probing is a separate, richer
research thread on post-training heterogeneity across Chinese labs that
deserves its own analysis section (or its own paper).
