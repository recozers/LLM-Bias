# Follow-up experiments — deeper refusal probing

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
