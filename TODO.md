# TODO: Before Write-Up

## Experiments Still Needed

### 1. Multilingual replication (Chinese + French)
- [ ] Translate scenarios to Chinese and French
- [ ] Translate MCQ prompts (justified/unjustified) to Chinese and French
- [ ] Run all 8 models with Chinese prompts
- [ ] Run all 8 models with French prompts
- [ ] Compare bias rankings across languages — do models behave differently in the language of a country they're evaluating?
- [ ] Key question: does Qwen's pro-China bias strengthen in Chinese? Does Mistral's bias shift in French?

### 2. Fictional country baselines
- [ ] Design 8 fictional country names: mix of phonetically neutral (e.g. "Terluna", "Voskara") and phonetically suggestive (Slavic-sounding, Arabic-sounding, Anglo-sounding, East Asian-sounding)
- [ ] Run full pipeline on fictional countries — single run, ~3 hours
- [ ] Residual bias in neutral names = methodology artifact (position/format bias)
- [ ] Bias in phonetically suggestive names = phonetic/cultural association bias

### 3. Merge USA results
- [ ] Pull USA-pair results from GPU box (running now)
- [ ] Replace "America" rows with "USA" rows in merged dataset
- [ ] Regenerate all plots

## Analysis & Write-Up

### 4. Determine paper theme
- [ ] Core finding: RLHF amplifies pre-training country bias by 5-10x
- [ ] Secondary: model origin determines bias direction (Western-made → anti-China, Chinese-made → pro-China)
- [ ] Tertiary: low-salience countries (Indonesia, Canada) get benefit of the doubt universally
- [ ] Frame: what does this mean for deploying LLMs in geopolitically sensitive contexts?

### 5. Final analysis
- [ ] Statistical comparison: base vs instruct effect sizes (paired t-test across model families)
- [ ] Compliance-weighted results for low-compliance models (Mistral-inst)
- [ ] Cross-language consistency analysis (after Chinese runs)
- [ ] Fictional baseline subtraction (after fictional country runs)

### 6. Write-up
- [ ] Introduction + related work
- [ ] Methodology section
- [ ] Results + figures
- [ ] Discussion
