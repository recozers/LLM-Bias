# TODO: Before Write-Up

## Experiments Still Needed

### 1. Multilingual replication (Chinese + French) -- DONE
- [x] Translate MCQ prompts (justified/unjustified) to Chinese and French
- [x] Run all 8 models with Chinese prompts → results/gpu_bias_zh/
- [x] Run all 8 models with French prompts → results/gpu_bias_fr/
- [x] Key findings:
  - Models become more favourable toward the country whose language they're prompted in
  - Mistral-inst: France 22%→42% in French; China 8%→29% in Chinese
  - Qwen-inst: pro-China bias rock-solid across all languages (93/89/94%)
  - RLHF alignment is partially language-dependent

### 2. Fictional country baselines -- DONE
- [x] Designed 8 fictional names: 4 neutral + 4 phonetically suggestive → results/gpu_bias_fictional/
- [x] Key findings:
  - Phonetic bias mirrors real-country bias (Zhaodong treated like China, Bretherland like Anglo countries)
  - Qwen-inst: Zhaodong 71%, Bretherland 26% — same pattern as China vs USA
  - Mistral-inst: Bretherland 69%, Zhaodong 40% — same mirror
  - "Neutral" names not fully neutral (Voskara disfavoured, Drethia favoured) — phonetic associations unavoidable
  - RLHF amplifies phonetic bias just like real-country bias

### 3. Merge USA results -- DONE
- [x] Replaced "America" with "USA" across all 8 models
- [x] Mistral-inst dropped 17pp (USA more disfavoured than America)

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
