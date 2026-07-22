# Reviewer Response Matrix

| Source | Concern | Revision or evidence | Status |
|---|---|---|---|
| Metareview | Claims exceed the evidence | Removed "humans, not data," "originates," and RLHF-causal framing throughout the manuscript | Completed |
| Metareview / R1 | Forced choice may not predict open-ended behaviour | Existing open-ended validation retained; expanded decoding/system-prompt protocol specified | Existing evidence plus compute required |
| R1 | Prompt bank appears politically superficial | Added construction criteria and blinded geopolitical-review protocol | Human review required |
| R1 | Engage with *The Neutral Mask* | Added to related work and used to limit output-to-representation claims | Completed |
| R2 | "Bias is a data problem" is a strawman | Introduction now states that pre-training and post-training both shape behaviour | Completed |
| R2 | European conclusions rest on Mistral | Mistral is treated as a family-specific existence case; no European population claim | Completed |
| R2 | Automated translations may be unreliable | Limitation strengthened; native-speaker validation protocol and review template added | Human review required |
| R3 | Contribution framed as a conceptual dispute that does not exist | Reframed as a narrower matched-checkpoint empirical investigation | Completed |
| R3 | Cannot identify which post-training component causes shifts | Explicitly limits attribution; staged checkpoints are future mechanistic work | Completed |
| R3 | Title overclaims "humans" and "not data" | Replaced with "Geopolitical Preferences Differ Between Matched Base and Post-Trained Language Models and Across Prompt Languages" | Completed |
| R3 | Missing UNQOVER and BBQ | Added both and acknowledged their direct methodological relevance | Completed |
| R3 | Missing post-training amplification literature | Added length bias, format bias, sycophancy, and shallow-alignment work | Completed |
| R3 | Dataset authorship/review unclear | Added data card and provenance fields; manuscript states construction criteria | Author confirmation and human review required |
| R3 | "Whose bias?" lacks a human baseline | Defined the construct as expressed preference and disclaimed normative correctness | Completed |
| R4 | Base-chat comparison cannot isolate SFT/RLHF/safety | Claims now localise change only between released checkpoints | Completed |
| R4 | Maker alignment and Chinese amplification statistically mixed | Maker result labelled exploratory; Chinese population claim labelled unsupported | Completed |
| R4 | Tokenizer/prefill corrections appear ad hoc | Prespecified side-by-side scoring sensitivity plan added | Compute required |
| R4 | Coherence filter creates selection bias | Complete human-approved set is now primary; coherence thresholds are sensitivity analyses | Manuscript policy completed; rerun required |
| R4 | Greedy open-ended validation is narrow | Expanded system-prompt and decoding protocol specified | Compute and annotation required |
| R4 | Limited model/language sample | Scope claims narrowed; maker-location generalisation removed | Completed |
| Internal audit | Base and post-trained checkpoints received different demonstrations and instructions | Defined checkpoint-appropriate released-interface contrast as primary; matched semantic prompt factorial tests sensitivity | Code/protocol audit completed; rerun required |
| Internal audit | Headline score used justified question only | Primary outcome now direction-corrects and combines both polarities | Plan completed; analysis implementation and rerun required |
| Internal audit | Constant label priors cancel, but prompt-dependent priors do not | Added content-free controls, full-sequence scoring, and prompt-protocol sensitivities | Plan completed; rerun required |
| Internal audit | Sixteen open-ended scenarios are underpowered for 70% agreement | Principal set increased to 50 scenarios; 16 retained only for the factorial subset | Completed |

## Response-Letter Position

The response should not argue that reviewers misunderstood the original paper. It should state that their convergence revealed a mismatch between the strongest empirical contribution and the original causal narrative, then show that the resubmission makes the behavioural evidence primary and the causal mechanisms explicitly unresolved.
