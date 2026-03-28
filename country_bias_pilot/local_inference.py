"""Local inference for cloze bias experiments.

Runs small models (GPT-2, Qwen-0.5B, etc.) locally on CPU/MPS.
Supports multiple runs per prompt to average out GPU non-determinism.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    RAW_DIR, DTYPE, N_RUNS,
    SCENARIOS_CLOZE, ALL_PAIRS_CLOZE,
    SCENARIOS_CLOZE_ZH, ALL_PAIRS_CLOZE_ZH,
)

logger = logging.getLogger(__name__)


def generate_all_prompts_cloze(pairs=None, scenarios=None, lang="en"):
    """Generate cloze prompts with both option orderings."""
    if lang == "zh":
        pairs = pairs or ALL_PAIRS_CLOZE_ZH
        scenarios = scenarios or SCENARIOS_CLOZE_ZH
    else:
        pairs = pairs or ALL_PAIRS_CLOZE
        scenarios = scenarios or SCENARIOS_CLOZE
    prompts = []
    for c1, c2 in pairs:
        for scen_name, template in scenarios.items():
            # Forward: c1 in COUNTRY_A role
            base_fwd = template.replace("[COUNTRY_A]", c1).replace("[COUNTRY_B]", c2)
            text_fwd_ab = base_fwd.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
            text_fwd_ba = base_fwd.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
            prompts.append({
                "prompt_id": f"{scen_name}__{c1}_vs_{c2}",
                "scenario": scen_name,
                "country_a": c1,
                "country_b": c2,
                "direction": "forward",
                "pair": (c1, c2),
                "text_ab": text_fwd_ab,
                "text_ba": text_fwd_ba,
            })
            # Reverse: c2 in COUNTRY_A role
            base_rev = template.replace("[COUNTRY_A]", c2).replace("[COUNTRY_B]", c1)
            text_rev_ab = base_rev.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
            text_rev_ba = base_rev.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
            prompts.append({
                "prompt_id": f"{scen_name}__{c2}_vs_{c1}",
                "scenario": scen_name,
                "country_a": c2,
                "country_b": c1,
                "direction": "reverse",
                "pair": (c1, c2),
                "text_ab": text_rev_ab,
                "text_ba": text_rev_ba,
            })
    return prompts


def _needs_space(name: str) -> bool:
    """Return False for CJK names (no space prefix needed)."""
    return not any("\u4e00" <= ch <= "\u9fff" for ch in name)


def _load_model(model_id: str):
    """Load model and tokenizer for local inference."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[DTYPE]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Loading {model_id} on {device} in {DTYPE}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def _score_prompt(model, tokenizer, device, prompt_text: str,
                  prefill_ids: list[int], score_token_a: int, score_token_b: int):
    """Run one forward pass and return log-probs for the two score tokens."""
    ids = tokenizer.encode(prompt_text, add_special_tokens=True) + prefill_ids
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[score_token_a].item(), lp[score_token_b].item()


def run_cloze_inference(model_name: str, model_id: str, prompts: list[dict],
                        n_runs: int = N_RUNS, lang: str = "en") -> list[dict]:
    """Run cloze inference locally with multiple runs per prompt.

    Uses letter-label scoring: prompts end with MCQ format "(A) ..\n(B) ..\nAnswer: ("
    and we score on the single tokens "A" and "B". This avoids multi-token country
    name issues and gives ~98% compliance with few-shot examples.

    In AB ordering, (A)=country_a, (B)=country_b.
    In BA ordering, (A)=country_b, (B)=country_a.
    We cross-map when averaging to get each country's true score.

    Returns one JSONL record per (prompt, run).
    """
    model, tokenizer, device = _load_model(model_id)

    # Letter tokens — always the same regardless of country names
    score_token_A = tokenizer.encode("A", add_special_tokens=False)[0]
    score_token_B = tokenizer.encode("B", add_special_tokens=False)[0]
    prefill_ids = []  # No prefill needed for letter labels

    results = []
    total = len(prompts) * n_runs
    done = 0

    for prompt in prompts:
        country_a = prompt["country_a"]
        country_b = prompt["country_b"]

        for run_id in range(n_runs):
            # Score both option orderings
            # AB: (A)=country_a, (B)=country_b
            lp_A_ab, lp_B_ab = _score_prompt(
                model, tokenizer, device,
                prompt["text_ab"], prefill_ids, score_token_A, score_token_B,
            )
            # BA: (A)=country_b, (B)=country_a
            lp_A_ba, lp_B_ba = _score_prompt(
                model, tokenizer, device,
                prompt["text_ba"], prefill_ids, score_token_A, score_token_B,
            )

            # Cross-map: country_a's score is "A" in AB, "B" in BA
            log_prob_a = (lp_A_ab + lp_B_ba) / 2.0
            log_prob_b = (lp_B_ab + lp_A_ba) / 2.0

            compliance_ab = math.exp(lp_A_ab) + math.exp(lp_B_ab)
            compliance_ba = math.exp(lp_A_ba) + math.exp(lp_B_ba)
            compliance = (compliance_ab + compliance_ba) / 2.0

            record = {
                "prompt_id": prompt["prompt_id"],
                "scenario": prompt["scenario"],
                "country_a": country_a,
                "country_b": country_b,
                "direction": prompt["direction"],
                "pair": list(prompt["pair"]),
                "model": model_name,
                "run_id": run_id,
                "log_prob_a": log_prob_a,
                "log_prob_b": log_prob_b,
                "first_token_a": "A",
                "first_token_b": "B",
                "prefill": "",
                "compliance": compliance,
                "log_prob_a_ab": lp_A_ab,
                "log_prob_b_ab": lp_B_ab,
                "log_prob_a_ba": lp_A_ba,
                "log_prob_b_ba": lp_B_ba,
            }
            results.append(record)
            done += 1

            if done % 50 == 0:
                logger.info(f"[{model_name}] {done}/{total} done")

    logger.info(f"[{model_name}] Completed {len(results)} records ({len(prompts)} prompts × {n_runs} runs)")

    # Save raw results
    out_dir = RAW_DIR / "cloze"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{lang}" if lang != "en" else ""
    outpath = out_dir / f"{model_name}_cloze{suffix}.jsonl"
    with open(outpath, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved to {outpath}")

    return results
