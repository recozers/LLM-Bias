"""Model loading and logit extraction."""

import logging
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import DTYPE, RAW_DIR, TOP_K_CHECK, COMPLIANCE_WARN

logger = logging.getLogger(__name__)


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_model(model_id: str, device: str = "cuda"):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    dtype = _resolve_dtype(DTYPE)
    logger.info(f"Loading {model_id} in {DTYPE} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _find_ab_token_ids(tokenizer) -> dict:
    """Identify token IDs for 'A', 'B' and variants like ' A', ' B'.

    Returns dict with keys 'A_ids' and 'B_ids' (lists of candidate token ids),
    plus 'primary_A' and 'primary_B' for the best single candidate.
    """
    candidates_a, candidates_b = [], []
    for text in ["A", " A", "A)", " A)"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            candidates_a.append((text, ids[0]))
    for text in ["B", " B", "B)", " B)"]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            candidates_b.append((text, ids[0]))

    # Prefer bare "A"/"B"; fall back to " A"/" B"
    primary_a = candidates_a[0][1] if candidates_a else tokenizer.encode("A", add_special_tokens=False)[-1]
    primary_b = candidates_b[0][1] if candidates_b else tokenizer.encode("B", add_special_tokens=False)[-1]

    info = {
        "A_candidates": candidates_a,
        "B_candidates": candidates_b,
        "A_ids": [c[1] for c in candidates_a],
        "B_ids": [c[1] for c in candidates_b],
        "primary_A": primary_a,
        "primary_B": primary_b,
    }
    logger.info(f"Token resolution: A={candidates_a}, B={candidates_b}")
    return info


def extract_logits_single(
    model,
    tokenizer,
    prompt_text: str,
    token_info: dict,
    device: str = "cuda",
) -> dict:
    """Run one forward pass and extract logits for A/B tokens.

    Returns dict with logit_a, logit_b, prob_a, prob_b, prob_a_normalized,
    compliance, top_k_tokens, and warning flags.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at the last token position (next-token prediction)
    last_logits = outputs.logits[0, -1, :]  # (vocab_size,)

    # --- pick the best A and B token ids based on which has highest logit ---
    a_ids = token_info["A_ids"] or [token_info["primary_A"]]
    b_ids = token_info["B_ids"] or [token_info["primary_B"]]

    logit_a = max(last_logits[tid].item() for tid in a_ids)
    logit_b = max(last_logits[tid].item() for tid in b_ids)
    best_a_id = max(a_ids, key=lambda tid: last_logits[tid].item())
    best_b_id = max(b_ids, key=lambda tid: last_logits[tid].item())

    # Normalized probability over just A and B
    ab_logits = torch.tensor([logit_a, logit_b])
    ab_probs = F.softmax(ab_logits, dim=0)
    prob_a_norm = ab_probs[0].item()

    # Full-softmax compliance: how much mass lands on A+B
    full_probs = F.softmax(last_logits, dim=0)
    prob_a_full = sum(full_probs[tid].item() for tid in a_ids)
    prob_b_full = sum(full_probs[tid].item() for tid in b_ids)
    compliance = prob_a_full + prob_b_full

    # Top-k check
    topk_vals, topk_ids = torch.topk(last_logits, TOP_K_CHECK)
    topk_tokens = [
        (tokenizer.decode([tid]), last_logits[tid].item())
        for tid in topk_ids.tolist()
    ]
    a_in_topk = best_a_id in topk_ids.tolist()
    b_in_topk = best_b_id in topk_ids.tolist()

    warnings = []
    if not a_in_topk:
        warnings.append(f"Token A (id={best_a_id}) not in top-{TOP_K_CHECK}")
    if not b_in_topk:
        warnings.append(f"Token B (id={best_b_id}) not in top-{TOP_K_CHECK}")
    if compliance < COMPLIANCE_WARN:
        warnings.append(f"Low compliance: {compliance:.4f}")

    if warnings:
        for w in warnings:
            logger.warning(w)

    return {
        "logit_a": logit_a,
        "logit_b": logit_b,
        "prob_a": prob_a_full,
        "prob_b": prob_b_full,
        "prob_a_normalized": prob_a_norm,
        "compliance": compliance,
        "best_a_token": tokenizer.decode([best_a_id]),
        "best_b_token": tokenizer.decode([best_b_id]),
        "a_in_top_k": a_in_topk,
        "b_in_top_k": b_in_topk,
        "top_k_tokens": topk_tokens[:5],  # store top-5 for inspection
        "warnings": warnings,
    }


def run_inference(
    model_name: str,
    model_id: str,
    prompts: list[dict],
    device: str = "cuda",
) -> list[dict]:
    """Load model, run all prompts, save raw results, return records."""
    model, tokenizer = load_model(model_id, device=device)
    token_info = _find_ab_token_ids(tokenizer)

    outpath = RAW_DIR / f"{model_name}_logits.jsonl"
    results = []

    with open(outpath, "w") as f:
        for prompt in tqdm(prompts, desc=f"Inference [{model_name}]"):
            logit_data = extract_logits_single(
                model, tokenizer, prompt["text"], token_info, device=device,
            )
            record = {
                "prompt_id": prompt["prompt_id"],
                "scenario": prompt["scenario"],
                "country_a": prompt["country_a"],
                "country_b": prompt["country_b"],
                "direction": prompt["direction"],
                "pair": list(prompt["pair"]),
                "model": model_name,
                **logit_data,
            }
            f.write(json.dumps(record) + "\n")
            results.append(record)

    logger.info(f"Saved {len(results)} records to {outpath}")

    # Free memory
    del model
    torch.cuda.empty_cache()

    return results
