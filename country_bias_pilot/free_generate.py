#!/usr/bin/env python3
"""Free-generation experiments.

--mode refusal   : decode GLM-chat's actual response text (illustrates the
                   refusal template); runs on 5 scenarios × 3 languages.
--mode reasoning : elicit a chain-of-thought 'explain briefly then answer
                   (A) or (B)' from Qwen / Mistral / Baichuan instruct
                   models; saves the generated text alongside the A/B
                   commitment, so we can quote the models' actual
                   justifications in the paper.

Uses the same scenarios.py bank and instruct prompt as gpu_bias.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_bias import (
    MODELS, QUESTIONS_BY_LANG, COUNTRIES_BY_LANG,
    _load_model, _has_chat_template,
)
from scenarios import build_scenario_dict


# Fixed 5 representative scenarios (one per interesting type)
SAMPLE_SCENARIOS = ["airspace_v1", "maritime_v1", "cyber_v1",
                    "sovereignty_v1", "internal_affairs_v1"]


def _prompt(scen_text, lang, model_name, tokenizer, is_instruct,
            mode="refusal", use_hedge=True):
    """Build the instruct or reasoning prompt and return its token ids."""
    lang_cfg = QUESTIONS_BY_LANG[lang]
    c1, c2 = "USA", "China"
    country_loc = COUNTRIES_BY_LANG.get(lang, COUNTRIES_BY_LANG["en"])
    c1_loc = country_loc.get(c1, c1)
    c2_loc = country_loc.get(c2, c2)

    narr = scen_text.replace("[COUNTRY_A]", c1_loc).replace("[COUNTRY_B]", c2_loc)
    just = lang_cfg["justified"]["instruct_q"].format(o1=c1_loc, o2=c2_loc)

    if mode == "refusal":
        # Standard instruct prompt (what GLM chat sees during main eval)
        hedge = lang_cfg["hedge"]
        user = f"{narr}\n\n{hedge}\n\n{just}" if use_hedge \
               else f"{narr}\n\n{just}"
    elif mode == "reasoning":
        # Ask for brief CoT before the letter
        hedge = lang_cfg["hedge"]
        user = (f"{narr}\n\n{hedge}\n\n{just}\n\n"
                "First explain your reasoning in 1-2 sentences, "
                "then give your final answer as (A) or (B).")
    else:
        raise ValueError(f"unknown mode {mode}")

    if is_instruct and _has_chat_template(tokenizer):
        msgs = [{"role": "user", "content": user}]
        # First get the prompt string, then encode — avoids tokenizer
        # return-type variation across Fast vs Slow tokenizers.
        prompt_str = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        return tokenizer.encode(prompt_str, add_special_tokens=False)
    else:
        return tokenizer.encode(user, add_special_tokens=False)


@torch.no_grad()
def generate(model, tokenizer, device, prompt_ids, max_new_tokens=60):
    ids_t = torch.tensor([prompt_ids], device=device)
    out = model.generate(
        ids_t,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][len(prompt_ids):].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, gen_ids


def run_refusal(outfile):
    """Decode GLM-4-9B-chat's actual responses across 5 scenarios × 3 langs."""
    model_key = "glm4-9b-chat"
    model_id, is_instruct = MODELS[model_key]
    print(f"Loading {model_id}...")
    model, tokenizer, device = _load_model(model_id)

    records = []
    for lang in ["en", "fr", "zh"]:
        scens = build_scenario_dict(lang)
        for sname in SAMPLE_SCENARIOS:
            if sname not in scens:
                print(f"  skip {sname} (not in {lang} bank)")
                continue
            ids = _prompt(scens[sname], lang, model_key, tokenizer, is_instruct,
                          mode="refusal", use_hedge=True)
            text, _ = generate(model, tokenizer, device, ids, max_new_tokens=80)
            print(f"[{lang}/{sname}] ---")
            print(f"  {text[:300]}{'…' if len(text) > 300 else ''}")
            records.append({
                "model": model_key, "lang": lang, "scenario": sname,
                "generated": text,
            })

    outfile.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nSaved {outfile}")


def run_reasoning(outfile):
    """Elicit CoT + answer on key instruct models, 5 scenarios EN only."""
    models = ["qwen2.5-7b-inst", "mistral-7b-inst", "baichuan2-7b-chat"]

    records = []
    for model_key in models:
        model_id, is_instruct = MODELS[model_key]
        print(f"\n=== {model_key} ===")
        model, tokenizer, device = _load_model(model_id)
        scens = build_scenario_dict("en")
        for sname in SAMPLE_SCENARIOS:
            if sname not in scens:
                continue
            ids = _prompt(scens[sname], "en", model_key, tokenizer, is_instruct,
                          mode="reasoning")
            text, _ = generate(model, tokenizer, device, ids, max_new_tokens=100)
            print(f"[{sname}] {text[:300]}{'…' if len(text) > 300 else ''}")
            records.append({
                "model": model_key, "lang": "en", "scenario": sname,
                "generated": text,
            })
        # Unload model between families
        del model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outfile.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\nSaved {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["refusal", "reasoning"])
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results" / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_out = out_dir / f"{args.mode}.json"
    outfile = Path(args.out) if args.out else default_out

    if args.mode == "refusal":
        run_refusal(outfile)
    elif args.mode == "reasoning":
        run_reasoning(outfile)


if __name__ == "__main__":
    main()
