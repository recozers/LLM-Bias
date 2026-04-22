#!/usr/bin/env python3
"""Free-generation validation experiment.

Goal: show that under open-ended generation (reasoning chain + final
answer), all 7 post-trained models produce the same directional
China-preference as the logit-based forced-choice probe.

This collapses §5's "four Chinese labs, four response formats" narrative:
the formats differ, the preference does not.

Design:
  - 7 post-trained models (all Western + all Chinese labs)
  - 10 scenarios (subset of the 29 coherent), selected to span types
  - 3 country pairs per scenario: USA/China, France/China, Indonesia/China
  - Both orderings (cancel position effects) = 2
  - Prompt asks for 1-2 sentence reasoning then "(A)" or "(B)"
  - Decoded answer extracted via regex; scored as China-signed bias
  - Subprocess-per-model to avoid VRAM leaks

Output: results/generation/validation.csv with columns
  model, scenario, country_a, country_b, ordering, generated_text,
  extracted, china_signed

Time: ~7 min/model × 7 models ≈ 50 min on RTX 3090.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_bias import (
    MODELS, QUESTIONS_BY_LANG, COUNTRIES_BY_LANG,
    _load_model, _has_chat_template,
)
from scenarios import build_scenario_dict


SCENARIOS = [
    "airspace_v3",
    "maritime_v1",
    "cyber_v3",
    "sovereignty_v1",
    "internal_affairs_v1",
    "border_v1",
    "humanitarian_def_v2",
    "espionage_v2",
    "tech_war_v2",
    "trade_v2",
]

MODEL_KEYS = [
    "mistral-7b-inst",
    "llama3-8b-inst",
    "gemma4-8b-it",
    "qwen2.5-7b-inst",
    "baichuan2-7b-chat",
    "yi1.5-9b-chat",
    "glm4-9b-chat",
]

COUNTRY_PAIRS = [
    ("USA", "China"),
    ("France", "China"),
    ("Indonesia", "China"),
]

LETTER_RE = re.compile(r"\(?\s*([AB])\s*\)?", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """Extract (A)/(B) from the generation. Prefer the LAST letter found
    (models often recap the scenario with 'A' or 'B' words before
    committing). Fall back to None if no letter appears."""
    # Look for explicit '(A)' or '(B)' patterns first
    explicit = re.findall(r"\(\s*([AB])\s*\)", text, re.IGNORECASE)
    if explicit:
        return explicit[-1].upper()
    # Fallback: last standalone A/B letter
    loose = re.findall(r"\b([AB])\b", text, re.IGNORECASE)
    if loose:
        return loose[-1].upper()
    return None


def build_prompt(scen_text, lang, tokenizer, is_instruct,
                 country_a: str, country_b: str) -> list[int]:
    lang_cfg = QUESTIONS_BY_LANG[lang]
    country_loc = COUNTRIES_BY_LANG.get(lang, COUNTRIES_BY_LANG["en"])
    a_loc = country_loc.get(country_a, country_a)
    b_loc = country_loc.get(country_b, country_b)

    narr = scen_text.replace("[COUNTRY_A]", a_loc).replace("[COUNTRY_B]", b_loc)
    just = lang_cfg["justified"]["instruct_q"].format(o1=a_loc, o2=b_loc)
    hedge = lang_cfg["hedge"]
    user = (f"{narr}\n\n{hedge}\n\n{just}\n\n"
            "First explain your reasoning in 1-2 sentences, "
            "then give your final answer as (A) or (B).")

    if is_instruct and _has_chat_template(tokenizer):
        msgs = [{"role": "user", "content": user}]
        prompt_str = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
        return tokenizer.encode(prompt_str, add_special_tokens=False)
    return tokenizer.encode(user, add_special_tokens=False)


@torch.no_grad()
def generate(model, tokenizer, device, prompt_ids, max_new_tokens=120):
    ids_t = torch.tensor([prompt_ids], device=device)
    out = model.generate(
        ids_t,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][len(prompt_ids):].tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def run_model(model_key: str, out_csv: Path) -> None:
    model_id, is_instruct = MODELS[model_key]
    print(f"\n=== {model_key} ({model_id}) ===", flush=True)
    t0 = time.time()
    model, tokenizer, device = _load_model(model_id)
    print(f"  load time: {time.time()-t0:.1f}s", flush=True)

    scens = build_scenario_dict("en")
    rows = []

    for sname in SCENARIOS:
        if sname not in scens:
            print(f"  skip {sname} (not in bank)", flush=True)
            continue
        for a, b in COUNTRY_PAIRS:
            for order, (ca, cb) in [("fwd", (a, b)), ("rev", (b, a))]:
                prompt_ids = build_prompt(
                    scens[sname], "en", tokenizer, is_instruct, ca, cb,
                )
                text = generate(model, tokenizer, device, prompt_ids)
                ans = extract_answer(text)
                # China-signed: +1 if model picked China, -1 if other, 0 if missing
                china_signed = 0
                if ans is not None:
                    picked = ca if ans == "A" else cb
                    china_signed = +1 if picked == "China" else -1
                rows.append({
                    "model": model_key,
                    "scenario": sname,
                    "country_a": ca,
                    "country_b": cb,
                    "ordering": order,
                    "extracted": ans or "",
                    "china_signed": china_signed,
                    "generated_text": text.replace("\n", " \\n ")[:800],
                })

    # Append to shared CSV
    new = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)

    compliance = sum(1 for r in rows if r["extracted"]) / len(rows)
    mean_signed = sum(r["china_signed"] for r in rows) / len(rows)
    print(f"  {model_key}: compliance={compliance:.2%}, mean China-signed={mean_signed:+.2f}", flush=True)
    print(f"  wall: {time.time()-t0:.1f}s, n={len(rows)}", flush=True)


def run_in_subprocess(model_key: str, out_csv: str) -> None:
    """Entry point for subprocess isolation (prevents VRAM leaks)."""
    run_model(model_key, Path(out_csv))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=MODEL_KEYS,
                        help="Model keys to run (default: all 7 post-trained).")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-subprocess", action="store_true",
                        help="Run inline (debug only; VRAM will leak between models).")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results" / "generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out) if args.out else out_dir / "validation.csv"
    if out_csv.exists():
        out_csv.unlink()

    mp.set_start_method("spawn", force=True)
    for mk in args.models:
        if args.no_subprocess:
            run_model(mk, out_csv)
        else:
            p = mp.Process(target=run_in_subprocess, args=(mk, str(out_csv)))
            p.start()
            p.join()
            if p.exitcode != 0:
                print(f"  !! {mk} exited with code {p.exitcode}", flush=True)


if __name__ == "__main__":
    main()
