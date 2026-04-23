"""Score free-gen commit-position log-odds.

For each existing free-gen row, reconstruct the prompt + reasoning-up-to-commit,
forward pass, compute log P(A-variants) − log P(B-variants). Forward/reverse
orderings cross-map to a c1-vs-c2 frame (same convention as forced-choice).
"""
from __future__ import annotations
import argparse
import csv
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, "/home/stuart/country_bias_pilot")
from gpu_bias import MODELS, QUESTIONS_BY_LANG, COUNTRIES_BY_LANG, _load_model, _has_chat_template
from scenarios import build_scenario_dict

# Mirror free_gen_validation build_prompt so we get the same prompt as generated.
def build_base_prompt(scen_text, lang, tokenizer, is_instruct,
                      country_a: str, country_b: str, model=None,
                      model_key: str = "") -> list[int]:
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
    if "baichuan" in model_key and model is not None:
        mod_names = [k for k in sys.modules
                     if "generation_utils" in k and "aichuan" in k]
        if not mod_names:
            raise RuntimeError("Baichuan build_chat_input missing")
        build_chat_input = sys.modules[mod_names[0]].build_chat_input
        msgs = [{"role": "user", "content": user}]
        return build_chat_input(model, tokenizer, msgs)[0].tolist()
    if is_instruct and _has_chat_template(tokenizer):
        msgs = [{"role": "user", "content": user}]
        s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(s, add_special_tokens=False)
    return tokenizer.encode(user, add_special_tokens=False)


def encode_letter_variants(tokenizer, letter: str) -> list[int]:
    """All single-token IDs that decode to letter (with leading space / paren variants)."""
    candidates = [letter, " " + letter, "(" + letter, "\n" + letter]
    ids = set()
    for c in candidates:
        try:
            for tid in tokenizer.encode(c, add_special_tokens=False):
                s = tokenizer.decode([tid])
                if s.strip().upper() == letter:
                    ids.add(tid)
        except Exception:
            pass
    return sorted(ids)


@torch.no_grad()
def score_commit_logodds(model, tokenizer, device, full_prefix_ids,
                         tok_A, tok_B):
    """Forward pass on prefix; return (logp_A_sum, logp_B_sum) at last position."""
    ids_t = torch.tensor([full_prefix_ids], device=device)
    out = model(ids_t, use_cache=False)
    logits = out.logits[0, -1, :]
    logp = torch.log_softmax(logits, dim=-1)
    # logsumexp over the variant IDs
    if tok_A:
        lA = torch.logsumexp(logp[torch.tensor(tok_A, device=device)], dim=0).item()
    else:
        lA = float("-inf")
    if tok_B:
        lB = torch.logsumexp(logp[torch.tensor(tok_B, device=device)], dim=0).item()
    else:
        lB = float("-inf")
    return lA, lB


def find_commit_prefix(generated_text: str, extracted_letter: str):
    """Return generated text up to (but not including) the commit letter's position.
    Returns None if the commit letter cannot be located."""
    if not extracted_letter: return None
    # Normalize newline encoding used in the CSV
    txt = generated_text.replace(" \\n ", "\n")
    # Find last occurrence of "(X)" or standalone letter at end
    m = list(re.finditer(rf"\(?\s*({extracted_letter})\s*\)?", txt, re.IGNORECASE))
    if not m: return None
    last = m[-1]
    # Cut right before the letter itself
    # last.span(1) is the letter's span
    letter_start = last.start(1)
    return txt[:letter_start]


def run_model(model_key: str, rows: list[dict], out_path: Path):
    model_id, is_instruct = MODELS[model_key]
    print(f"\n=== {model_key} ===", flush=True)
    t0 = time.time()
    model, tokenizer, device = _load_model(model_id)
    print(f"  load: {time.time()-t0:.1f}s", flush=True)

    tok_A = encode_letter_variants(tokenizer, "A")
    tok_B = encode_letter_variants(tokenizer, "B")
    print(f"  A variants: {tok_A}, B variants: {tok_B}", flush=True)

    scens = build_scenario_dict("en")
    out_rows = []

    for r in rows:
        sname = r["scenario"]
        ca, cb = r["country_a"], r["country_b"]
        ordering = r["ordering"]
        extracted = r["extracted"]
        if sname not in scens or not extracted:
            continue
        prefix = find_commit_prefix(r["generated_text"], extracted)
        if prefix is None:
            continue

        base_ids = build_base_prompt(
            scens[sname], "en", tokenizer, is_instruct, ca, cb,
            model=model, model_key=model_key,
        )
        cont_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = base_ids + cont_ids
        try:
            lA, lB = score_commit_logodds(model, tokenizer, device, full_ids, tok_A, tok_B)
        except Exception as e:
            print(f"  skip {sname}/{ca}v{cb}/{ordering}: {e}", flush=True)
            continue
        log_odds = lA - lB  # positive = model prefers letter A at commit
        out_rows.append({
            "model": model_key, "scenario": sname,
            "country_a": ca, "country_b": cb, "ordering": ordering,
            "extracted": extracted, "log_odds_A_B": log_odds,
        })

    # Aggregate: group by (scenario, {ca,cb}) and average fwd/rev
    # China-signed: for each pair, project to China-vs-other frame
    paired = defaultdict(dict)
    for r in out_rows:
        key = (r["scenario"], frozenset([r["country_a"], r["country_b"]]))
        # In fwd: (ca, cb); A-token refers to ca. log_odds_A_B = log P(ca preferred) - log P(cb preferred).
        # To get China-signed: if ca=China, log_odds_c1=log_odds_A_B where c1=China; else negate.
        if r["country_a"] == "China":
            china_signed = r["log_odds_A_B"]
        elif r["country_b"] == "China":
            china_signed = -r["log_odds_A_B"]
        else:
            continue
        paired[key][r["ordering"]] = china_signed

    per_pair_avg = []
    for key, by_ord in paired.items():
        if "fwd" in by_ord and "rev" in by_ord:
            per_pair_avg.append((by_ord["fwd"] + by_ord["rev"]) / 2)
        elif by_ord:
            per_pair_avg.append(list(by_ord.values())[0])

    mean_logodds = sum(per_pair_avg) / len(per_pair_avg) if per_pair_avg else 0.0
    print(f"  {model_key}: n_pairs={len(per_pair_avg)}, "
          f"mean China-signed log-odds (post-reasoning) = {mean_logodds:+.3f}", flush=True)

    # Append to shared CSV
    new = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        if new: w.writeheader()
        w.writerows(out_rows)


def run_sub(model_key, rows, out_path):
    run_model(model_key, rows, Path(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-csvs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    rows_by_model = defaultdict(list)
    for p in args.validation_csvs:
        with open(p) as f:
            for r in csv.DictReader(f):
                rows_by_model[r["model"]].append(r)

    models = args.models or list(rows_by_model.keys())

    out_path = Path(args.out)
    if out_path.exists(): out_path.unlink()

    mp.set_start_method("spawn", force=True)
    for mk in models:
        if mk not in rows_by_model:
            print(f"skip {mk}: no rows")
            continue
        p = mp.Process(target=run_sub, args=(mk, rows_by_model[mk], str(out_path)))
        p.start(); p.join()
        if p.exitcode != 0:
            print(f"!! {mk} exited {p.exitcode}")


if __name__ == "__main__":
    main()
