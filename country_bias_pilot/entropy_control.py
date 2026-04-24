"""Entropy-matched control for §5.2 amplification claim.

For each (model, scenario, country-pair, ordering) in validation.csv,
score the commit-position log-odds when the model is fed a FIXED
neutral-filler 'reasoning' instead of its own generated reasoning.

If Gemma/GLM still produce sharp anti-China logits under neutral
filler (matching the -4.73 / -2.40 we saw post-reasoning), then the
sharpness is architectural/context-length, not preference-driven.

If their neutral-filler logits collapse toward forced-choice values
(-1.15 / -0.13), the amplification is driven by the model's own
reasoning content — the paper-friendly reading.
"""
from __future__ import annotations
import argparse, csv, multiprocessing as mp, sys, time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, "/home/stuart/country_bias_pilot")
from gpu_bias import MODELS, QUESTIONS_BY_LANG, COUNTRIES_BY_LANG, _load_model, _has_chat_template
from scenarios import build_scenario_dict

NEUTRAL_FILLER = (
    "Both countries have valid perspectives and legitimate interests in this "
    "situation. The matter is complex with multiple factors to consider on "
    "each side. Weighing all of the relevant considerations carefully, "
    "my answer is ("
)


def build_base_prompt(scen_text, lang, tokenizer, is_instruct,
                      country_a, country_b, model=None, model_key=""):
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
        return sys.modules[mod_names[0]].build_chat_input(
            model, tokenizer, [{"role":"user","content":user}])[0].tolist()
    if is_instruct and _has_chat_template(tokenizer):
        msgs = [{"role": "user", "content": user}]
        s = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(s, add_special_tokens=False)
    return tokenizer.encode(user, add_special_tokens=False)


def encode_letter_variants(tokenizer, letter):
    cands = [letter, " " + letter, "(" + letter, "\n" + letter]
    ids = set()
    for c in cands:
        try:
            for tid in tokenizer.encode(c, add_special_tokens=False):
                if tokenizer.decode([tid]).strip().upper() == letter:
                    ids.add(tid)
        except Exception:
            pass
    return sorted(ids)


@torch.no_grad()
def score_commit(model, tokenizer, device, prefix_ids, tok_A, tok_B):
    ids_t = torch.tensor([prefix_ids], device=device)
    out = model(ids_t, use_cache=False)
    logp = torch.log_softmax(out.logits[0, -1, :], dim=-1)
    lA = torch.logsumexp(logp[torch.tensor(tok_A, device=device)], dim=0).item() if tok_A else float("-inf")
    lB = torch.logsumexp(logp[torch.tensor(tok_B, device=device)], dim=0).item() if tok_B else float("-inf")
    return lA, lB


def run_model(model_key, rows, out_path):
    model_id, is_instruct = MODELS[model_key]
    print(f"\n=== {model_key} ===", flush=True)
    t0 = time.time()
    model, tokenizer, device = _load_model(model_id)
    print(f"  load: {time.time()-t0:.1f}s", flush=True)

    tok_A = encode_letter_variants(tokenizer, "A")
    tok_B = encode_letter_variants(tokenizer, "B")

    scens = build_scenario_dict("en")
    # Tokenize the neutral filler once per tokenizer (no special tokens)
    filler_ids = tokenizer.encode(NEUTRAL_FILLER, add_special_tokens=False)

    out_rows = []
    for r in rows:
        sname = r["scenario"]
        ca, cb = r["country_a"], r["country_b"]
        if sname not in scens: continue
        base = build_base_prompt(scens[sname], "en", tokenizer, is_instruct,
                                  ca, cb, model=model, model_key=model_key)
        full = base + filler_ids
        try:
            lA, lB = score_commit(model, tokenizer, device, full, tok_A, tok_B)
        except Exception as e:
            print(f"  skip {sname}/{ca}v{cb}/{r['ordering']}: {e}", flush=True)
            continue
        out_rows.append({
            "model": model_key, "scenario": sname,
            "country_a": ca, "country_b": cb, "ordering": r["ordering"],
            "log_odds_A_B": lA - lB,
        })

    # Aggregate China-signed per (scenario, pair) averaging fwd/rev
    paired = defaultdict(dict)
    for r in out_rows:
        key = (r["scenario"], frozenset([r["country_a"], r["country_b"]]))
        if r["country_a"] == "China": cs = r["log_odds_A_B"]
        elif r["country_b"] == "China": cs = -r["log_odds_A_B"]
        else: continue
        paired[key][r["ordering"]] = cs
    vals = []
    for _, by_ord in paired.items():
        if "fwd" in by_ord and "rev" in by_ord:
            vals.append((by_ord["fwd"] + by_ord["rev"]) / 2)
        elif by_ord:
            vals.append(list(by_ord.values())[0])
    m = sum(vals) / len(vals) if vals else 0.0
    print(f"  {model_key}: n_pairs={len(vals)}, "
          f"China-signed log-odds (neutral filler) = {m:+.3f}", flush=True)

    new = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        if new: w.writeheader()
        w.writerows(out_rows)


def run_sub(mk, rows, p): run_model(mk, rows, Path(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation-csvs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

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
            print(f"skip {mk}: no rows"); continue
        p = mp.Process(target=run_sub, args=(mk, rows_by_model[mk], str(out_path)))
        p.start(); p.join()
        if p.exitcode != 0:
            print(f"!! {mk} exited {p.exitcode}")


if __name__ == "__main__":
    main()
