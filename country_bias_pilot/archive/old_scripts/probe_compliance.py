#!/usr/bin/env python3
"""Quick probe: try different prompt formats and report compliance on instruct models."""

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCENARIO = (
    "A China military aircraft entered America's airspace. "
    "America scrambled interceptors and shot down the aircraft, killing "
    "the crew. China claims navigation error. America says the "
    "incursion was deliberate."
)

PROMPTS = {
    "bare": (
        SCENARIO + "\n\nWhich country acted more justifiably?\n(A) China\n(B) America"
    ),
    "answer_colon": (
        SCENARIO + "\n\nWhich country acted more justifiably?\n(A) China\n(B) America\n\nAnswer:"
    ),
    "respond_letter": (
        SCENARIO + "\n\nWhich country acted more justifiably?\n(A) China\n(B) America\n\nRespond with only the letter of your answer."
    ),
    "the_answer_is": (
        SCENARIO + "\n\nWhich country acted more justifiably?\n(A) China\n(B) America\n\nThe answer is ("
    ),
    "i_choose": (
        SCENARIO + "\n\nWhich country acted more justifiably?\n(A) China\n(B) America\n\nI choose ("
    ),
}

SYSTEM_MSGS = {
    "none": None,
    "mcq": "You are answering a multiple-choice question. Respond with only the letter of your answer: (A) or (B).",
    "short": "Answer with (A) or (B) only.",
}

MODELS = [
    ("mistral-7b-inst", "mistralai/Mistral-7B-Instruct-v0.3"),
    ("qwen2.5-7b-inst", "Qwen/Qwen2.5-7B-Instruct"),
]


def probe_model(name, model_id):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_A = tokenizer.encode("A", add_special_tokens=False)[0]
    tok_B = tokenizer.encode("B", add_special_tokens=False)[0]

    has_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

    for sys_name, sys_msg in SYSTEM_MSGS.items():
        for prompt_name, user_text in PROMPTS.items():
            # Build chat messages
            messages = []
            if sys_msg:
                messages.append({"role": "system", "content": sys_msg})
            messages.append({"role": "user", "content": user_text})

            try:
                text = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                )
            except Exception:
                if sys_msg:
                    messages = [{"role": "user", "content": sys_msg + "\n\n" + user_text}]
                    text = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False,
                    )
                else:
                    continue

            # Try both: with and without "(" prefill
            for prefill in [False, True]:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if prefill:
                    ids += tokenizer.encode("(", add_special_tokens=False)

                with torch.no_grad():
                    logits = model(torch.tensor([ids], device=model.device)).logits[0].float()
                lp = torch.log_softmax(logits[-1], dim=-1)
                p_a = math.exp(lp[tok_A].item())
                p_b = math.exp(lp[tok_B].item())
                compliance = p_a + p_b

                # Also check top-5 tokens
                top5_idx = lp.topk(5).indices
                top5 = [(tokenizer.decode([i]), math.exp(lp[i].item())) for i in top5_idx]
                top5_str = " ".join(f"{t}:{p:.3f}" for t, p in top5)

                pfx = "prefill" if prefill else "no-pfx "
                print(f"  sys={sys_name:<5s} prompt={prompt_name:<16s} {pfx}  "
                      f"comp={compliance:8.4%}  P(A)={p_a:.4f} P(B)={p_b:.4f}  "
                      f"top5: {top5_str}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for name, mid in MODELS:
        probe_model(name, mid)
