"""Shared data prep: Alpaca-cleaned -> chat template -> tokenized with prompt tokens masked (-100)."""
from datasets import load_dataset


def build_dataset(tok, n=5000, max_len=512):
    ds = load_dataset("yahma/alpaca-cleaned", split=f"train[:{n}]")
    out = []
    for ex in ds:
        user = ex["instruction"] + (f"\n\n{ex['input']}" if ex["input"] else "")
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        full = prompt + ex["output"] + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
        # loss only on the assistant's answer: mask the prompt with -100
        labels = ([-100] * len(p_ids) + f_ids[len(p_ids):])[:max_len]
        if len(p_ids) >= len(f_ids):   # prompt filled the whole window, nothing to learn
            continue
        out.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels})
    return out