"""
Step 3: full fine-tune of SmolLM2-135M with a hand-written training loop.
Gradient accumulation, cosine LR, bf16 autocast, loss masking, peak-memory report.

    pip install torch transformers datasets
    python 03_full_finetune_pytorch.py
"""
import math, time, torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import build_dataset

MODEL = "HuggingFaceTB/SmolLM2-135M"          # base model; we teach it to follow instructions
device = "cuda"
BATCH, ACCUM, LR, EPOCHS, WARMUP = 8, 4, 2e-5, 1, 20

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")   # reuse the chat template
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)  # fp32 master weights
model.gradient_checkpointing_enable()
model.config.use_cache = False

data = build_dataset(tok)

def collate(batch):
    L = max(len(b["input_ids"]) for b in batch)
    pad = lambda seq, v: seq + [v] * (L - len(seq))
    return {
        "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
        "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
    }

loader = DataLoader(data, batch_size=BATCH, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
total_steps = EPOCHS * len(loader) // ACCUM
lr_at = lambda s: LR * (s + 1) / WARMUP if s < WARMUP else LR * 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (total_steps - WARMUP)))

def sample(q):
    model.eval(); model.config.use_cache = True
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = tok(p, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=80, do_sample=False, pad_token_id=tok.pad_token_id)
    model.train(); model.config.use_cache = False
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

q = "Give me three tips for learning to code."
print("BEFORE:", sample(q), "\n")

model.train()
step, t0 = 0, time.time()
for epoch in range(EPOCHS):
    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / ACCUM      # scale so accumulated grads average correctly
        loss.backward()
        if (i + 1) % ACCUM == 0:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                print(f"step {step}/{total_steps} | loss {loss.item() * ACCUM:.3f} | gnorm {gn:.2f} | "
                      f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB | {time.time() - t0:.0f}s")

print("\nAFTER:", sample(q))
model.save_pretrained("out_pytorch"); tok.save_pretrained("out_pytorch")