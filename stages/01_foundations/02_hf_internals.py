"""
Step 2: poke at a real small LM through Hugging Face transformers.
Tokenizer, chat template, config, parameter/memory accounting, loss, generation.

    pip install torch transformers
    python 02_hf_internals.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)

# ---- 1. config: this is the architecture, as data ----
print(model.config, "\n")

# ---- 2. parameters and where they live ----
total = sum(p.numel() for p in model.parameters())
print(f"total params: {total / 1e6:.1f}M")
for name, p in list(model.named_parameters())[:12]:
    print(f"  {name:55s} {tuple(p.shape)}")
print("  ...")
print("embedding tied to lm_head:", model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr())

# ---- 3. memory math: why full fine-tuning gets expensive ----
# weights(bf16)=2B + grads=2B (or fp32 4B) + AdamW states (2 x fp32 = 8B) + fp32 master copy 4B ~ 16B/param
print(f"\nweights only (bf16): {total * 2 / 1e9:.2f} GB")
print(f"full FT estimate (~16 B/param, excl. activations): {total * 16 / 1e9:.2f} GB")
print(f"same estimate for a 7B model: {7e9 * 16 / 1e9:.0f} GB  <- why LoRA/QLoRA exist\n")

# ---- 4. tokenizer + chat template ----
msgs = [{"role": "user", "content": "What is 2+2?"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print("chat template output:\n" + prompt)
ids = tok(prompt, return_tensors="pt").to(device)
print("token ids:", ids["input_ids"][0].tolist())
print("tokens   :", tok.convert_ids_to_tokens(ids["input_ids"][0]), "\n")

# ---- 5. the loss, by hand ----
text = "The capital of France is Paris."
batch = tok(text, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**batch, labels=batch["input_ids"])
    logits = out.logits.float()
# HF shifts internally: position t predicts token t+1
manual = torch.nn.functional.cross_entropy(logits[0, :-1], batch["input_ids"][0, 1:])
print(f"HF loss: {out.loss.item():.4f} | manual loss: {manual.item():.4f} | ppl: {manual.exp().item():.1f}")

# ---- 6. generation ----
gen = model.generate(**ids, max_new_tokens=40, do_sample=False)
print("\ngeneration:", tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))