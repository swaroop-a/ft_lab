"""
Step 4: the same fine-tune using HF Trainer. Compare against 03: same loss curve, far less code.

    python 04_full_finetune_trainer.py
"""
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)
from data import build_dataset

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", torch_dtype=torch.float32)

args = TrainingArguments(
    output_dir="out_trainer",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    num_train_epochs=1,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)
trainer = Trainer(
    model=model, args=args, train_dataset=build_dataset(tok),
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
)
trainer.train()
trainer.save_model("out_trainer"); tok.save_pretrained("out_trainer")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")