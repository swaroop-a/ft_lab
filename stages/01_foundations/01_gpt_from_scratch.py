"""
Step 1: a GPT written from scratch in plain PyTorch (~10M params), trained on
tiny-shakespeare at character level. Goal: understand every tensor shape in a
decoder-only transformer. Runs in ~5-10 min on a GPU, uses < 3 GB VRAM.

    python 01_gpt_from_scratch.py
"""
import math, os, time, urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 65
    block_size: int = 256   # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head, self.dropout = c.n_head, c.dropout
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape                       # batch, time, channels
        q, k, v = self.qkv(x).split(C, dim=2)   # each (B, T, C)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # softmax(QK^T / sqrt(d)) V with a causal mask, fused (FlashAttention when available)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble heads
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.down = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block with residual connections."""
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln2, self.mlp = nn.LayerNorm(c.n_embd), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)   # learned positions (RoPE comes later)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight             # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():                  # GPT-2 style residual scaling
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.ln_f(x))                   # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    # ---- data: char-level tokenizer ----
    path = "data/shakespeare.txt"
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(chars[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = Config(vocab_size=len(chars))
    batch_size, max_iters, eval_every = 64, 3000, 250
    lr, min_lr, warmup = 1e-3, 1e-4, 100

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - cfg.block_size - 1, (batch_size,))
        x = torch.stack([d[i:i + cfg.block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + cfg.block_size] for i in ix])   # shifted by one = next-token targets
        return x.to(device), y.to(device)

    model = GPT(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(it):  # linear warmup + cosine decay
        if it < warmup:
            return lr * (it + 1) / warmup
        p = (it - warmup) / (max_iters - warmup)
        return min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (lr - min_lr)

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split in ("train", "val"):
            out[split] = sum(model(*get_batch(split))[1].item() for _ in range(20)) / 20
        model.train()
        return out

    amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda"))
    t0 = time.time()
    for it in range(max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if it % eval_every == 0:
            l = evaluate()
            print(f"step {it:4d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time() - t0:.0f}s")
        x, y = get_batch("train")
        with amp:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- sample ---")
    print(decode(model.generate(ctx, 500, temperature=0.8, top_k=40)[0].tolist()))


if __name__ == "__main__":
    main()