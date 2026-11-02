"""Sanity-check the GPU stack. Run after installing requirements."""
import torch

print("torch:", torch.__version__, "| built for CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"gpu: {p.name} | compute capability: {p.major}.{p.minor}")
    print(f"vram free/total: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    x = torch.randn(1024, 1024, device="cuda")
    print("matmul ok:", (x @ x).shape)
    print("flash/sdpa backend available:", torch.backends.cuda.flash_sdp_enabled())
try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except Exception as e:
    print("bitsandbytes problem:", e)
import transformers, peft, trl, datasets
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| trl", trl.__version__, "| datasets", datasets.__version__)
