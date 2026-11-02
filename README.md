# ftlab: LLM/SLM fine-tuning lab

    python -m venv .venv && source .venv/bin/activate
    pip install -U pip && pip install -r requirements.txt
    python scripts/check_env.py

Layout
- `src/ftlab/`   reusable code (LoRA, data utils, training loops, eval helpers)
- `stages/`      one folder per roadmap stage; run scripts from inside their stage folder
- `configs/`     YAML experiment configs
- `data/`        raw and processed datasets (git-ignored)
- `outputs/`     checkpoints, adapters, logs (git-ignored)
