# Practice is All You Need

Exploring emergent reasoning capabilities in small language models for math solving. We compare three approaches:

1. **Pure RL** — Reinforcement learning from scratch on a base model (R1-zero).
2. **SFT** — Supervised fine-tuning on reasoning traces from a strong model, applied to math questions on the same base model.
3. **SFT + RL (cold start)** — SFT on reasoning traces first, then applying RL (GRPO) to the fine-tuned model.

## Getting Started

1. Create and activate a virtual environment (e.g. `rlvr`).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
* If working on a lab computer, it is reccomended to use the venv from `TDDE09` instead to save memory:
   ```bash
   source /courses/TDDE09/venv/bin/activate
   ```
3. To use the kernel in Jupyter notebooks, run:
   ```bash
   python -m ipykernel install --user --name rlvr --display-name "rlvr"
   ```
   Then select "rlvr" as the kernel when opening notebooks.
