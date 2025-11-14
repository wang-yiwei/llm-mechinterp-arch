#src/modern/llama_style_transformer.py

# Author: Yiwei Huang
# Date: 2025-11-14



# Information about the script:
# This script implements a MoE FFN layer with the following features:
# - ExpertFFN: SwiGLU-like FFN (LLaMA/Qwen-ish).
# - SparseMoE: top-k routing over experts (dense compute version for clarity).
# - moe_load_balancing_loss: simple aux loss encouraging uniform expert usage.