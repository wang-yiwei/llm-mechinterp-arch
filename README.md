# mechinterp-llm-architectures

A small, educational collection of **minimal PyTorch implementations** of
Transformer / LLM architectures, designed as a learning roadmap for
mechanistic interpretability.

The goal is **not** to be fast or production-ready, but to:
- mirror the math in papers / slides,
- keep the code short and readable,
- show how these pieces map to real HuggingFace models.

## Contents

### 1. Basics: vanilla Transformer LM

`src/basics/`

- `tiny_transformer.py`  
  A small decoder-only Transformer language model with:
  - token + positional embeddings,
  - multi-head self-attention,
  - LayerNorm + residual connections,
  - 2-layer FFN with GELU,
  - tied embedding / unembedding.

This corresponds closely to the "standard Transformer" refresher in most
mechinterp tutorials.

### 2. LLaMA-style blocks

`src/llama_style/`

- `llama_block.py`  
  A single Transformer block with:
  - RMSNorm instead of LayerNorm,
  - Rotary Positional Embeddings (RoPE) inside attention,
  - SwiGLU-style FFN.

- `rope.py`  
  Minimal RoPE implementation for Q/K projections.

### 3. Mixture-of-Experts (MoE)

`src/moe/`

- `sparse_moe_ffn.py`  
  - `ExpertFFN`: FFN expert with SwiGLU gating.
  - `SparseMoE`: top-k sparse router over multiple experts,
    with a simple load-balancing loss helper.

- `moe_transformer_block.py`  
  Transformer block where the FFN is replaced by `SparseMoE`, similar
  in structure to Mixtral / Qwen2MoE-style MoE layers.

### 4. HuggingFace inspection helpers

`src/hf_wrappers/`

- `inspect_llama.py`  
  Shows how the LLaMA-style block maps to a real HF model:
  - where `W_E` / `W_U` live,
  - where RMSNorm, RoPE, and SwiGLU live.

- `inspect_moe.py`  
  Shows how the MoE FFN maps to a real MoE model (e.g. Mixtral
  and/or Qwen2MoE):
  - router (gate) weights,
  - number of experts, top-k,
  - how to access router logits.

---

## Getting started

### 1. Clone & create environment

```bash
git clone git@github.com:<your-user>/mechinterp-llm-architectures.git
cd mechinterp-llm-architectures

python -m venv .venv  # or use conda  conda create --name llm-mechinterp python=3.12.7

source .venv/bin/activate   

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run a tiny Transformer example

This will just create a toy vocab, sample random token IDs, and run them through the TinyTransformerLM: `python scripts/run_tiny_transformer.py`.

Expect something like:
- printed tensor shapes (batch, seq_len, vocab_size),
- maybe a small dump of the next-token distribution.


### 3. Run the LLaMA-style block

If you have access to a LLaMA-like model on HuggingFace:

```bash
python scripts/run_llama_style.py \
  --model_id meta-llama/Llama-2-7b-hf \
  --prompt "Mechanistic interpretability is"
```

This script can:

- tokenize the prompt,
- run the HF model with output_hidden_states=True,
- print which HF components correspond to:
    - embeddings (W_E),
    - unembedding (W_U),
    - RMSNorm,
    - RoPE application,
    - FFN weights.


### 4. Run a MoE example

For the toy MoE block: `python scripts/run_moe_example.py`

That script can, for example:
- create a SparseMoE with `num_experts=8`, `top_k=2`,
- feed random inputs or a real hidden state,
- print router probabilities and show which experts are selected
for which tokens.

For a real MoE model (e.g. Mixtral):

```bash
python src/hf_wrappers/inspect_moe.py \
  --model_id mistralai/Mixtral-8x7B-Instruct \
  --prompt "Mixture-of-Experts models route tokens to different experts"
```


### 5. Adding a new architecture

When you add a new model variant (e.g. Qwen2-style, Mamba, etc.), try to:

1. Put the code under a new folder in `src/`: such as `src/qwen_style/`, `src/mamba/`, etc.
2. Add a short `README` section describing:
    - what changes compared to the vanilla Transformer,
    - where those changes live (norm, attention, FFN, routing, etc.).

3. (Optional) add a small script in `scripts/` that:
    - instantiates the model,
    - runs a forward pass,
    - prints key tensor shapes.

This keeps the repo as a library of “mechinterp-friendly toy architectures” you can reference later.