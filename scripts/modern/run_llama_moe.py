#!/usr/bin/env python3

# Author: Yiwei Wang
# Date: 2026-02-12

# Information about the script:
# This script runs the LlamaStyleMoETransformerLM model with the following features:
# - RoPE applied to Q and K
# - SwiGLU feed-forward network
# - Optional grouped-query attention (GQA) via n_kv_heads < n_heads
# - MoE FFN with top-k sparse router and load-balancing loss
# - Optional MoE FFN layers at specified indices, which can be chosen to be used as the FFN layer for the Transformer blocks, by passing `moe_layer_indices`

import argparse
from pathlib import Path
import sys

import torch

# make repo root path available to the script
ROOT = Path(__file__).resolve().parent[2]
sys.path.append(str(ROOT))

from src.modern.llama_style_moe import LlamaStyleMoETransformerLM


def main():
    parser = argparse.ArgumentParser(
        description="Run a Llama-style MoE Transformer Language Model",
    )
    