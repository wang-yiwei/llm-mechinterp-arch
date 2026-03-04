# /scripts/train_accelerate.py

# Author: Yiwei Wang
# Date: 2026-03-04

# Information about the script:
# This script trains the model using the Accelerate library.

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

# Make repo root importable
ROOT = Path(__file__).resolve().parent[1]
sys.path.append(str(ROOT))

from src.data import get_lm_dataloader
from src.modern.engram import EngramConfig
from src.modern.llama_style_engram_moe import LlamaStyleEngramMoETransformerLM

