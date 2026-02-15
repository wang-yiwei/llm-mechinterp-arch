# src/modern/engram.py

# Author: Yiwei Wang
# Date: 2026-02-15

# Information about the script:
# Minimal implementation of the Engram model proposed in "Engram: A Memory-based Transformer" by DeepSeek
# - Tokenizer + optional compression module
# - Prime moduli per head
# - N-Gram hashing (multi-head)
# - Lookup (embedding) separate from Gate and Fusion

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Importing the necessary libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from sympy import isprime
except ImportError:
    isprime = None

try:
    from transformers import AutoTokenizer
    from tokenizers import normalizers, Regex
except Exception: # pragma: no cover
    AutoTokenizer = None
    normalizers = None
    Regex = None


