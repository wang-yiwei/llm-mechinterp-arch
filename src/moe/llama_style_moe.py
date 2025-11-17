#src/moe/llama_style_moe.py

# Author: Yiwei Wang
# Date: 2025-11-17



# Information about the script:
# This script implements a MoE FFN layer with the following features:


# Importing the necessary libraries
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# relative imports
from src.modern.llama_style_transformer import RMSNorm, SwiGLU, LlamaStyleAttention
