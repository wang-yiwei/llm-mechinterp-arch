# src/data/data.py

# Author: Yiwei Wang
# Date: 2026-03-04

# Information about the script:
# This script implements the data loading for the model.

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
