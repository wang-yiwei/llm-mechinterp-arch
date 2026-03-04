#! /usr/bin/env python

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

# Make src importable
ROOT = Path(__file__).resolve().parent[1]
sys.path.append(str(ROOT))
