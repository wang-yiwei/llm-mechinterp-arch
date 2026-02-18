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


# --------------------
# Tokenizer + optional compression module (DeepSeek's implementation)
# --------------------

class CharTokenizer:
    """
    Offline fallback tokenizer (character-level)
    Educational purpose only, not ideal for production-level LLMs, but demo runnable without HF dependencies
    """

    def __init__(
        self,
        extra_chars: str = "",
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",        
    ) -> None:
        self.extra_chars = extra_chars
        self.pad_token = pad_token
        self.unk_token = unk_token
        base = [pad_token, unk_token]
        chars = sorted(set[str](list(extra_chars)))
        self.itos = base + chars
        self.stoi = {s: i for i,s in enumerate(self.itos)}

    @property
    def pad_token_id(self) -> int:
        return self.stoi[self.pad_token]
    
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def train_from_text(self, text: str) -> None:
        # build vocab from text characters
        chars = sorted(set(list(text)))
        base = [self.pad_token, self.unk_token]
        self.itos = base + chars
        self.stoi = {s: i for i,s in enumerate(self.itos)}
    
    def encode(self, text: str) -> List[int]:
        return [
            self.stoi.get(ch, self.stoi[self.unk_token])
            for ch in text
        ]
    
    def decode(self, ids: List[int]) -> str:
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
            else:
                out.append(self.unk_token)
        return "".join(out)


class CompressTokenizer:
    """
    DeepSeek-style token-ID compression:
    - decode each token id -> normalise text (NFKC/NFD/StripAccents/Lowercase/whitespace) -> merge duplicates
    - build lookup table: old_id -> new_id
    
    This works only for HF-style tokenizers where decode(id) is meaningful.
    """
    def __init__(
        self,
        tokenizer_name_or_path: str,
        trust_remote_code: bool = False,
        pad_token_id: Optional[int] = None,
    ) -> None:
        
        if AutoTokenizer is None or normalizers is None or Regex is None:
            raise ImportError(
                "Need transformers, tokenizers, and sympy to use CompressTokenizer"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])

        self.lookup_table, self.num_new_tokens = self._build_lookup_table()
        self.lookup_table_torch = torch.tensor(
            self.lookup_table,
            dtype=torch.long,
        )
    
        if pad_token_id is not None:
            # map HF pad_token_id to new pad_token_id
            self.pad_token_id = int(self.lookup_table[pad_token_id])
        else:
            self.pad_token_id = None
        
        def __len__(self) -> int:
            return len(self.num_new_tokens)

        def _build_lookup_table(self):
            import numpy as np

            old_to_new: Dict[int, int] = {}
            key_to_new: Dict[str, int] = {}
            new_tokens: List[str] = []

            vocab_size = len(self.tokenizer)
            for old_id in range(vocab_size):
                text = self.tokenizer.decode(
                    [old_id],
                    skip_special_tokens=False,
                )
                # deepseek employs a special handling if decode contains "�"
                if "�" in text:
                    key = self.tokenizer.convert_ids_to_tokens(old_id)
                else:
                    norm = self.normalizer.normalize_str(text)
                    key = norm if norm else text
                
                new_id = key_to_new.get(key)
                if new_id is None:
                    new_id = len(new_tokens)
                    key_to_new[key] = new_id
                    new_tokens.append(key)
                old_to_new[old_id] = new_id
            
            lookup = np.empty(vocab_size, dtype=np.int64)
            for old_id in range(vocab_size):
                lookup[old_id] = old_to_new[old_id]
            
            return lookup, len(new_tokens)

        def compress(
            self,
            input_ids: torch.Tensor,
        ) -> torch.Tensor:
            """
            input_ids: (batch, seq_len)
            return: (batch, seq_len), compressed input_ids of the same shape
            """
            lut = self.lookup_table_torch.to(device=input_ids.device)
            return lut[input_ids]


# --------------------
# Prime moduli helper (per head, per n-gram, per layer)
# --------------------

def _find_next_prime(
    start: int,
    seen: set,
) -> int:
    # uses hash functions with prime numbers to avoid collisions
    # like assigning each phrase a unique locker number in a massive library
    if isprime is None:
        # fallback: start+1, not ideal but keeps the code running
        return start + 1
    cand = start + 1
    while True:
        if isprime(cand) and cand not in seen:
            return cand
        cand += 1


def build_prime_moduli(
    layer_ids: List[int],
    vocab_size_per_ngram: List[int],
    max_ngram_size: int,
    n_head_per_ngram: int,
) -> Dict[int, List[List[int]]]:
    """
    Build prime moduli for each layer, n-gram, and head.
    Args:
        layer_ids: List of layer indices
        vocab_size_per_ngram: List of vocabulary sizes per n-gram
        max_ngram_size: Maximum n-gram size
        n_head_per_ngram: Number of heads per n-gram
    Returns:
        moduli[layer_id][ngram_index][head_index] (ngram_index: 0 for 2-gram, 1 for 3-gram, ...)
        Each head uses a unique prime modulus slightly above the nominal vocabulary size for the n-gram.
    """
    assert len(vocab_size_per_ngram) == (max_ngram_size - 1)
    seen_primes = set()
    moduli: Dict[int, List[List[int]]] = {}
    for layer_id in layer_ids:
        per_layer: List[List[int]] = []
        for ngram in range(2, max_ngram_size + 1):
            base_vocab = int(vocab_size_per_ngram[ngram -2])
            primes_for_heads: List[int] = []
            start = base_vocab - 1
            for _ in range(n_head_per_ngram):
                p = _find_next_prime(start, seen_primes)
                seen_primes.add(p)
                primes_for_heads.append(p)
                start = p
            per_layer.append(primes_for_heads)
        moduli[layer_id] = per_layer
    return moduli


# --------------------
# N-Gram hashing (multi-head in torch version)
# --------------------

@dataclass
class EngramConfig:
    # where to inject engram tokens (0: before Q, 1: after Q, 2: after K, 3: after V)
    layer_ids: List[int] = field(default_factory=lambda: [1])

    # n-gram config
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    n_embed_per_ngram: int = 512
    vocab_size_per_ngram: List[int] = field(default_factory=lambda: [200_000, 200_000])

    # tokenizer & compression
    tokenizer_type: str = "hf"
    tokenizer_name_or_path: str = "meta-llama/Llama-3.1-8B"
    trust_remote_code: bool = False
    pad_token_id: Optional[int] = 0
    use_compressed_tokenizer: bool = True

    # hashing seeds
    seed: int = 88
    prime_seed_constant: int = 10007

    # fusion
    use_conv: bool = True
    conv_kernel_size: int = 4

    # gating transform
    gate_signed_sqrt: bool = True


class NgramHashMapping(nn.Module):
        