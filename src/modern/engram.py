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
from turtle import forward
from typing import Dict, List, Optional, Tuple


# Importing the necessary libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.training_args import OptimizerNames


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

# Moduli is the prime numbers for each head per ngram per layer,
# while multipliers are the odd integers for each ngram per layer.
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
    """
    Produces hash ids:
        hash_ids[layer_id]: (batch, seq_len, head_total_size)
    where head_total_size = (max_ngram_size - 1) * n_head_per_ngram

    Core logic mirrors the implementation in the demo:
      - layer-specific odd multipliers (2, 3, 5, 7, 11, 13, 17, 19)
      - shift_k padding with pad_id 
      - XOR mixing
      - per-head mod with primes-per-head    
    """
    def __init__(self, cfg: EngramConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.max_ngram_size >= 2
        assert len (cfg.vocab_size_per_ngram) == (cfg.max_ngram_size - 1)

        # tokenizer & compressor
        self.char_tokenizer: Optional[CharTokenizer] = None
        self.compressed: Optional[CompressTokenizer] = None

        if cfg.tokenizer_type == "char":
            self.char_tokenizer = CharTokenizer()
            self.tokenizer_vocab_size = None
            self.pad_token_id = cfg.pad_token_id
        else:
            # HF tokenizer (optionally compressed)
            if cfg.use_compressed_tokenizer:
                self.compressed = CompressTokenizer(
                    tokenizer_name_or_path=cfg.tokenizer_name_or_path,
                    trust_remote_code=cfg.trust_remote_code,
                    pad_token_id=cfg.pad_token_id,
                )
                self.tokenizer_vocab_size = len(self.compressed)
                self.pad_token_id = int(self.compressed.pad_token_id) if self.compressed.pad_token_id is not None else cfg.pad_token_id
            else:
                # uncompressed HF tokenizer
                if AutoTokenizer is None:
                    raise ImportError("Need transformers to use HF tokenizer.")
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg.tokenizer_name_or_path,
                    trust_remote_code=cfg.trust_remote_code,
                )
                self.tokenizer_vocab_size = len(tokenizer)
                self.pad_token_id = cfg.pad_token_id
        
        # layer-specific multipliers (odd int64s)
        self.layer_multipliers: Dict[int, torch.Tensor] = {}
        for layer_id in cfg.layer_ids:
            base_seed = int(cfg.seed + cfg.prime_seed_constant * int(layer_id))
            g = torch.Generator()
            g.manual_seed(base_seed)
            # generate random multipliers, then force odd integers
            r = torch.randint(
                low=0,
                high=2**30,
                size=(cfg.max_ngram_size,),
                generator=g,
                dtype=torch.int64,
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[int(layer_id)] = multipliers

        
        # prime moduli per head per ngram per layer - layer_id -> ngram_index -> head_index
        moduli = build_prime_moduli(
            layer_ids=cfg.layer_ids,
            vocab_size_per_ngram=cfg.vocab_size_per_ngram,
            max_ngram_size=cfg.max_ngram_size,
            n_head_per_ngram=cfg.n_head_per_ngram,
        )

        # store moduli in a tensor for each layer
        self.layer_moduli: Dict[int, torch.Tensor] = {}
        for layer_id in cfg.layer_ids:
            self.layer_moduli[layer_id] = torch.tensor(
                moduli[int(layer_id)],
                dtype=torch.int64,
            )
        
    def compress_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.compressed is None:
            return input_ids
        return self.compressed.compress(input_ids)

    @staticmethod
    def _shift_left_pad(
        x: torch.Tensor,
        k: int,
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        Shift left and pad with pad_token_id.
        Args:
            x: Input tensor of shape (batch, seq_len)
            k: Number of positions to shift left
            pad_token_id: Token ID to use for padding
        Returns:
            Shifted and padded tensor of shape (batch, seq_len + k)
        """
        if k == 0:
            return x
        batch_size, seq_len = x.shape
        return F.pad(
            x,
            (k, 0),
            value=pad_token_id,
        )[:, :seq_len]

    def _hash_for_layer(
        self,
        input_ids: torch.Tensor,
        layer_id: int
    ) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        return: (batch_size, seq_len, head_total_size)
        """
        cfg = self.cfg
        layer_id = int(layer_id)

        x = input_ids.to(torch.int64)
        batch_size, seq_len = x.shape

        multipliers = self.layer_multipliers[layer_id].to(x.device)
        moduli = self.layer_moduli[layer_id].to(x.device)

        # shifts[0..max_ngram_size-1]: (max_ngram_size,)
        shifts = [
            self._shift_left_pad(x, k, self.pad_token_id)
            for k in range(cfg.max_ngram_size)
        ]

        all_hashes = []
        for n in range(2, cfg.max_ngram_size + 1):
            ngram_index = n - 2  # we use 2-gram as base index
            token_ids = shifts[:n]

            mix = token_ids[0] * multipliers[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(
                    mix,
                    token_ids[k] * multipliers[k]
                )

            # per head mod with prime moduli
            for h in range(cfg.n_head_per_ngram):
                m = int(moduli[ngram_index, h].item()) # structure of moduli: layer_id -> ngram_index -> head_index
                all_hashes.append(mix % m).to(torch.int64)
        
        return torch.stack(all_hashes, dim=-1) #stacks the hashes for all ngrams along the last dimension

    def hash(
        self,
        input_ids: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """
        input_ids: (batch_size, seq_len)
        return dict: {layer_id: hash_ids(batch_size, seq_len, head_total_size)}
        """
        x = self.compress_ids(input_ids)
        return {
            int(layer_id): self._hash_for_layer(x, int(layer_id)) for layer_id in self.cfg.layer_ids
        }


# --------------------
# Multi-head embedding + lookup (separate from gate and fusion)
# --------------------

class MultiHeadEmbedding(nn.Module):   
    """
    Using multiple heads for the embedding lookup to improve the performance of the model by allowing the model to learn different representations (different n-grams) of the input data.
    As suggested in the demo, pack multiple heads into one nn.Embedding module by using offsets.
    """

    def __init__(
        self,
        head_sizes: List[int],
        head_dim: int,
    ) -> None:
        super().__init__()
        self.head_sizes = [int(n) for n in head_sizes]
        self.head_dim = int(head_dim)

        offsets = [0]
        for n in self.head_sizes[:-1]:  # last element is the total number of heads, we don't need to add it to the offsets.
            offsets.append(offsets[-1] + n) # adds the number of heads to the previous offset.
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)

        total = sum(self.head_sizes)
        self.embedding = nn.Embedding(total, self.head_dim)

    def forward(self, head_ids: torch.Tensor) -> torch.Tensor:
        # head_ids: (batch_size, seq_len, head_total_size)
        shifted = head_ids + self.offsets.view(1, 1, -1).to(device=head_ids.device)
        return self.embedding(shifted) # (batch_size, seq_len, head_total_size, head_dim)


class EngramLookup(nn.Module):
    """
    Pure lookup module that does not use any activation functions:
      input_ids -> hash_ids -> embedding lookup -> flattened memory vector
    No gating or fusion in this module
    """
    def __init__(
        self,
        cfg: EngramConfig,
        layer_id: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_id = int(layer_id)
        self.hash_mapping = NgramHashMapping(cfg)

        # Build head_sizes for this layer (flatten across ngram orders)
        # Using prime mmoduli list, so head_sizes = primes per head
        moduli = self.hash_mapping.layer_moduli[self.layer_id] # (ngram_order, heads)
        head_sizes = [
            int(x) for x in moduli.flatten().tolist()
        ]
        self.head_total_size = len(head_sizes)
        head_dim = cfg.n_embed_per_ngram // cfg.n_head_per_ngram # total embedding dimension divided by the number of heads
        self.head_dim = head_dim

        self.multi_head_embedding = MultiHeadEmbedding(
            head_sizes=head_sizes,
            head_dim=head_dim,
        )
        
        # flattened memory vector dimension
        # (max_ngram_size - 1) * n_head_per_ngram  = mem_dim, because head_total_size = (max_ngram_size - 1) * n_head_per_ngram
        self.mem_dim = self.head_total_size * self.head_dim

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            memory_vector: (batch_size, seq_len, mem_dim)
            memory_weights: (batch_size, seq_len, head_total_size)
        """
        hash_ids = self.hash_mapping.hash(input_ids)[self.layer_id]  # (batch_size, seq_len, head_total_size)
        embedding = self.multi_head_embedding(hash_ids) # (batch_size, seq_len, head_total_size, head_dim)
        mem_flattened = embedding.view(start_dim=-2) # reshape the embedding tensor to (batch_size, seq_len, mem_dim)
        return mem_flattened, hash_ids



# --------------------
# Gating & fusion module
# --------------------

class EngramKVProjector(nn.Module):
    def __init__(
        self,
        mem_dim: int,
        d_model: int,
        bias: bool = True,
    ):
        super().__init__()
        self.key_proj = nn.Linear(mem_dim, d_model, bias=bias)
        self.value_proj = nn.Linear(mem_dim, d_model, bias=bias)
    
    def forward(
        self,
        mem_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_proj(mem_flat), self.value_proj(mem_flat)





