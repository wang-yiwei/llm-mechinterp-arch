# scripts/run_engram_moe.py

# Author: Yiwei Wang
# Date: 2026-03-03

# Information about the script:
# This script runs the Engram + MoE FFN Transformer Language Model

import argparse
from random import choices
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# make repo root importable
ROOT = Path(__file__).resolve().parent[2]
sys.path.append(str(ROOT))

from src.modern.engram import (
    EngramConfig,
    CharTokenizer,
)
from src.modern.llama_style_engram_moe import LlamaStyleEngramMoETransformerLM

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

def chunkify(
    ids,
    seq_len,
):
    # make (batch_size, seq_len+1) chunks of ids
    n = (len(ids) // (seq_len + 1)) * (seq_len + 1)
    ids = ids[:n]
    x = torch.tensor(ids, dtype=torch.long).view(-1, seq_len + 1)  # reshapes the tensor into a 2D tensor with shape (batch_size, seq_len + 1)
    return x[:, :-1], x[:, 1:]  # returns the first seq_len tokens and the last seq_len tokens

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_type", choices=["hf", "char"], default="hf")
    ap.add_argument("--tokenizer_name_or_path", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--text", default="NLP bridges the gap between human communication and computer understanding by combining computational linguistics with statistical modeling, machine learning, and deep learning")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=0, help="0 = forward pass, 1 = backward pass, 2 = forward + backward")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    # tokenize
    if args.tokenizer_type == "hf":
        if AutoTokenizer is None:
            raise RuntimeError("transformers not found. Use --tokenizer_type char instead.")
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        ids = tokenizer(
            args.text,
            return_tensors="pt",
        ).input_ids[0].tolist()
        vocab_size = len(tokenizer)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    else:
        char_tokenizer = CharTokenizer()
        char_tokenizer.train_from_text(args.text)
        ids = char_tokenizer.encode(args.text)
        vocab_size = char_tokenizer.vocab_size
        pad_token_id = char_tokenizer.pad_token_id

    # ensure text length is enough for seq_len + 1 chunks
    if len(ids) < args.seq_len + 1:
        ids = ids * ((args.seq_len + 1) // max(1, len(ids)) + 1)  # repeats the ids tensor until it is at least as long as args.seq_len + 1
    
    x,y = chunkify(ids, args.seq_len)
    x = x[:2] # only use 2 chunks for testing
    y = y[:2]

    print("x shape:", x.shape)
    print("y shape:", y.shape)

    print("vocab size:", vocab_size)

    # engram config
    engram_config = EngramConfig(
        layer_ids=[0, 1],   # early injection
        max_ngram_size=3,
        n_head_per_ngram=8,
        n_embed_per_ngram=512,
        vocab_size_per_ngram=[50_000, 50_000],
        tokenizer_type=args.tokenizer_type,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        pad_token_id=pad_token_id,
        use_compressed_tokenizer=(args.tokenizer_type=="hf"),
        use_conv=True,
        conv_kernel_size=4,
        conv_dilation=3,
        gate_signed_sqrt=True,
        seed=88,
        prime_seed_constant=10007,
    )


    # language model
    model = LlamaStyleEngramMoETransformerLM(
        vocab_size=vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_layers=4,
        max_seq_len=args.seq_len,
        n_kv_heads=2,
        rope_base=10000.0,
        num_experts=8,
        top_k=2,
        moe_layer_indices=[0, 1, 2, 3],
        engram_config=engram_config,
        engram_layer_indices=[0, 1], # early injection in the first 2 layers
    )


    # forward pass
    logits, all_router_logits, all_load_balancing_loss, engram_aux = model(
        x,
        return_router_logits=True,
        return_load_balancing_loss=True,
        return_engram_aux=True,
    )

    print("logits shape:", logits.shape)
    print("num router layers:", len(all_router_logits) if all_router_logits is not None else None)
    print("num load balancing losses:", len(all_load_balancing_loss) if all_load_balancing_loss is not None else None)
    print("num engram entries:", len(engram_aux) if engram_aux is not None else None)


    # optional minimal training loop
    if args.steps > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        model.train()
        for i in range(args.steps):
            logits, _, all_load_balancing_loss, _ = model(
                x,
                return_router_logits=False,
                return_load_balancing_loss=True,
                return_engram_aux=False,
            )
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )
            
            # sum the auxiliary losses from all MoE layers and scale by a coefficient e.g. 0.1
            if all_load_balancing_loss:
                aux_loss = sum(all_load_balancing_loss)
            else:
                # ensure it's a tensor on the same device as x so .backward() and / item() work safely
                aux_loss = torch.tensor(0.0, device=x.device)
            total_loss = loss + 0.1 * aux_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"step {i} loss: {total_loss.item():.4f}")
            print(f"step {i} aux loss: {aux_loss.item():.4f}")


    # optional minimal evaluation loop
    if args.steps > 0:
        model.eval()
        with torch.no_grad():
            logits, _, _, _ = model(
                x,
                return_router_logits=False,
                return_load_balancing_loss=False,
                return_engram_aux=False,
            )
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
            )
            print(f"eval loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()