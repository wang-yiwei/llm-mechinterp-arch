# /scripts/modern/train_accelerate_engram.py

# Author: Yiwei Wang
# Date: 2026-03-04

# Information about the script:
# This script trains LlamaStyleEngramMoETransformerLM model using the Accelerate library:
# - Cross-entropy loss
# - optional MoE load-balancing auxiliary loss
# - checkpointing via accelerate.save_state() and accelerate.load_state()
# - grad accumulation + mixed precision training

import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Make repo root importable
ROOT = Path(__file__).resolve().parent[2]
sys.path.append(str(ROOT))

from src.data.data_loader import get_lm_dataloader
from src.modern.engram import EngramConfig
from src.modern.llama_style_engram_moe import LlamaStyleEngramMoETransformerLM

def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [
        int(x) for x in s.split(",")
    ]

def parse_int_list_optional(
    s: Optional[str],
    default: List[int],
) -> List[int]:
    if s is None:
        return default
    s = s.strip()
    if not s:
        return default
    return [
        int(x) for x in s.split(",")
    ]

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    
    # dataset
    ap.add_argument("--dataset_name_or_path", type=str, default="wikitext")
    ap.add_argument("--dataset_config", default="wikitext:wikitext-2-v1")
    ap.add_argument("--split_train", default="train")
    ap.add_argument("--split_eval", default="validation")
    ap.add_argument("--text_column", default="text")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=88)

    # tokenizer
    ap.add_argument("--tokenizer_name_or_path", default="gpt2")
    ap.add_argument("--trust_remote_code", action="store_true")

    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=1024)
    ap.add_argument("n_layers", type=int, default=4)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("rope_base", type=float, default=10000.0)

    # moe
    ap.add_argument("--num_experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--moe_layers", default="", help="comma-separated list of layer indices, e.g. '0,1,2,3' (empty = all layers)")

    # engram
    ap.add_argument("--use_engram", action="store_true")
    ap.add_argument("--engram_layers", default="0,1", help="comma-separated list of layer indices, e.g. '0,1' (injection usually in the early layers)")
    ap.add_argument("--max_ngram_size", type=int, default=3)
    ap.add_argument("--n_head_per_ngram", type=int, default=8)
    ap.add_argument("--n_embed_per_ngram", type=int, default=512)
    ap.add_argument("--vocab_size_per_ngram", default="50000,50000", help="comma-separated list of vocabulary sizes per n-gram, e.g. '50000,50000' (2-gram and 3-gram)")
    ap.add_argument("--use_compressed_tokenizer", action="store_true")
    ap.add_argument("--conv_kernel_size", type=int, default=4)
    ap.add_argument("--conv_dilation", type=int, default=3)
    ap.add_argument("--gate_signed_sqrt", action="store_true")
    ap.add_argument("--prime_seed_constant", type=int, default=10007)

    # optimisation
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 = train indefinitely; >0 = train for max_steps steps")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    # auxiliary moe loss
    ap.add_argument("--aux_coef", type=float, default=0.1)

    # accelerator & logging
    ap.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    ap.add_argument("--log_every", type-int, default=10)
    ap.add_argument("--output_dir", type=str, default="checkpoints")
    ap.add_argument("--save_every", type=int, default=0, help="0 = save only at the end; >0 = save every save_every steps")
    ap.add_argument("--resume_from", type=str, default="", help="path to accelerator state checkpoint to resume from")

    args = ap.parse_args()

    set_seed(args.seed)
    torch.backend.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = len(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # --- DataLoader ---
    # get_lm_dataloader expects **dataset_kwargs
    dataset_kwargs = {}
    # Huggingface datasets use `name=` for configs like `wikitext:wikitext-2-v1`
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config

    train_loader = get_lm_dataloader(
        dataset_name_or_path=args.dataset_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        split=args.split_train,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_column=args.text_column,
        seed=args.seed,
        drop_last=True,
        **dataset_kwargs,
    )
    eval_loader = None if args.split_eval is None else get_lm_dataloader(
        dataset_name_or_path=args.dataset_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        split=args.split_eval,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        text_column=args.text_column,
        seed=args.seed,
        drop_last=False,
        **dataset_kwargs,
    )

    # --- Layer Indexing ---
    moe_layer_indices = parse_int_list_optional(
        args.moe_layers,
        default=list(range(args.n_layers)),
    )
    engram_layer_indices = parse_int_list(args.engram_layers) if args.use_engram else []

    # --- Engram Config ---
    engram_config = None
    if args.use_engram:
        vocab_sizes = [
            int(x) for x in args.vocab_size_per_ngram.split(",") if x.strip()
        ]
        if len(vocab_sizes) != (args.max_ngram_size - 1): # checks if the number of vocabulary sizes is equal to the number of n-grams minus one
            raise ValueError(f"Expected {args.max_ngram_size - 1} vocabulary sizes, got {len(vocab_sizes)}")
    engram_config = EngramConfig(
        layer_ids=engram_layer_indices,
        max_ngram_size=args.max_ngram_size,
        n_head_per_ngram=args.n_head_per_ngram,
        n_embed_per_ngram=args.n_embed_per_ngram,
        vocab_size_per_ngram=vocab_sizes,
        tokenizer_type="hf",
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        trust_remote_code=args.trust_remote_code,
        pad_token_id=pad_token_id,
        use_compressed_tokenizer=args.use_compressed_tokenizer,
        use_conv=True,
        conv_kernel_size=args.conv_kernel_size,
        conv_dilation=args.conv_dilation,
        gate_signed_sqrt=args.gate_signed_sqrt,
        seed=args.seed,
        prime_seed_constant=args.prime_seed_constant,
    )

    # --- Model ---
    model = LlamaStyleEngramMoETransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        n_kv_heads=args.n_kv_heads,
        rope_base=args.rope_base,
        num_experts=args.num_experts,
        top_k=args.top_k,
        moe_layer_indices=moe_layer_indices,
        engram_config=engram_config,
        engram_layer_indices=engram_layer_indices,
    )

    # --- Optimizer ---
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name.lower() or "bias" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    
    optimizer = AdamW(
        [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
    )

    # --- Scheduler ---
    # If max_step is set, use it, else compute from epochs and steps per epoch
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = len(train_loader) * args.epochs

    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --- Accelerate ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )
    
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)
    
    # --- Resume from checkpoint ---
    if args.resume_from:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from}")
        accelerator.load_state(args.resume_from)
     os.makedirs(args.output_dir, exist_ok=True)

    # --- Training Loop ---
    global_step = 0
    model.train()
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]   # (batch_size, seq_len+1)   tokenized input text, and the predicted next token

            x = input_ids[:, :-1].contiguous()
            y = input_ids[:, 1:].contiguous()

            with accelerator.accumulate(model):
                # Warning: load balancing loss depends on the router logits, which are only available if it's set to True in the forward pass
                want_aux = args.aux_coef > 0.0
                logits, router_logits, load_balancing_loss, engram_aux = model(
                    x,
                    return_router_logits=want_aux,
                    return_load_balancing_loss=want_aux,
                    return_engram_aux=False,
                )

                cross_entropy_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1), # (batch_size * seq_len, )
                )
                
                if want_aux and load_balancing_loss is not None and len(load_balancing_loss) > 0:
                    aux_loss = torch.stack(
                        [loss for loss in load_balancing_loss]
                    ).mean()
                else:
                    aux_loss = torch.tensor(0.0, device=cross_entropy_loss.device)
                
                loss = cross_entropy_loss + args.aux_coef * aux_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_every == 0 and accelerator.is_main_process:
                    accelerator.print(
                        f"epoch={epoch}, step={global_step} / {total_steps} "
                        f"cross_entropy_loss={cross_entropy_loss.item():.4f}, aux_loss={aux_loss.item():.4f}, total_loss={loss.item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.6f}"
                    )
                
                if args.save_every > 0 and global_step % args.save_every == 0 and accelerator.is_main_process:
                    checkpoint_dir = Path(args.output_dir) / f"checkpoint-{global_step:06d}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    accelerator.save_state(checkpoint_dir)
                    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")
                
                if args.max_step > 0 and global_step >= args.max_step:
                    break
        if args.max_step > 0 and global_step >= args.max_step:
            break
    
        # --- Evaluation Loop ---
        if eval_loader is not None:
            model.eval()
            losses = []
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch["input_ids"]
                    x = input_ids[:, :-1].contiguous()
                    y = input_ids[:, 1:].contiguous()

                    logits, _, _, _ = model(
                        x,
                        return_router_logits=False,
                        return_load_balancing_loss=False,
                        return_engram_aux=False,
                    )
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                    )
                    losses.append(accelerator.gather_for_metrics(loss))
            if accelerator.is_main_process:
                val_loss = torch.cat(losses).mean().item()
                accelerator.print(f"eval loss: {val_loss:.4f}")
                accelerator.print(f"eval loss: {val_loss:.4f}")
            
            model.train()
    
    # --- Save the final model ---
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.save_state(final_dir)
        accelerator.print(f"Saved final model to {final_dir}")
    
    accelerator.print("Training completed!")

if __name__ == "__main__":
    main()