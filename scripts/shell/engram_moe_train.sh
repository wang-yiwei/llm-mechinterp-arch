#!/user/bin/env bash
set -euo pipefail

# Example commands for training Engram+MoE models with Accelerate.
# Adjust paths and parameters to your machine.

# 0) Setup (run once)
# pip install accelerate datasets transformers tokenizers sympy
# accelerate config

# 1) Toy "pretraining from scratch" on Wikitext-2 (small baseline)
accelerate launch scripts/modern/train_accelerate_engram.py \
  --dataset_name_or_path wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --tokenizer_name_or_path gpt2 \
  --seq_len 256 \
  --batch_size 8 \
  --epochs 1 \
  --grad_accum 1 \
  --mixed_precision fp16 \
  --use_engram \
  --use_compressed_tokenizer \
  --engram_layers 0,1 \
  --moe_layers 0,1,2,3 \
  --aux_coef 0.1 \
  --output_dir checkpoints/pretrain_wt2

# 2) Longer training with max_steps + periodic checkpoints
accelerate launch scripts/modern/train_accelerate_engram.py \
  --dataset_name_or_path wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --tokenizer_name_or_path gpt2 \
  --seq_len 256 \
  --batch_size 8 \
  --max_steps 2000 \
  --grad_accum 4 \
  --mixed_precision fp16 \
  --use_engram \
  --use_compressed_tokenizer \
  --engram_layers 0,1 \
  --moe_layers 0,1,2,3 \
  --save_every 200 \
  --aux_coef 0.1 \
  --output_dir checkpoints/pretrain_wt2_long

# 3) Resume training from a saved Accelerate state
# (example checkpoint path; adjust to your actual checkpoint dir)
accelerate launch scripts/modern/train_accelerate_engram.py \
  --dataset_name_or_path wikitext \
  --dataset_config wikitext-2-raw-v1 \
  --tokenizer_name_or_path gpt2 \
  --seq_len 256 \
  --batch_size 8 \
  --max_steps 3000 \
  --grad_accum 4 \
  --mixed_precision fp16 \
  --use_engram \
  --use_compressed_tokenizer \
  --engram_layers 0,1 \
  --moe_layers 0,1,2,3 \
  --resume_from checkpoints/pretrain_wt2_long/checkpoint-000200 \
  --output_dir checkpoints/pretrain_wt2_long

# 4) Low-resource adaptation on a local JSONL corpus
# Requires your train script + dataloader to pass `data_files` into load_dataset.
# Example: data/hist_corpus.jsonl with {"text": "..."} lines. In case of an uncurated corpus, add an argument like: --data_files_train path/to.jsonl.

accelerate launch scripts/modern/train_accelerate_engram.py \
  --dataset_name_or_path json \
  --dataset_config "" \
  --tokenizer_name_or_path gpt2 \
  --seq_len 256 \
  --batch_size 8 \
  --max_steps 1000 \
  --lr 1e-4 \
  --warmup_ratio 0.05 \
  --grad_accum 4 \
  --mixed_precision fp16 \
  --use_engram \
  --use_compressed_tokenizer \
  --engram_layers 0,1 \
  --moe_layers 0,1,2,3 \
  --resume_from checkpoints/pretrain_wt2_long/final_model \
  --output_dir checkpoints/adapt_hist
#   --data_files_train data/hist_corpus.jsonl

echo "Done. Edit examples/train_commands.sh to match your environment."