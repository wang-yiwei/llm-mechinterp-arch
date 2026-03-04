# src/data/data.py

# Author: Yiwei Wang
# Date: 2026-03-04

# Information about the script:
# This script implements the data loading for the model.

import itertools
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def get_lm_dataloader(
    dataset_name_or_path: str,
    tokenizer_name_or_path: str,
    split: str = "train",
    seq_len: int = 2048,
    batch_size: int = 8,
    num_workers: int = 0,
    text_column: str = "text",
    seed: int = 0,
    drop_last: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    
    dataset = load_dataset(
        dataset_name_or_path,
        split=split,
        **dataset_kwargs,
    )

    if text_column not in dataset.column_names:
        raise ValueError(f"Text column {text_column} not found in dataset columns: {dataset.column_names}")
    
    # optinal shuffle before mapping
    if split == "train":
        dataset = dataset.shuffle(seed=seed)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    def tokenize_function(examples):
        # return only input_ids to avoid carrying attention_mask around
        outputs = tokenizer(
            examples[text_column],
            add_special_tokens=False
        )
        return {"input_ids": outputs["input_ids"]}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset...",
        num_proc=num_workers,
    )

    chunk_size = seq_len + 1 # next word prediction

    # chains the input_ids lists together into a single list
    def group_texts(examples):
        ids = list(
            itertools.chain.from_iterable(
                examples["input_ids"]  
            )
        )
        total_length = (len(ids) // chunk_size) * chunk_size
        ids = ids[:total_length]
        chunks = [
            ids[i:i+chunk_size] for i in range(0, total_length, chunk_size)
        ]
        return {"input_ids": chunks}

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into chunks of {chunk_size}...",
        num_proc=num_workers,
    )
    lm_dataset.set_format(
        type="torch",
        columns=["input_ids"],
    )
    
    return DataLoader(
        lm_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        drop_last=(drop_last and split == "train"),
        num_workers=num_workers,
        pin_memory=True,
    )
