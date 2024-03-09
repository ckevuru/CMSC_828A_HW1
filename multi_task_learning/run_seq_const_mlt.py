import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import numpy as np

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

from transformers import AutoTokenizer, get_scheduler, SchedulerType
from utils_ner_and_nli import load_ner_dataloaders, evaluate_ner, load_ner_model, load_nli_dataloaders, evaluate_nli, load_nli_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(
    accelerator,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs,
    num_steps_per_epoch,
    weight_lambda,
    task
):
    (
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
    )
    num_training_steps=num_steps_per_epoch * num_epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_epochs):
        model.train()
        scale = weight_lambda if task == "ner" else 1 - weight_lambda
        total_loss = 0.0
        for _, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            output = model(**batch)

            loss = scale * output["loss"]
            accelerator.backward(loss)
            total_loss += loss.detach().float()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        total_avg_loss = total_loss.item() / (num_epochs / num_training_steps)
        print(f"epoch {epoch}: train_loss:", total_avg_loss)

        if task == "ner":
            _ = evaluate_ner(
                accelerator,
                model,
                val_dataloader,
                epoch,
                evaluate.load("seqeval"),
            )
        else:
            _ = evaluate_nli(
                accelerator,
                model,
                val_dataloader,
                epoch,
               evaluate.load("glue", "mnli"),
            )

    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_samples", 
        type=int, 
        default=None
    )
    parser.add_argument(
        "--max_val_samples", 
        type=int, 
        default=None
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--weight_lambda", 
        type=float, 
        default=0.5
    )
    args = parser.parse_args()

    assert (
        args.weight_lambda >= 0 and args.weight_lambda <= 1
    ), "Weight must be between 0 and 1"

    return args

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    ner_train_dataloader, ner_val_dataloader = load_ner_dataloaders(tokenizer)

    nli_train_dataloader, nli_val_dataloader = load_nli_dataloaders(tokenizer)

    ner_model = load_ner_model(args.model_name_or_path)
    nli_model = load_nli_model(args.model_name_or_path)

    params = set(list(ner_model.parameters()) + list(nli_model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    num_steps_per_epoch = math.ceil(min(len(ner_train_dataloader), len(nli_train_dataloader)) / args.gradient_accumulation_steps)

    nli_trained_model = train(
        accelerator,
        nli_model,
        nli_train_dataloader,
        nli_val_dataloader,
        optimizer,
        args.num_train_epochs,
        num_steps_per_epoch,
        args.weight_lambda,
        "nli"
    )

    nli_ner_trained_model = train(
        accelerator,
        ner_model,
        ner_train_dataloader,
        ner_val_dataloader,
        optimizer,
        args.num_train_epochs,
        num_steps_per_epoch,
        args.weight_lambda,
        "ner"
    )

    nli_ner_trained_model.save_pretrained(os.path.join(args.output_dir, "nli_ner_model"), is_main_process=accelerator.is_main_process, save_function=accelerator.save)

    params = set(list(ner_model.parameters()) + list(nli_model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    ner_trained_model = train(
        accelerator,
        ner_model,
        ner_train_dataloader,
        ner_val_dataloader,
        optimizer,
        args.num_train_epochs,
        num_steps_per_epoch,
        args.weight_lambda,
        "ner"
    )

    ner_nli_trained_model = train(
        accelerator,
        nli_model,
        nli_train_dataloader,
        nli_val_dataloader,
        optimizer,
        args.num_train_epochs,
        num_steps_per_epoch,
        args.weight_lambda,
        "nli"

    )

    ner_nli_trained_model.save_pretrained(os.path.join(args.output_dir, "ner_nli_model"), is_main_process=accelerator.is_main_process, save_function=accelerator.save)


if __name__ == "__main__":
    main()