import numpy as np
import argparse
import os
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    default_data_collator,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)

ner_tag_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
ner_id_to_tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}

#NER util functions
def load_ner_model(model_name: str):
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        id2label=ner_id_to_tag,
        label2id=ner_tag_to_id,
    )

def tokenize_and_align_ner_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        label = [ner_id_to_tag[x] for x in label]
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(ner_tag_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def evaluate_ner(accelerator, model, eval_dataloader, epoch, metric):
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        predictions = accelerator.pad_across_processes(
            predictions, dim=1, pad_index=-100
        )

        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        predictions = predictions_gathered.detach().cpu().clone().numpy()
        labels = labels_gathered.detach().cpu().clone().numpy()

        true_labels = [[ner_tag_to_id[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [ner_tag_to_id[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(f"NER for epoch:{epoch} results: {results}")
    return results

def load_ner_dataloaders(tokenizer, batch_size=16):
    #Load the ner dataset
    train_ds, val_ds, test_ds = load_dataset(
        "Babelscape/wikineural", 
        split=["train_en", "val_en", "test_en"]
    )

    train_tokenized_dataset = train_ds.map(
        tokenize_and_align_ner_labels,
        batched=True,
        remove_columns=train_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    val_tokenized_dataset = val_ds.map(
        tokenize_and_align_ner_labels,
        batched=True,
        remove_columns=val_ds.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_tokenized_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
    )

    eval_dataloader = DataLoader(
        val_tokenized_dataset, 
        collate_fn=data_collator, 
        batch_size=batch_size
    )

    return train_dataloader, eval_dataloader

#NLI util functions
def load_nli_model(model_name, num_labels=3):
    config = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels, finetuning_task="mnli"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        ignore_mismatched_sizes=True,
    )
    return model

def evaluate_nli(accelerator, model, eval_dataloader, epoch, metric):
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"NLI  for epoch:{epoch} val accuracy: {eval_metric}")
    return eval_metric

def preprocess_nli_datasets(raw_datasets, tokenizer, pad_to_max_length, max_length):
    sentence1_key, sentence2_key = "premise", "hypothesis"
    padding = "max_length" if pad_to_max_length else False

    def preprocess_nli_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=max_length, truncation=True
        )

        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_nli_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets

def load_nli_dataloaders(
    tokenizer,
    pad_to_max_length=False,
    max_length=128,
    max_train_samples=None,
    max_eval_samples=None,
    train_batch_size=16,
    eval_batch_size=16,
):
    # Downloading and loading the MultiNLI dataset
    raw_datasets = load_dataset("glue", "mnli")

    # Preprocessing the datasets
    processed_datasets = preprocess_nli_datasets(
        raw_datasets,
        tokenizer,
        max_length=max_length,
        pad_to_max_length=pad_to_max_length,
    )

    train_dataset = processed_datasets["train"]
    eval_dataset_matched = processed_datasets["validation_matched"]

    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=train_batch_size,
    )

    eval_matched_dataloader = DataLoader(
        eval_dataset_matched,
        collate_fn=data_collator,
        batch_size=eval_batch_size,
    )

    return train_dataloader, eval_matched_dataloader
