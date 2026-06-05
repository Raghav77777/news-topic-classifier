#!/usr/bin/env python3
"""Fine-tune a transformer to classify news topics from a NewsData.io dataset.

Reads the CSV produced by fetch_data.py (columns: text, label), fine-tunes a
pre-trained BERT-style model with the HuggingFace Trainer, and saves the model,
tokenizer and label map to ./model for use by predict.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/news_dataset.csv")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="model")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found at {data_path}. Run fetch_data.py first."
        )

    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if len(df) < 20:
        raise SystemExit("Not enough data to train. Fetch more articles first.")

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    print(f"Loaded {len(df)} examples across {len(labels)} topics: {labels}")

    train_df, eval_df = train_test_split(
        df, test_size=args.test_size, stratify=df["label_id"], random_state=42
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(
        train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    ).map(tokenize, batched=True)
    eval_ds = Dataset.from_pandas(
        eval_df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
        preserve_index=False,
    ).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final evaluation:", metrics)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    with open(out / "label_map.json", "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)
    print(f"Saved model to {out}/")


if __name__ == "__main__":
    main()
