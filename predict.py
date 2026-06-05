#!/usr/bin/env python3
"""Classify a news headline by topic using the fine-tuned model in ./model."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="Headline / text to classify.")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(
            f"Model not found at {model_dir}. Train one first with train.py."
        )

    text = " ".join(args.text).strip()
    if not text:
        text = input("Enter a news headline: ").strip()
    if not text:
        raise SystemExit("No text provided.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with open(model_dir / "label_map.json") as f:
        id2label = json.load(f)["id2label"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    k = min(args.top_k, probs.shape[-1])
    top = torch.topk(probs, k)
    print(f"\nText: {text}\n")
    print("Predicted topics:")
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        label = id2label.get(str(idx), str(idx))
        print(f"  {label:<14} {score:.3f}")


if __name__ == "__main__":
    main()
