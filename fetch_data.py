#!/usr/bin/env python3
"""Fetch labeled news data from NewsData.io for topic classification.

Uses the FREE /news endpoint's `category` field as the topic label, building a
CSV dataset of (text, label) pairs suitable for fine-tuning a BERT classifier.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

API_URL = "https://newsdata.io/api/1/news"

# Free-tier categories used directly as classification labels.
CATEGORIES = [
    "business",
    "entertainment",
    "environment",
    "food",
    "health",
    "politics",
    "science",
    "sports",
    "technology",
    "world",
]


def get_api_key():
    key = os.environ.get("NEWSDATA_API_KEY")
    if not key:
        sys.exit(
            "ERROR: NEWSDATA_API_KEY is not set.\n"
            "Get a free key at https://newsdata.io and run:\n"
            "  export NEWSDATA_API_KEY=your_key_here"
        )
    return key


def fetch_category(api_key, category, pages, language, country, sleep):
    """Fetch up to `pages` pages of articles for one category."""
    rows = []
    next_page = None
    for page_num in range(pages):
        params = {
            "apikey": api_key,
            "category": category,
            "language": language,
        }
        if country:
            params["country"] = country
        if next_page:
            params["page"] = next_page

        try:
            resp = requests.get(API_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  [{category}] network error: {exc}")
            break

        if resp.status_code == 401:
            sys.exit("ERROR: Invalid API key (401). Check NEWSDATA_API_KEY.")
        if resp.status_code == 429:
            print(f"  [{category}] rate limited (429); waiting 15s ...")
            time.sleep(15)
            continue
        if resp.status_code in (403, 422):
            print(
                f"  [{category}] request rejected ({resp.status_code}); "
                "skipping (likely a free-plan limitation)."
            )
            break
        if resp.status_code != 200:
            print(f"  [{category}] unexpected status {resp.status_code}; stopping.")
            break

        try:
            payload = resp.json()
        except ValueError:
            print(f"  [{category}] could not parse response; stopping.")
            break

        results = payload.get("results") or []
        if not results:
            print(f"  [{category}] no more results.")
            break

        for art in results:
            title = (art.get("title") or "").strip()
            desc = (art.get("description") or "").strip()
            text = (title + ". " + desc).strip(". ").strip()
            if len(text) < 20:
                continue
            rows.append({"text": text, "label": category})

        next_page = payload.get("nextPage")
        print(
            f"  [{category}] page {page_num + 1}: {len(results)} articles "
            f"(kept so far: {len(rows)})"
        )
        if not next_page:
            break
        time.sleep(sleep)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages", type=int, default=3,
        help="Pages to fetch per category (~10 articles/page on the free tier).",
    )
    parser.add_argument("--language", default="en", help="Language code (e.g. en).")
    parser.add_argument("--country", default="", help="Optional country code filter.")
    parser.add_argument(
        "--categories", nargs="*", default=CATEGORIES,
        help="Categories to use as labels.",
    )
    parser.add_argument("--out", default="data/news_dataset.csv")
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to wait between requests (free-tier friendly).",
    )
    args = parser.parse_args()

    api_key = get_api_key()
    all_rows = []
    for category in args.categories:
        print(f"Fetching '{category}' ...")
        all_rows.extend(
            fetch_category(
                api_key, category, args.pages, args.language, args.country, args.sleep
            )
        )

    # Deduplicate by text.
    seen = set()
    unique = []
    for row in all_rows:
        if row["text"] in seen:
            continue
        seen.add(row["text"])
        unique.append(row)

    if not unique:
        sys.exit("No data collected. Try different categories or check your plan.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(unique)

    counts = {}
    for row in unique:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print(f"\nWrote {len(unique)} examples to {out_path}")
    for label, count in sorted(counts.items()):
        print(f"  {label:<14} {count}")


if __name__ == "__main__":
    main()
