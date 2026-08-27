#!/usr/bin/env python3
"""Fetch labeled news data from NewsData.io for topic classification.

Uses the FREE /news endpoint's `category` field as the topic label, building a
CSV dataset of (text, label) pairs suitable for fine-tuning a BERT classifier.

Everything here works on a brand-new free NewsData.io key: only the `/news`
endpoint and its free parameters (`category`, `language`, `country`, `page`)
are used. No paid-only fields (sentiment, ai_tag, /archive) are required.

Examples:
    export NEWSDATA_API_KEY=your_key_here

    # default: 1 page per category, English, worldwide
    python fetch_data.py

    # 3 pages per category, US-only articles
    python fetch_data.py --pages 3 --country us

    # a couple of countries, just two categories
    python fetch_data.py --country us,gb --categories sports technology
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

# NewsData.io accepts a small number of comma-separated country codes per call
# on the free tier. Keep the list short to avoid a 422 from the API.
MAX_COUNTRIES = 5


def get_api_key():
    key = os.environ.get("NEWSDATA_API_KEY")
    if not key:
        sys.exit(
            "ERROR: NEWSDATA_API_KEY is not set.\n"
            "Get a free key at https://newsdata.io and run:\n"
            "  export NEWSDATA_API_KEY=your_key_here"
        )
    return key


def normalize_country(raw):
    """Normalize a --country value into the string the API expects.

    Accepts a single code ("us") or a comma-separated list ("us, gb").
    Returns None when no country filter was requested, meaning the API
    returns worldwide results (the default free-tier behaviour).
    """
    if not raw:
        return None
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if not codes:
        return None
    for code in codes:
        if not code.isalpha() or len(code) != 2:
            sys.exit(
                f"ERROR: invalid country code {code!r}. Use 2-letter ISO codes, "
                "e.g. --country us or --country us,gb"
            )
    if len(codes) > MAX_COUNTRIES:
        sys.exit(
            f"ERROR: at most {MAX_COUNTRIES} country codes are supported "
            "(free-tier limit)."
        )
    return ",".join(codes)


def build_text(article, include_description=True):
    """Turn one API result into a single training string."""
    title = (article.get("title") or "").strip()
    if not title or title.lower() in {"none", "null"}:
        return ""
    if not include_description:
        return title
    description = (article.get("description") or "").strip()
    if description and description.lower() not in {"none", "null"}:
        return f"{title}. {description}"
    return title


def fetch_category(
    api_key,
    category,
    pages,
    language,
    country=None,
    include_description=True,
    sleep=1.0,
):
    """Fetch up to `pages` pages of articles for one category.

    `country` is optional; when given it is passed straight through to the
    NewsData.io `country` parameter so the dataset is geographically targeted.
    """
    rows = []
    next_page = None

    for page_num in range(1, pages + 1):
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
            print(f"  ! network error on page {page_num}: {exc}")
            break

        if resp.status_code == 429:
            print("  ! rate limited by NewsData.io (free tier) - stopping here.")
            break

        if resp.status_code != 200:
            message = ""
            try:
                body = resp.json()
                message = (body.get("results") or {}).get("message") or body.get(
                    "message", ""
                )
            except ValueError:
                message = resp.text[:200]
            print(f"  ! HTTP {resp.status_code} on page {page_num}: {message}")
            break

        try:
            payload = resp.json()
        except ValueError:
            print(f"  ! could not decode JSON on page {page_num}")
            break

        if payload.get("status") != "success":
            print(f"  ! API returned status={payload.get('status')}")
            break

        results = payload.get("results") or []
        for article in results:
            text = build_text(article, include_description=include_description)
            if text:
                rows.append((text, category))

        print(f"  page {page_num}: {len(results)} articles")

        next_page = payload.get("nextPage")
        if not next_page:
            break
        if page_num < pages:
            time.sleep(sleep)

    return rows


def write_csv(rows, out_path):
    out_path = Path(out_path)
    if out_path.parent and str(out_path.parent) not in ("", "."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Pages of results to request per category (default: 1).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="NewsData.io language code (default: en).",
    )
    parser.add_argument(
        "--country",
        default=None,
        help=(
            "Optional NewsData.io country filter, e.g. --country us. "
            "Accepts a comma-separated list (--country us,gb) up to "
            f"{MAX_COUNTRIES} codes. Omit for worldwide results."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
        choices=CATEGORIES,
        metavar="CATEGORY",
        help="Subset of categories to fetch (default: all free-tier categories).",
    )
    parser.add_argument(
        "--titles-only",
        action="store_true",
        help="Use only the headline as the training text (skip the description).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to wait between requests, to stay under free-tier limits.",
    )
    parser.add_argument(
        "--out",
        default="data/news_dataset.csv",
        help="Output CSV path (default: data/news_dataset.csv).",
    )
    args = parser.parse_args()

    if args.pages < 1:
        sys.exit("ERROR: --pages must be at least 1.")

    api_key = get_api_key()
    country = normalize_country(args.country)

    scope = f"country={country}" if country else "worldwide"
    print(
        f"Fetching {args.pages} page(s) per category "
        f"({len(args.categories)} categories, language={args.language}, {scope})\n"
    )

    all_rows = []
    for category in args.categories:
        print(f"[{category}]")
        rows = fetch_category(
            api_key,
            category,
            pages=args.pages,
            language=args.language,
            country=country,
            include_description=not args.titles_only,
            sleep=args.sleep,
        )
        print(f"  -> {len(rows)} usable rows")
        all_rows.extend(rows)
        time.sleep(args.sleep)

    # Drop duplicate texts (the same story often appears in several pages).
    seen = set()
    deduped = []
    for text, label in all_rows:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, label))

    if not deduped:
        sys.exit(
            "\nNo articles were collected. Check your API key, and if you used "
            "--country try a broader value or drop the flag."
        )

    out_path = write_csv(deduped, args.out)

    print(f"\nSaved {len(deduped)} rows to {out_path}")
    counts = {}
    for _, label in deduped:
        counts[label] = counts.get(label, 0) + 1
    for label in sorted(counts):
        print(f"  {label:<15} {counts[label]}")

    thin = [label for label, n in counts.items() if n < 10]
    if thin:
        print(
            "\nNote: few examples for " + ", ".join(sorted(thin)) +
            ". Try a higher --pages, or a different/no --country."
        )


if __name__ == "__main__":
    main()
