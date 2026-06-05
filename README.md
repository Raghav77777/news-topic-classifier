# News Topic Classifier

Fine-tune a BERT model to classify news headlines into topics, using labeled
training data pulled live from the [NewsData.io](https://newsdata.io) news API.
The free `/news` endpoint's `category` field is used as the topic label, so you
can build a real, working classifier without any paid plan.

Built with Python + HuggingFace Transformers. A training notebook is included.

## What it does

1. **`fetch_data.py`** downloads recent articles from NewsData.io across several
   categories and saves a `(text, label)` CSV dataset.
2. **`train.py`** fine-tunes `distilbert-base-uncased` on that dataset with the
   HuggingFace `Trainer` and saves the model to `./model`.
3. **`predict.py`** loads the fine-tuned model and classifies any headline.
4. **`news_topic_classifier.ipynb`** walks through the whole flow in a notebook.

## Topics (labels)

`business`, `entertainment`, `environment`, `food`, `health`, `politics`,
`science`, `sports`, `technology`, `world` — these are NewsData.io free-tier
categories used directly as classification labels.

## Setup

```bash
git clone https://github.com/<your-username>/news-topic-classifier.git
cd news-topic-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### API key

Get a **free** API key at https://newsdata.io and export it:

```bash
export NEWSDATA_API_KEY=your_key_here     # Windows: set NEWSDATA_API_KEY=your_key_here
```

The key is read from the `NEWSDATA_API_KEY` environment variable and is **never**
hardcoded. The scripts exit with a clear message if it is unset.

## Usage

```bash
# 1. Build a labeled dataset (3 pages per category)
python fetch_data.py --pages 3

# 2. Fine-tune the classifier
python train.py --epochs 3

# 3. Classify a headline
python predict.py "Stock markets rally as technology shares surge"
```

Example output:

```
Text: Stock markets rally as technology shares surge

Predicted topics:
  business       0.812
  technology     0.121
  world          0.029
```

## Notebook

Open `news_topic_classifier.ipynb` in Jupyter or Google Colab for an end-to-end,
runnable walkthrough of fetching, training and prediction.

## Free vs. paid plan

This project is designed to run entirely on the **free** NewsData.io tier. It
only uses the free `/news` endpoint with the `category`, `q`, `language`,
`country` and `page` parameters. The fetcher handles invalid keys (401), rate
limits (429), plan-restriction responses (403/422) and empty results gracefully.

Paid-only features are **not** required and are **not** used:

- sentiment analysis
- AI fields (`ai_tag`, `ai_region`, `ai_org`, `ai_summary`)
- the historical `/archive` endpoint and long date ranges
- advanced full-text query operators

If you upgrade your plan you can simply fetch more pages for a larger dataset,
but all defaults stay within free-tier limits.

## License

MIT
