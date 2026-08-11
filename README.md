# Weak Supervision Data Labeling Factory (Enterprise Scale)

Programmatic data labeling pipeline that replaces thousands of hours of manual annotation with **heuristic rules, regex patterns, and HuggingFace Transformer models**, mathematically combined via [Snorkel](https://www.snorkel.org/)'s generative `LabelModel` — then validated by training a downstream classifier on the resulting labels.

## Why this project

Enterprise NLP's real bottleneck is rarely modeling — it's **clean labeled data**. Hand-labeling is slow, expensive, and doesn't scale as label schemas evolve. This project demonstrates the *weak supervision* paradigm: encode domain knowledge as cheap, noisy labeling functions, then let a statistical model learn how much to trust each one — no ground truth required.

To prove its real-world applicability and scale, this project operates on **100,000 real app reviews** from HuggingFace, utilizing Data Engineering best practices (Parquet storage) and Advanced NLP (`distilbert` sentiment pipeline).

## Architecture

```
Raw unlabeled text (100,000 HuggingFace app_reviews)
        │
        ▼
┌───────────────────────────┐
│  Labeling Functions (LFs) │   10 heuristics: keyword rules, regex
│  transformers / rules     │   patterns, DistilBERT sentiment model
└──────────────┬────────────┘
               ▼
     LF vote matrix (100k × 10 LFs)
               ▼
┌───────────────────────────┐
│   Snorkel LabelModel      │  Learns per-LF accuracy + correlations
│   (generative, no GT)     │  via matrix completion
└──────────────┬────────────┘
               ▼
   Probabilistic weak labels (Saved as Parquet)
               ▼
┌───────────────────────────┐
│  TF-IDF + SGDClassifier   │  Out-of-core scalable downstream 
│  (Log Loss Optimization)  │  classifier trained on weak labels
└──────────────┬────────────┘
               ▼
   Validated against held-out ground truth
```

## Project structure

```
weak-supervision-labeling-factory/
├── src/
│   ├── data_prep.py            # Fetches HuggingFace dataset (100k) & exports Parquet
│   ├── labeling_functions.py   # LFs including HuggingFace Transformers pipeline
│   ├── train_label_model.py    # fits Snorkel LabelModel -> weak labels
│   ├── train_classifier.py     # scalable SGDClassifier training
│   └── evaluate_baseline.py    # LabelModel vs. majority-vote comparison
├── app.py                      # Premium Streamlit demo UI
├── main.py                     # runs the full pipeline end to end
├── requirements.txt
├── data/                       # generated at runtime
├── models/                     # generated at runtime
└── reports/                    # generated at runtime
```

## Setup

```bash
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python3 main.py
```

This runs, in order:
1. `data_prep.py` — Fetches ~100,000 unlabelled app reviews from HuggingFace and mixes in a tiny ~200-row simulated hand-labeled validation set.
2. `train_label_model.py` — applies all 10 LFs, fits the Snorkel `LabelModel`, and writes probabilistic labels to `data/labeled_reviews.parquet`.
3. `train_classifier.py` — trains a TF-IDF + SGDClassifier pipeline on the weak labels only, and reports accuracy.
4. `evaluate_baseline.py` — compares the LabelModel against naive majority vote to quantify the value of generative label modeling.

Then launch the interactive demo:

```bash
streamlit run app.py
```

ier` to ensure the downstream training didn't bottleneck memory.
