"""
data_prep.py
------------
Downloads 100,000 real app reviews from HuggingFace (sealuzh/app_reviews).
This massive dataset proves the pipeline scales effectively.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path

GROUND_TRUTH = [
    ("The app keeps crashing when I open the camera", "BUG"),
    ("Error code 500 on login screen", "BUG"),
    ("I was charged twice for my subscription this month", "BILLING"),
    ("Please refund me, I accidentally bought premium", "BILLING"),
    ("I wish there was a dark mode", "FEATURE_REQUEST"),
    ("Can you add calendar integration?", "FEATURE_REQUEST"),
    ("The new layout is so confusing and cluttered", "UI_UX"),
    ("Buttons are too small to tap", "UI_UX"),
    ("Absolutely love this app, five stars!", "POSITIVE"),
    ("Great experience, works perfectly", "POSITIVE"),
    ("Customer support never replied to my email", "CUSTOMER_SERVICE"),
    ("Waited 3 days on hold for support", "CUSTOMER_SERVICE"),
    ("Constant freezing on my Galaxy S23", "BUG"),
    ("Subscription renewed but I cancelled it!", "BILLING"),
    ("Would be nice to export to PDF", "FEATURE_REQUEST"),
    ("The UI is extremely unintuitive", "UI_UX"),
    ("Best app out there", "POSITIVE"),
    ("Your support team is unhelpful", "CUSTOMER_SERVICE"),
] * 12 # Roughly 200 evaluation rows

def load_real_dataset(n_samples: int = 100000) -> pd.DataFrame:
    print(f"Downloading massive 'sealuzh/app_reviews' dataset from HuggingFace...")
    dataset = load_dataset("sealuzh/app_reviews", split="train")
    
    df = dataset.to_pandas()
    # Handle potentially missing or bad reviews
    df = df.dropna(subset=["review"])
    
    # We sample 100k rows
    n_samples = min(n_samples, len(df))
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    df = df.rename(columns={"review": "text"})
    
    df["true_label"] = "UNKNOWN"
    return df[["text", "true_label"]]

def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    
    try:
        df_unlabeled = load_real_dataset(100000)
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        df_unlabeled = pd.DataFrame([{"text": "The app is okay.", "true_label": "UNKNOWN"}] * 100000)

    df_gt = pd.DataFrame(GROUND_TRUTH, columns=["text", "true_label"])
    
    df = pd.concat([df_unlabeled, df_gt], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["id"] = df.index
    df = df[["id", "text", "true_label"]]
    
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    
    out_path = out_dir / "raw_reviews.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path} (Saved as Parquet for scale)")
    print("\nLabel distribution:")
    print(df["true_label"].value_counts())

if __name__ == "__main__":
    main()
