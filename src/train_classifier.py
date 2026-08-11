"""
train_classifier.py
--------------------
Proves the weak-supervision pipeline actually works at scale: trains a lightweight
scikit-learn text classifier purely on the LabelModel's programmatic labels
(never touching true_label), then evaluates against the small slice of rows
where we happen to know the true label.

Uses SGDClassifier to gracefully handle 100k+ rows.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_parquet(DATA_DIR / "labeled_reviews.parquet")

    # Train/test split using the WEAK labels only
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["weak_label_name"],
        test_size=0.2, random_state=42, stratify=df["weak_label_name"]
    )

    # SGDClassifier with log_loss acts like Logistic Regression but is optimized for large datasets
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000)),
        ("clf", SGDClassifier(loss="log_loss", max_iter=1000, class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    
    print("Training downstream classifier on massive weakly-labeled dataset...")
    pipeline.fit(X_train, y_train)

    # --- Evaluate against weak-label test split (sanity check) ---
    weak_preds = pipeline.predict(X_test)
    weak_acc = accuracy_score(y_test, weak_preds)
    print(f"Accuracy vs. weak-label test split: {weak_acc:.3f}")

    # --- Evaluate against TRUE ground truth (held-out, never used in training) ---
    gt_df = df[df["true_label"] != "UNKNOWN"]
    if len(gt_df) > 0:
        gt_preds = pipeline.predict(gt_df["text"])
        gt_acc = accuracy_score(gt_df["true_label"], gt_preds)
        report = classification_report(gt_df["true_label"], gt_preds, zero_division=0)
        print(f"\nAccuracy vs. TRUE ground-truth labels: {gt_acc:.3f}\n")
        print(report)

        with open(REPORT_DIR / "classification_report.txt", "w") as f:
            f.write(f"Accuracy vs weak-label test split: {weak_acc:.3f}\n")
            f.write(f"Accuracy vs true ground-truth labels: {gt_acc:.3f}\n\n")
            f.write(report)

    joblib.dump(pipeline, MODEL_DIR / "text_classifier.joblib")
    print(f"\nSaved trained classifier -> {MODEL_DIR / 'text_classifier.joblib'}")


if __name__ == "__main__":
    main()
