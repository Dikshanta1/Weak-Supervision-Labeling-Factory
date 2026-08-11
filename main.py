"""
main.py
-------
Runs the full Weak Supervision Data Labeling Factory pipeline end to end:

  1. Generate/load raw unlabeled data
  2. Apply labeling functions + train Snorkel LabelModel -> weak labels
  3. Train downstream scikit-learn classifier on weak labels
  4. Compare LabelModel vs. naive majority vote baseline

Usage:
    python main.py
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"

STEPS = [
    ("Step 1/4: Preparing data", "data_prep.py"),
    ("Step 2/4: Training Snorkel LabelModel", "train_label_model.py"),
    ("Step 3/4: Training downstream classifier", "train_classifier.py"),
    ("Step 4/4: Evaluating vs. majority-vote baseline", "evaluate_baseline.py"),
]


def main():
    for title, script in STEPS:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        result = subprocess.run(
            [sys.executable, str(SRC / script)],
            cwd=str(SRC),
        )
        if result.returncode != 0:
            print(f"\nPipeline stopped: {script} failed.")
            sys.exit(1)

    print("\nPipeline complete. See /data, /models, and /reports for outputs.")


if __name__ == "__main__":
    main()
