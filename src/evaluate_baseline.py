"""
evaluate_baseline.py
---------------------
Quantifies WHY the generative LabelModel is better than naive majority vote
over the labeling functions. This comparison is the single most compelling
number for a resume bullet / interview talking point:

    "Snorkel's LabelModel improved weak-label accuracy by X points over
     simple majority vote by learning per-LF reliability weights."
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel, MajorityLabelVoter

from labeling_functions import ALL_LFS, LABEL_NAMES, ABSTAIN

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    df = pd.read_parquet(DATA_DIR / "raw_reviews.parquet")
    gt_df = df[df["true_label"] != "UNKNOWN"].copy()

    applier = PandasLFApplier(lfs=ALL_LFS)
    L = applier.apply(df=gt_df)

    # Baseline: majority vote (ties broken randomly)
    majority_model = MajorityLabelVoter(cardinality=6)
    maj_preds = majority_model.predict(L)

    # Snorkel's learned generative model
    label_model = LabelModel(cardinality=6, verbose=False)
    label_model.fit(L_train=L, n_epochs=500, seed=42)
    lm_preds = label_model.predict(L, tie_break_policy="abstain")

    name_lookup = np.vectorize(lambda i: LABEL_NAMES.get(i, "ABSTAIN"))
    maj_preds_named = name_lookup(maj_preds)
    lm_preds_named = name_lookup(lm_preds)

    # Only score rows where the model actually made a prediction
    def scored_accuracy(preds_named, true_labels):
        mask = preds_named != "ABSTAIN"
        if mask.sum() == 0:
            return 0.0, 0
        return accuracy_score(true_labels[mask], preds_named[mask]), mask.sum()

    maj_acc, maj_n = scored_accuracy(maj_preds_named, gt_df["true_label"].values)
    lm_acc, lm_n = scored_accuracy(lm_preds_named, gt_df["true_label"].values)

    print(f"Majority Vote   -> accuracy: {maj_acc:.3f}  (scored {maj_n}/{len(gt_df)} rows)")
    print(f"Snorkel LabelModel -> accuracy: {lm_acc:.3f}  (scored {lm_n}/{len(gt_df)} rows)")
    print(f"\nImprovement from generative modeling: {(lm_acc - maj_acc) * 100:+.1f} points")

    with open(DATA_DIR.parent / "reports" / "baseline_comparison.txt", "w") as f:
        f.write(f"Majority Vote accuracy: {maj_acc:.3f} (n={maj_n})\n")
        f.write(f"Snorkel LabelModel accuracy: {lm_acc:.3f} (n={lm_n})\n")
        f.write(f"Improvement: {(lm_acc - maj_acc) * 100:+.1f} points\n")


if __name__ == "__main__":
    main()
