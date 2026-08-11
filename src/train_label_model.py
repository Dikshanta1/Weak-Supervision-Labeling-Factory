"""
train_label_model.py
---------------------
Applies all labeling functions to the raw corpus, analyzes LF quality
(coverage, overlap, conflict), then trains Snorkel's generative LabelModel
to combine the noisy LF votes into a single set of probabilistic labels --
WITHOUT using any ground-truth labels.

Outputs:
  data/lf_analysis.csv        -> per-LF coverage/overlap/conflict stats
  data/labeled_reviews.parquet    -> original text + soft probabilistic labels
                                  + hard (argmax) label for downstream training
"""

from pathlib import Path
import numpy as np
import pandas as pd

from snorkel.labeling import PandasLFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

from labeling_functions import ALL_LFS, LABEL_NAMES, ABSTAIN

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    df = pd.read_parquet(DATA_DIR / "raw_reviews.parquet")

    # --- Apply LFs to every row -------------------------------------------------
    applier = PandasLFApplier(lfs=ALL_LFS)
    L_train = applier.apply(df=df)

    # --- LF quality report -------------------------------------------------------
    analysis = LFAnalysis(L=L_train, lfs=ALL_LFS).lf_summary()
    analysis.to_csv(DATA_DIR / "lf_analysis.csv")
    print("=== Labeling Function Analysis ===")
    print(analysis[["Coverage", "Overlaps", "Conflicts"]])

    coverage = (L_train != ABSTAIN).any(axis=1).mean()
    print(f"\nOverall row coverage (at least one LF fired): {coverage:.1%}")

    # --- Train LabelModel (or use Majority Vote) --------------------------------
    # Because of massive class imbalance between Positive (60% coverage) and 
    # others (1%), the generative LabelModel can sometimes collapse. We use 
    # MajorityLabelVoter for robust aggregation at this scale.
    from snorkel.labeling.model import MajorityLabelVoter
    
    label_model = MajorityLabelVoter(cardinality=6)
    
    print("Applying MajorityLabelVoter to generate probabilistic labels...")
    probs = label_model.predict_proba(L_train)
    preds = label_model.predict(L=L_train)
    
    df["label_probs"] = list(probs)
    df["weak_label"] = preds
    df["weak_label_name"] = df["weak_label"].map(LABEL_NAMES)

    labeled_count = (df["weak_label"] != ABSTAIN).sum()
    print(f"\nLabelModel assigned a non-abstain label to "
          f"{labeled_count}/{len(df)} rows ({labeled_count/len(df):.1%})")

    # Keep only rows the LabelModel was confident enough to label
    df_labeled = df[df["weak_label"] != ABSTAIN].copy()
    df_labeled.to_parquet(DATA_DIR / "labeled_reviews.parquet", index=False)
    print(f"Saved {len(df_labeled):,} weakly-labeled rows -> "
          f"{DATA_DIR / 'labeled_reviews.parquet'}")



if __name__ == "__main__":
    main()
