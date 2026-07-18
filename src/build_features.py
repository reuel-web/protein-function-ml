import pandas as pd
from pathlib import Path
from features import featurize_sequence, STANDARD_AMINO_ACIDS

IN_PATH = Path("data/processed/labeled_sequences.csv")
OUT_PATH = Path("data/processed/features.csv")

df = pd.read_csv(IN_PATH)

valid_chars = set(STANDARD_AMINO_ACIDS)
is_standard = df["Sequence"].apply(lambda seq: set(seq).issubset(valid_chars))
dropped = (~is_standard).sum()
print(f"Dropping {dropped} sequences with non-standard amino acid codes")
df = df[is_standard].reset_index(drop=True)

feature_rows = [featurize_sequence(seq) for seq in df["Sequence"]]
features_df = pd.DataFrame(feature_rows)

result = pd.concat([df[["Entry", "function_category"]], features_df], axis=1)
result.to_csv(OUT_PATH, index=False)

print(f"Saved {result.shape[0]} rows x {result.shape[1]} columns to {OUT_PATH}")
print(result["function_category"].value_counts())
