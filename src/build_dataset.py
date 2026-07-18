import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/labeled_sequences.csv")

CATEGORY_FILES = {
    "enzyme": "enzyme.tsv",
    "transporter": "transporter.tsv",
    "receptor": "receptor.tsv",
    "structural_protein": "structural_protein.tsv",
    "transcription_factor": "transcription_factor.tsv",
}

frames = []
for category, filename in CATEGORY_FILES.items():
    df = pd.read_csv(RAW_DIR / filename, sep="\t")
    df["function_category"] = category
    frames.append(df)
    print(f"{category}: {len(df)} entries")

combined = pd.concat(frames, ignore_index=True)

# Some proteins carry more than one function tag (e.g. a receptor tyrosine
# kinase is both a Receptor and catalytically active), so the same
# accession can appear in more than one category file. For a clean
# single-label classifier, keep only accessions that landed in exactly
# one category.
counts = combined["Entry"].value_counts()
overlap_count = (counts > 1).sum()
print(f"Dropping {overlap_count} accessions that matched more than one category")

clean = combined[combined["Entry"].isin(counts[counts == 1].index)].copy()
clean.to_csv(OUT_PATH, index=False)

print(f"Saved {len(clean)} labeled sequences to {OUT_PATH}")
print(clean["function_category"].value_counts())
