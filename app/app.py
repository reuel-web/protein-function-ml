import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import featurize_sequence, STANDARD_AMINO_ACIDS

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"


@st.cache_resource
def load_model_bundle():
    return joblib.load(MODEL_PATH)


bundle = load_model_bundle()
model = bundle["model"]
label_encoder = bundle["label_encoder"]
feature_cols = bundle["feature_cols"]

st.set_page_config(page_title="Protein Function Predictor", page_icon="🧬")
st.title("🧬 Protein Function Predictor")
st.markdown(
    "Paste an amino acid sequence (plain letters or FASTA format) to get a "
    "predicted function category — enzyme, transporter, receptor, structural "
    f"protein, or transcription factor — from a {bundle['model_name']} model "
    "trained on 426 features engineered from sequence composition alone. "
    "No structure, no alignment, no deep learning."
)

raw_input = st.text_area(
    "Amino acid sequence",
    height=150,
    placeholder=">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGH...",
)


def clean_sequence(raw_text):
    lines = raw_text.strip().splitlines()
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    sequence = "".join(lines)
    sequence = "".join(sequence.split())  # strip any remaining whitespace
    return sequence.upper()


if st.button("Predict function", type="primary"):
    sequence = clean_sequence(raw_input)

    if not sequence:
        st.error("Paste a sequence first.")
    elif not set(sequence).issubset(set(STANDARD_AMINO_ACIDS)):
        bad_chars = sorted(set(sequence) - set(STANDARD_AMINO_ACIDS))
        st.error(
            f"This sequence contains characters outside the 20 standard amino "
            f"acids: {', '.join(bad_chars)}. The model can't featurize these."
        )
    else:
        if len(sequence) < 50 or len(sequence) > 2000:
            st.warning(
                f"This sequence is {len(sequence)} residues long. The model was "
                "trained only on sequences between 50 and 2000 residues — treat "
                "this prediction as an extrapolation."
            )

        features = featurize_sequence(sequence)
        X_input = pd.DataFrame([features])[feature_cols].values
        proba = model.predict_proba(X_input)[0]
        predicted_idx = int(np.argmax(proba))
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

        st.subheader(f"Predicted: {predicted_label}")

        proba_df = pd.DataFrame({
            "category": label_encoder.classes_,
            "probability": proba,
        }).sort_values("probability", ascending=False)
        st.bar_chart(proba_df.set_index("category"))

        with st.expander("Computed physicochemical properties"):
            st.write({
                "length": features["length"],
                "molecular_weight (Da)": round(features["molecular_weight"], 1),
                "gravy (hydrophobicity)": round(features["gravy"], 3),
                "net_charge_ph7": round(features["net_charge_ph7"], 2),
                "isoelectric_point": round(features["isoelectric_point"], 2),
                "aromaticity": round(features["aromaticity"], 3),
            })
