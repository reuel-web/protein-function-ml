"""Reusable sequence -> feature functions for the protein function project."""

from itertools import product
from Bio.SeqUtils.ProtParam import ProteinAnalysis

STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES = ["".join(pair) for pair in product(STANDARD_AMINO_ACIDS, repeat=2)]


def amino_acid_composition(sequence):
    """Fraction of each of the 20 standard amino acids in the sequence."""
    length = len(sequence)
    return {aa: sequence.count(aa) / length for aa in STANDARD_AMINO_ACIDS}


def dipeptide_composition(sequence):
    """Fraction of each of the 400 possible adjacent amino-acid pairs."""
    total_windows = len(sequence) - 1
    counts = {dp: 0 for dp in DIPEPTIDES}
    for i in range(total_windows):
        counts[sequence[i:i + 2]] += 1
    return {dp: count / total_windows for dp, count in counts.items()}


def physicochemical_properties(sequence):
    """Molecular weight, hydrophobicity, charge, isoelectric point, aromaticity."""
    analysed = ProteinAnalysis(sequence)
    return {
        "length": len(sequence),
        "molecular_weight": analysed.molecular_weight(),
        "gravy": analysed.gravy(),
        "net_charge_ph7": analysed.charge_at_pH(7.0),
        "isoelectric_point": analysed.isoelectric_point(),
        "aromaticity": analysed.aromaticity(),
    }


def featurize_sequence(sequence):
    """Combine all three feature groups into one flat dict for a single sequence."""
    features = {}
    features.update(amino_acid_composition(sequence))
    features.update(dipeptide_composition(sequence))
    features.update(physicochemical_properties(sequence))
    return features
