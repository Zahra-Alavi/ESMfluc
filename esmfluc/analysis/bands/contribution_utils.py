"""Reconstruct column-wise evidence from attention and contributions."""
import numpy as np

def reconstruct_evidence(contribution: np.ndarray, attention: np.ndarray):
    denominator = np.sum(attention * attention, axis=0)
    evidence = np.divide(
        np.sum(attention * contribution, axis=0),
        denominator,
        out=np.zeros(contribution.shape[1], dtype=float),
        where=denominator > 0,
    )
    error = float(np.max(np.abs(contribution - attention * evidence[None, :])))
    return evidence, error
