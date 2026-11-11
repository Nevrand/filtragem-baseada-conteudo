from __future__ import annotations

import numpy as np
from scipy import sparse


def _as_2d_array(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray | sparse.spmatrix) -> np.ndarray:
    """Calcula similaridade de cosseno entre vetor e matriz esparsa."""
    if isinstance(vector_b, sparse.spmatrix):
        left = _as_2d_array(vector_a)
        numerator = left @ vector_b.T
        row_norms = np.linalg.norm(left, axis=1)
        col_norms = np.sqrt(np.asarray(vector_b.multiply(vector_b).sum(axis=1)).ravel())
        denom = row_norms[:, None] * col_norms[None, :]
        denom = np.where(denom == 0, 1.0, denom)
        result = numerator / denom
        return np.asarray(result).ravel()

    left = _as_2d_array(vector_a)
    right = _as_2d_array(vector_b)
    if left.shape[1] != right.shape[1]:
        raise ValueError("As matrizes têm dimensões incompatíveis para cosseno.")

    denom = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(right, axis=1)[None, :]
    denom = np.where(denom == 0, 1.0, denom)
    result = (left @ right.T) / denom
    return np.asarray(result).ravel()


__all__ = ["cosine_similarity"]
