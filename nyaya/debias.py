"""
nyaya/debias.py

Hard Debiasing via subspace projection (Bolukbasi et al., NeurIPS 2016).

Takes a set of embeddings and removes the bias direction(s) from all of them.
No pkl files. No external dependencies beyond numpy and sklearn.
The bias subspace is computed fresh at startup from HF API embeddings.
"""

import numpy as np
from sklearn.decomposition import PCA


def compute_bias_subspace(
    group_A_embs: np.ndarray,
    group_B_embs: np.ndarray,
    n_directions: int = 10,
) -> np.ndarray:
    """
    Identify the bias subspace via PCA on difference vectors.

    Args:
        group_A_embs: Embeddings for Group A (e.g. Brahmin names). Shape (n, dim).
        group_B_embs: Embeddings for Group B (e.g. Dalit names). Shape (m, dim).
        n_directions: Number of PCA directions to keep. More = more thorough
                      debiasing but risks removing useful semantic content.
                      10 covers ~50-70% of bias variance.

    Returns:
        np.ndarray of shape (n_directions, dim) — the bias directions.

    How it works:
        For each name pair (A_i, B_i), compute A_i - B_i.
        These difference vectors point in the "bias direction" in embedding space.
        PCA finds the principal axes of this difference cloud.
        We remove these axes from all embeddings.
    """
    n = min(len(group_A_embs), len(group_B_embs))
    diffs = group_A_embs[:n] - group_B_embs[:n]

    n_components = min(n_directions, n, diffs.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(diffs)

    # PCA components are already unit-length — use directly as bias directions
    return pca.components_  # shape (n_components, dim)


def hard_debias(
    embeddings: np.ndarray,
    bias_subspace: np.ndarray,
) -> np.ndarray:
    """
    Remove bias directions from all embeddings via subspace projection.

    For each bias direction d in the subspace:
        embedding = embedding - (embedding · d) * d

    This zeroes out the component of every embedding that lies along
    the bias direction, preserving all other semantic dimensions.

    Args:
        embeddings:    np.ndarray of shape (n, dim)
        bias_subspace: np.ndarray of shape (n_directions, dim)

    Returns:
        np.ndarray of shape (n, dim), re-normalised to unit length.
    """
    debiased = embeddings.copy()

    for direction in bias_subspace:
        # Project out this bias direction from all embeddings at once
        # projection of each row onto direction = (embs @ dir)
        # component to remove = outer product with the direction
        projections = debiased @ direction          # shape (n,)
        debiased = debiased - np.outer(projections, direction)

    # Re-normalise to unit length after projection
    norms = np.linalg.norm(debiased, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return debiased / norms


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the mean embedding (centroid) of a group.
    Used to build capability and limitation word profiles.

    Args:
        embeddings: np.ndarray of shape (n, dim)

    Returns:
        np.ndarray of shape (dim,), unit-normalised.
    """
    centroid = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-10:
        return centroid
    return centroid / norm