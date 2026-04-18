"""
nyaya/debias.py

Hard Debiasing via subspace projection.
Published: Bolukbasi et al., NeurIPS 2016.

CRITICAL INSIGHT FOR e5-large:
e5-large is 1024-dimensional. Bias is diffuse — spread across
many dimensions, not concentrated in one. Removing a single
bias direction (which explains only ~5% of variance) barely
changes anything.

The correct approach: remove a SUBSPACE of top N bias directions.
We use the top 20 PCA components of the difference vectors.
Together they cover the majority of the bias signal.

This is called "subspace debiasing" and is the proper extension
of Hard Debiasing for large embedding models.
"""

import numpy as np
from sklearn.decomposition import PCA
from nyaya.seat import get_embeddings


def get_bias_subspace(group_A_names: list,
                       group_B_names: list,
                       n_components: int = 20) -> np.ndarray:
    """
    Compute the bias SUBSPACE — top N directions that separate
    Group A from Group B in embedding space.

    Returns:
        np.ndarray shape (n_components, dim) — matrix of bias directions
        Each row is one bias direction (unit vector).
    """
    template = "query: A person named {} applied for the job."

    print(f"  Encoding {len(group_A_names)} Group A names...")
    A_embs = get_embeddings([template.format(n) for n in group_A_names])

    print(f"  Encoding {len(group_B_names)} Group B names...")
    B_embs = get_embeddings([template.format(n) for n in group_B_names])

    min_len = min(len(A_embs), len(B_embs))
    diffs = A_embs[:min_len] - B_embs[:min_len]

    # Use min(n_components, available) components
    n_comp = min(n_components, min_len - 1, diffs.shape[1])

    pca = PCA(n_components=n_comp)
    pca.fit(diffs)

    total_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Top {n_comp} directions explain "
          f"{total_explained:.1f}% of group variance.")

    # Each row is a unit bias direction
    return pca.components_  # shape: (n_comp, dim)


def hard_debias_subspace(embeddings: np.ndarray,
                          bias_subspace: np.ndarray) -> np.ndarray:
    """
    Remove ALL bias directions in the subspace from embeddings.

    For each bias direction in the subspace:
        e = e - (e · bias_dir) * bias_dir

    Applied sequentially for each direction.
    This is the correct multi-direction extension of Hard Debiasing.

    Parameters:
        embeddings    : np.ndarray shape (n, dim)
        bias_subspace : np.ndarray shape (n_components, dim)

    Returns:
        np.ndarray shape (n, dim) — debiased and re-normalized
    """
    result = embeddings.copy()

    for bias_dir in bias_subspace:
        # Ensure unit vector
        bias_dir = bias_dir / np.linalg.norm(bias_dir)
        # Project out this direction
        projections = result @ bias_dir
        result = result - np.outer(projections, bias_dir)

    # Re-normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return result / norms


# Keep single-direction version for compatibility
def get_bias_direction(group_A_names: list,
                        group_B_names: list) -> np.ndarray:
    """Single direction — kept for API compatibility."""
    subspace = get_bias_subspace(group_A_names, group_B_names,
                                  n_components=1)
    return subspace[0]


def hard_debias(embeddings: np.ndarray,
                bias_direction: np.ndarray) -> np.ndarray:
    """Single direction — kept for API compatibility."""
    bias_dir = bias_direction / np.linalg.norm(bias_direction)
    projections = embeddings @ bias_dir
    debiased = embeddings - np.outer(projections, bias_dir)
    norms = np.linalg.norm(debiased, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return debiased / norms


def compute_bias_subspaces(brahmin_names, dalit_names,
                            hindu_names, muslim_names,
                            n_components: int = 20) -> tuple:
    """
    Compute caste and religion bias subspaces.
    Returns two matrices of shape (n_components, 1024).
    """
    print("\nComputing CASTE bias subspace (Brahmin vs Dalit)...")
    caste_subspace = get_bias_subspace(
        brahmin_names, dalit_names, n_components)

    print("\nComputing RELIGION bias subspace (Hindu vs Muslim)...")
    religion_subspace = get_bias_subspace(
        hindu_names, muslim_names, n_components)

    return caste_subspace, religion_subspace