"""
nyaya/seat.py

Sentence Encoder Association Test (SEAT) adapted for Indian caste
and religion bias detection.

Uses intfloat/multilingual-e5-small — 70 MB RAM vs 1.5 GB for e5-large.
Fits Railway free tier (512 MB RAM limit). d-scores still significant.

Model loaded ONCE at module level — stays in memory between requests.
All functions use the same "query: ..." prefix required by e5 models.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as scipy_cosine

# ── Load model once at import time ────────────────────────────────────────────
# e5-small: ~70 MB on disk, ~200 MB RAM — fits Railway free tier
# e5-large: ~2.1 GB on disk, ~1.5 GB RAM — crashes Railway free tier
print("Loading multilingual-e5-small model...")
_MODEL = SentenceTransformer("intfloat/multilingual-e5-small")
print("Model loaded.")


def get_embeddings(sentences: list) -> np.ndarray:
    """
    Encode a list of sentences into normalised embeddings.

    Args:
        sentences: List of strings. Must already include the "query: " prefix
                   if using e5 models. All callers in this codebase add it.

    Returns:
        np.ndarray of shape (len(sentences), embedding_dim), normalised.
    """
    embeddings = _MODEL.encode(
        sentences,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.array(embeddings)


def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors (1 - cosine distance)."""
    return float(1.0 - scipy_cosine(u, v))


def _association_score(
    name_emb: np.ndarray,
    attribute_X_embs: np.ndarray,
    attribute_Y_embs: np.ndarray,
) -> float:
    """
    Compute SEAT association score for one name embedding.

    Association = mean cosine similarity to X attributes
                - mean cosine similarity to Y attributes

    Positive = name is more associated with X (capability words).
    Negative = name is more associated with Y (limitation words).
    """
    sim_X = float(np.mean([_cosine_sim(name_emb, a) for a in attribute_X_embs]))
    sim_Y = float(np.mean([_cosine_sim(name_emb, a) for a in attribute_Y_embs]))
    return sim_X - sim_Y


def seat_score(
    target_A: list,
    target_B: list,
    attribute_X: list,
    attribute_Y: list,
) -> dict:
    """
    Run SEAT and return Cohen's d-score measuring bias between two name groups.

    Args:
        target_A:    List of names for Group A (e.g. brahmin_surnames)
        target_B:    List of names for Group B (e.g. dalit_obc_surnames)
        attribute_X: List of positive attribute words (e.g. capability_words)
        attribute_Y: List of negative attribute words (e.g. limitation_words)

    Returns:
        dict with keys:
            d_score         — Cohen's d effect size (float)
            mean_A          — mean association score for Group A
            mean_B          — mean association score for Group B
            interpretation  — "no bias" / "slight" / "moderate" / "significant"

    How it works:
        1. Encode all names with template "query: A person named {n} applied for the job."
        2. Encode all attribute words with template "query: This person is {w}."
        3. For each name compute: mean_sim(capability_words) - mean_sim(limitation_words)
        4. Cohen's d = (mean(Group_A_scores) - mean(Group_B_scores)) / std(all_scores)
        A high d-score means Group A names are geometrically closer to capability words
        in embedding space than Group B names — structural bias.
    """
    # Build sentences using e5 query template
    A_sents = [f"query: A person named {n} applied for the job." for n in target_A]
    B_sents = [f"query: A person named {n} applied for the job." for n in target_B]
    X_sents = [f"query: This person is {w}." for w in attribute_X]
    Y_sents = [f"query: This person is {w}." for w in attribute_Y]

    # Encode everything
    A_embs = get_embeddings(A_sents)
    B_embs = get_embeddings(B_sents)
    X_embs = get_embeddings(X_sents)
    Y_embs = get_embeddings(Y_sents)

    # Compute per-name association scores
    A_scores = np.array([_association_score(e, X_embs, Y_embs) for e in A_embs])
    B_scores = np.array([_association_score(e, X_embs, Y_embs) for e in B_embs])

    # Cohen's d effect size
    all_scores = np.concatenate([A_scores, B_scores])
    std_all = float(np.std(all_scores))

    if std_all < 1e-10:
        d = 0.0
    else:
        d = float((np.mean(A_scores) - np.mean(B_scores)) / std_all)

    # Interpretation thresholds from IndiBias NAACL 2024
    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "no significant bias"
    elif abs_d < 0.5:
        interp = "slight bias"
    elif abs_d < 0.8:
        interp = "moderate bias"
    else:
        interp = "significant bias"

    return {
        "d_score":        round(d, 4),
        "mean_A":         round(float(np.mean(A_scores)), 4),
        "mean_B":         round(float(np.mean(B_scores)), 4),
        "interpretation": interp,
    }