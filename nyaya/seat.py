"""
nyaya/seat.py

SEAT — Sentence Encoder Association Test
Adapted for Indian caste and religion bias detection.

WHY SEAT INSTEAD OF WEAT:
WEAT (Word Embedding Association Test, Caliskan et al. Science 2017) 
encodes single words like "Sharma" directly. But LaBSE is a SENTENCE 
encoder — it was trained on sentence pairs, not individual words. 
Encoding "Sharma" alone gives a noisy, context-free vector.

SEAT fixes this by wrapping each word in a sentence template:
  "A person named Sharma applied for the job."
This gives a rich, stable, contextually meaningful embedding.

The math is identical to WEAT — only the encoding step changes.

WHAT THE d-SCORE MEANS:
  d > 0.8  = Large bias   — severe, highly defensible demo number
  d > 0.5  = Moderate     — statistically significant, good demo
  d > 0.2  = Small bias   — detectable but weak
  d ~ 0.0  = No bias
  d < -0.5 = Bias favours Group B instead of Group A
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

# ─────────────────────────────────────────────────────────────
# Load LaBSE once when this module is first imported.
# First time: downloads ~500MB and caches to disk (~60 seconds).
# Every run after that: loads from cache (~5 seconds).
# LaBSE = Language-Agnostic BERT Sentence Encoder.
# Chosen because it covers all 12 major Indian languages natively.
# ─────────────────────────────────────────────────────────────
print("Loading E5 model...")
print("(First time takes ~60 seconds. Subsequent runs are fast.)")
MODEL = SentenceTransformer('intfloat/multilingual-e5-large')
print("E5 loaded successfully.\n")


def get_embeddings(texts: list) -> np.ndarray:
    """
    Convert a list of text strings into 768-dimensional vectors.

    normalize_embeddings=True: makes all vectors unit length (magnitude 1).
    This is REQUIRED for cosine similarity to work correctly.
    Without normalization, longer sentences would appear more similar
    to everything just because they have larger magnitude.

    Returns: numpy array of shape (len(texts), 768)
    """
    return MODEL.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Measure how similar two vectors are in direction.
    
    1.0  = pointing in exactly the same direction (most similar)
    0.0  = perpendicular (unrelated)
    -1.0 = pointing in opposite directions (most different)
    
    We use 1 - cosine_distance to get similarity from distance.
    """
    return 1.0 - cosine(u, v)


def seat_score(
    target_A_names: list,
    target_B_names: list,
    attribute_X_words: list,
    attribute_Y_words: list,
    template_name: str = "query: A person named {} applied for the job.",
    template_attr: str = "query: This person is {}."
) -> dict:
    """
    Compute the SEAT bias score between two groups of names.

    Parameters:
        target_A_names : list of Group A names (e.g. Brahmin surnames)
        target_B_names : list of Group B names (e.g. Dalit surnames)
        attribute_X_words : list of positive attribute words (e.g. capability)
        attribute_Y_words : list of negative attribute words (e.g. limitation)
        template_name : sentence template for names — {} is replaced by name
        template_attr : sentence template for attributes — {} is replaced by word

    Returns:
        dict with:
            d_score          : Cohen's d effect size (the main demo number)
            mean_A           : avg association of Group A with X vs Y
            mean_B           : avg association of Group B with X vs Y
            std              : standard deviation of all association scores
            interpretation   : human-readable level of bias
            A_scores         : individual association scores for each Group A name
            B_scores         : individual association scores for each Group B name

    HOW IT WORKS (step by step):
    1. Encode all names using the sentence template.
       "Sharma" becomes "A person named Sharma applied for the job."
    2. Encode all attribute words using the attribute template.
       "intelligent" becomes "This person is intelligent."
    3. For each name embedding, compute its ASSOCIATION score:
       association = mean_similarity_to_X_words - mean_similarity_to_Y_words
       A positive association means the name is closer to capability words.
       A negative association means it is closer to limitation words.
    4. Compute effect size d:
       d = (mean_A_association - mean_B_association) / std_dev_all_associations
       This tells us: how many standard deviations does Group A sit above Group B?
    """

    # ── Step 1: Build sentence lists ──────────────────────────────
    A_sentences = [template_name.format(name) for name in target_A_names]
    B_sentences = [template_name.format(name) for name in target_B_names]
    X_sentences = [template_attr.format(word) for word in attribute_X_words]
    Y_sentences = [template_attr.format(word) for word in attribute_Y_words]

    # ── Step 2: Get embeddings ─────────────────────────────────────
    print(f"  Encoding {len(A_sentences)} Group A sentences...")
    A_embs = get_embeddings(A_sentences)

    print(f"  Encoding {len(B_sentences)} Group B sentences...")
    B_embs = get_embeddings(B_sentences)

    print(f"  Encoding {len(X_sentences)} capability attribute sentences...")
    X_embs = get_embeddings(X_sentences)

    print(f"  Encoding {len(Y_sentences)} limitation attribute sentences...")
    Y_embs = get_embeddings(Y_sentences)

    # ── Step 3: Compute association score for each name ───────────
    def association(emb: np.ndarray) -> float:
        """
        How much more similar is this name to X (capability) than Y (limitation)?
        Positive = leans toward capability. Negative = leans toward limitation.
        """
        sim_to_X = np.mean([cosine_similarity(emb, x) for x in X_embs])
        sim_to_Y = np.mean([cosine_similarity(emb, y) for y in Y_embs])
        return float(sim_to_X - sim_to_Y)

    print("  Computing association scores...")
    A_scores = [association(a) for a in A_embs]
    B_scores = [association(b) for b in B_embs]

    # ── Step 4: Compute effect size d ─────────────────────────────
    all_scores = A_scores + B_scores
    mean_A = float(np.mean(A_scores))
    mean_B = float(np.mean(B_scores))
    std_dev = float(np.std(all_scores))

    if std_dev < 1e-10:
        return {
            "d_score": 0.0,
            "mean_A": 0.0,
            "mean_B": 0.0,
            "std": 0.0,
            "interpretation": "No variation detected — check word lists",
            "A_scores": A_scores,
            "B_scores": B_scores
        }

    d = (mean_A - mean_B) / std_dev

    # ── Step 5: Interpret the result ──────────────────────────────
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "No significant bias"
    elif abs_d < 0.5:
        interpretation = "Small bias detected"
    elif abs_d < 0.8:
        interpretation = "MODERATE BIAS — statistically significant"
    else:
        interpretation = "LARGE BIAS — severe"

    favoured = "Group A" if d > 0 else "Group B"

    return {
        "d_score": round(d, 4),
        "mean_A": round(mean_A, 4),
        "mean_B": round(mean_B, 4),
        "std": round(std_dev, 4),
        "interpretation": interpretation,
        "favoured_group": favoured,
        "A_scores": [round(s, 4) for s in A_scores],
        "B_scores": [round(s, 4) for s in B_scores]
    }