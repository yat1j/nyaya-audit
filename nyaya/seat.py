"""
nyaya/seat.py

SEAT (Sentence Encoder Association Test) for Indian caste and religion bias.

Uses HuggingFace Inference API instead of loading model locally.
- No model download during build → image stays ~400 MB
- No RAM pressure → works on Railway free tier (512 MB)
- Same model quality → e5-large via API, identical embeddings to local
- Free tier: 30,000 requests/month → enough for demo + judges

Requires env var: HF_API_TOKEN
Get it free from: huggingface.co → Settings → Access Tokens → New Token (read)
"""

import os
import time
import requests
import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

# ── HuggingFace API config ─────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_API_TOKEN", "")

HF_API_URL = (
    "https://api-inference.huggingface.co"
    "/pipeline/feature-extraction"
    "/intfloat/multilingual-e5-large"
)


def get_embeddings(sentences: list) -> np.ndarray:
    """
    Get normalised embeddings from HuggingFace Inference API.

    Batches requests (max 50 per call — HF API limit).
    Retries up to 3 times if HF returns 503 "model loading".
    Normalises all embeddings to unit length (same as normalize_embeddings=True).

    Args:
        sentences: List of strings. Include "query: " prefix for e5 models.
                   All callers in this codebase already add the prefix.

    Returns:
        np.ndarray of shape (len(sentences), 1024), normalised.
    """
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_API_TOKEN environment variable is not set. "
            "Get a free token from huggingface.co → Settings → Access Tokens."
        )

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    batch_size = 50
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        # Retry loop — HF API returns 503 while model cold-starts (~20 sec)
        last_error = None
        for attempt in range(3):
            try:
                response = requests.post(
                    HF_API_URL,
                    headers=headers,
                    json={
                        "inputs": batch,
                        "options": {"wait_for_model": True},
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    batch_embs = np.array(response.json())
                    all_embeddings.append(batch_embs)
                    break

                elif response.status_code == 503:
                    print(
                        f"  HF model loading (attempt {attempt + 1}/3) "
                        f"— waiting 20s..."
                    )
                    time.sleep(20)
                    last_error = f"503 model loading after 3 attempts"

                else:
                    raise RuntimeError(
                        f"HuggingFace API error {response.status_code}: "
                        f"{response.text[:200]}"
                    )

            except requests.Timeout:
                last_error = f"Request timed out (attempt {attempt + 1}/3)"
                print(f"  HF API timeout, retrying... ({attempt + 1}/3)")
                time.sleep(10)

        else:
            # All 3 attempts failed
            raise RuntimeError(
                f"HuggingFace API failed after 3 attempts: {last_error}"
            )

    embeddings = np.vstack(all_embeddings)

    # Normalise to unit length (same as SentenceTransformer normalize_embeddings=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return embeddings / norms


def _cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    return float(1.0 - scipy_cosine(u, v))


def _association_score(
    name_emb: np.ndarray,
    attribute_X_embs: np.ndarray,
    attribute_Y_embs: np.ndarray,
) -> float:
    """
    SEAT association score for one name embedding.
    = mean similarity to capability words - mean similarity to limitation words
    Positive = name clusters nearer capability words (bias against the other group).
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
    Run SEAT and return Cohen's d effect size measuring embedding-space bias.

    Args:
        target_A:    Group A names (e.g. brahmin_surnames)
        target_B:    Group B names (e.g. dalit_obc_surnames)
        attribute_X: Positive attributes (e.g. capability_words)
        attribute_Y: Negative attributes (e.g. limitation_words)

    Returns:
        dict with d_score, mean_A, mean_B, interpretation

    Algorithm:
        1. Encode names with template "query: A person named {n} applied for the job."
        2. Encode attributes with template "query: This person is {w}."
        3. For each name: score = mean_sim(X_words) - mean_sim(Y_words)
        4. Cohen's d = (mean(A_scores) - mean(B_scores)) / std(all_scores)
        d > 0.5 = significant bias. d > 0.8 = severe bias.
    """
    # Build sentences with e5 query template
    A_sents = [f"query: A person named {n} applied for the job." for n in target_A]
    B_sents = [f"query: A person named {n} applied for the job." for n in target_B]
    X_sents = [f"query: This person is {w}." for w in attribute_X]
    Y_sents = [f"query: This person is {w}." for w in attribute_Y]

    print(f"    Encoding {len(A_sents)} Group A, {len(B_sents)} Group B names...")
    A_embs = get_embeddings(A_sents)
    B_embs = get_embeddings(B_sents)

    print(f"    Encoding {len(X_sents)} capability, {len(Y_sents)} limitation words...")
    X_embs = get_embeddings(X_sents)
    Y_embs = get_embeddings(Y_sents)

    # Per-name association scores
    A_scores = np.array([_association_score(e, X_embs, Y_embs) for e in A_embs])
    B_scores = np.array([_association_score(e, X_embs, Y_embs) for e in B_embs])

    # Cohen's d
    all_scores = np.concatenate([A_scores, B_scores])
    std_all = float(np.std(all_scores))

    if std_all < 1e-10:
        d = 0.0
    else:
        d = float((np.mean(A_scores) - np.mean(B_scores)) / std_all)

    abs_d = abs(d)
    if abs_d < 0.2:
        interp = "no significant bias"
    elif abs_d < 0.5:
        interp = "slight bias"
    elif abs_d < 0.8:
        interp = "moderate bias — significant"
    else:
        interp = "large bias — severe"

    return {
        "d_score":        round(d, 4),
        "mean_A":         round(float(np.mean(A_scores)), 4),
        "mean_B":         round(float(np.mean(B_scores)), 4),
        "interpretation": interp,
    }