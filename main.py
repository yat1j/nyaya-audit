"""
main.py — Nyaya FastAPI Server

NO PKL FILES. NO KeyError. EVER.

All bias profiles and subspaces are computed at startup from word_lists.json
using the HuggingFace API. Results stored in memory. Every restart recomputes
(takes 60-90 seconds on startup — acceptable, server then runs indefinitely).

Only file needed in data/: word_lists.json

Endpoints:
  GET  /health       — health check, returns 200 immediately
  POST /audit        — full bias audit on uploaded CSV
  POST /retroactive  — retroactive audit: who gets different outcome
"""

import os
import json
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nyaya.seat import get_embeddings, seat_score
from nyaya.debias import compute_bias_subspace, hard_debias, compute_centroid
from nyaya.gemini_explain import get_explanation


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Nyaya Bias Audit API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: compute everything from word_lists.json via HF API ────────────────
# No pkl files. No KeyError. Runs once on container start (~60-90 seconds).
# All results stored in module-level variables — shared across all requests.

print("=" * 55)
print("NYAYA API STARTUP")
print("=" * 55)

# Step 1 — Load word lists (only file needed in data/)
print("\n[1/6] Loading word_lists.json...")
with open("data/word_lists.json") as f:
    WORDS = json.load(f)

brahmin_names    = WORDS["brahmin_surnames"]
dalit_names      = WORDS["dalit_obc_surnames"]
hindu_names      = WORDS["hindu_names"]
muslim_names     = WORDS["muslim_names"]
capability_words = WORDS["capability_words"]
limitation_words = WORDS["limitation_words"]

print(f"    Brahmin: {len(brahmin_names)} | Dalit: {len(dalit_names)}")
print(f"    Hindu: {len(hindu_names)} | Muslim: {len(muslim_names)}")
print(f"    Capability: {len(capability_words)} | Limitation: {len(limitation_words)}")

# Step 2 — Encode word groups via HF API
print("\n[2/6] Encoding name groups (HuggingFace API)...")

brahmin_embs = get_embeddings(
    [f"query: A person named {n} applied for the job." for n in brahmin_names]
)
dalit_embs = get_embeddings(
    [f"query: A person named {n} applied for the job." for n in dalit_names]
)
hindu_embs = get_embeddings(
    [f"query: A person named {n} applied for the job." for n in hindu_names]
)
muslim_embs = get_embeddings(
    [f"query: A person named {n} applied for the job." for n in muslim_names]
)
print(f"    Name embeddings shape: {brahmin_embs.shape}")

print("\n[3/6] Encoding attribute words (HuggingFace API)...")
cap_embs = get_embeddings(
    [f"query: This person is {w}." for w in capability_words]
)
lim_embs = get_embeddings(
    [f"query: This person is {w}." for w in limitation_words]
)
print(f"    Attribute embeddings shape: {cap_embs.shape}")

# Step 3 — Compute BIASED profiles (centroids before debiasing)
print("\n[4/6] Computing biased profiles (pre-debiasing centroids)...")
CAP_PROFILE_BIASED = compute_centroid(cap_embs)
LIM_PROFILE_BIASED = compute_centroid(lim_embs)

# Step 4 — Compute bias subspaces via PCA on difference vectors
print("\n[5/6] Computing bias subspaces (PCA on Brahmin-Dalit differences)...")
CASTE_SUBSPACE    = compute_bias_subspace(brahmin_embs, dalit_embs,    n_directions=10)
RELIGION_SUBSPACE = compute_bias_subspace(hindu_embs,   muslim_embs,   n_directions=10)
print(f"    Caste subspace:    {CASTE_SUBSPACE.shape}")
print(f"    Religion subspace: {RELIGION_SUBSPACE.shape}")

# Step 5 — Compute DEBIASED profiles (centroids after debiasing)
print("\n[6/6] Computing debiased profiles...")
cap_embs_debiased  = hard_debias(cap_embs, CASTE_SUBSPACE)
cap_embs_debiased  = hard_debias(cap_embs_debiased, RELIGION_SUBSPACE)
lim_embs_debiased  = hard_debias(lim_embs, CASTE_SUBSPACE)
lim_embs_debiased  = hard_debias(lim_embs_debiased, RELIGION_SUBSPACE)

CAP_PROFILE_DEBIASED = compute_centroid(cap_embs_debiased)
LIM_PROFILE_DEBIASED = compute_centroid(lim_embs_debiased)

print("\n" + "=" * 55)
print("STARTUP COMPLETE — all profiles ready in memory")
print("API accepting requests.")
print("=" * 55 + "\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine_sim(u, v):
    return float(1.0 - scipy_cosine(u, v))


def _normalise(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.ones_like(scores) * 0.5
    return (scores - mn) / (mx - mn)


def _read_csv(contents: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as t:
        t.write(contents)
        path = t.name
    df = pd.read_csv(path)
    os.unlink(path)
    return df


def _parity(df: pd.DataFrame, score_col: str) -> dict:
    """Demographic parity ratio between brahmin and dalit groups."""
    if "true_group" not in df.columns:
        return {"parity": None, "passes_fairness": None, "group_shortlist_rates": {}}

    scores    = df[score_col].values
    threshold = np.percentile(scores, 50)
    shortlist = (scores >= threshold).astype(int)

    rates = {}
    for g in ["brahmin", "dalit"]:
        mask = (df["true_group"] == g).values
        if mask.sum() > 0:
            rates[g] = float(shortlist[mask].mean())

    if len(rates) < 2:
        return {
            "parity": None,
            "passes_fairness": None,
            "group_shortlist_rates": {g: round(r * 100, 1) for g, r in rates.items()},
        }

    parity = min(rates.values()) / max(rates.values())
    return {
        "parity":             round(parity, 4),
        "passes_fairness":    bool(parity >= 0.80),
        "group_shortlist_rates": {g: round(r * 100, 1) for g, r in rates.items()},
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Instant health check — no model loading, returns immediately."""
    return {
        "status":  "running",
        "model":   "intfloat/multilingual-e5-large (via HF API)",
        "service": "Nyaya Bias Audit API",
        "version": "1.0.0",
        "profiles": "loaded",
    }


@app.post("/audit")
async def audit(file: UploadFile = File(...)):
    """
    Full bias audit on a CSV of decisions.

    Required columns: name, text
    Optional columns: id, true_group (brahmin/dalit)

    Steps:
      1. SEAT caste d-score BEFORE debiasing (uses HF API)
      2. SEAT religion d-score BEFORE debiasing
      3. Score each row against BIASED profiles
      4. Demographic parity BEFORE
      5. Debias name embeddings (subspace projection)
      6. Score each row against DEBIASED profiles
      7. Demographic parity AFTER
      8. SEAT caste d-score AFTER (should drop significantly)
      9. SEAT religion d-score AFTER
      10. Gemini explanation
    """
    # Read CSV
    try:
        df = _read_csv(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Cannot read CSV: {e}")

    for col in ["name", "text"]:
        if col not in df.columns:
            raise HTTPException(
                400,
                f"Missing column '{col}'. Found: {list(df.columns)}"
            )

    print(f"\n/audit — {len(df)} rows")

    # SEAT before
    print("  SEAT caste (before)...")
    caste_before = seat_score(
        brahmin_names, dalit_names, capability_words, limitation_words
    )
    print(f"    d = {caste_before['d_score']} ({caste_before['interpretation']})")

    print("  SEAT religion (before)...")
    religion_before = seat_score(
        hindu_names, muslim_names, capability_words, limitation_words
    )
    print(f"    d = {religion_before['d_score']}")

    # Encode name column
    print("  Encoding CSV names via HF API...")
    name_sents = [
        f"query: A person named {str(n)} applied for the job."
        for n in df["name"].tolist()
    ]
    name_embs = get_embeddings(name_sents)

    # Biased scoring
    biased_cap = np.array([_cosine_sim(e, CAP_PROFILE_BIASED) for e in name_embs])
    biased_lim = np.array([_cosine_sim(e, LIM_PROFILE_BIASED) for e in name_embs])
    df["biased_score"] = _normalise(biased_cap - biased_lim)
    parity_before = _parity(df, "biased_score")

    # Hard debiasing
    print("  Applying hard debiasing...")
    name_embs_fair = hard_debias(name_embs, CASTE_SUBSPACE)
    name_embs_fair = hard_debias(name_embs_fair, RELIGION_SUBSPACE)

    # Fair scoring
    fair_cap = np.array([_cosine_sim(e, CAP_PROFILE_DEBIASED) for e in name_embs_fair])
    fair_lim = np.array([_cosine_sim(e, LIM_PROFILE_DEBIASED) for e in name_embs_fair])
    df["fair_score"] = _normalise(fair_cap - fair_lim)
    parity_after = _parity(df, "fair_score")

    # SEAT after
    print("  SEAT caste (after)...")
    caste_after = seat_score(
        brahmin_names, dalit_names, capability_words, limitation_words
    )
    print("  SEAT religion (after)...")
    religion_after = seat_score(
        hindu_names, muslim_names, capability_words, limitation_words
    )

    # Gemini
    print("  Gemini explanation...")
    explanation = get_explanation(
        caste_d_before = caste_before["d_score"],
        caste_d_after  = caste_after["d_score"],
        parity_before  = parity_before.get("parity") or 0.0,
        parity_after   = parity_after.get("parity")  or 0.0,
    )

    print(
        f"  Done. d: {caste_before['d_score']}→{caste_after['d_score']} | "
        f"parity: {parity_before.get('parity')}→{parity_after.get('parity')}"
    )

    return {
        "status":                         "success",
        "dataset_rows":                   len(df),
        "caste_seat_before":              caste_before["d_score"],
        "caste_seat_after":               caste_after["d_score"],
        "caste_interpretation_before":    caste_before["interpretation"],
        "caste_interpretation_after":     caste_after["interpretation"],
        "religion_seat_before":           religion_before["d_score"],
        "religion_seat_after":            religion_after["d_score"],
        "religion_interpretation_before": religion_before["interpretation"],
        "religion_interpretation_after":  religion_after["interpretation"],
        "demographic_parity_before":      parity_before.get("parity"),
        "demographic_parity_after":       parity_after.get("parity"),
        "passes_fairness_before":         parity_before.get("passes_fairness"),
        "passes_fairness":                parity_after.get("passes_fairness"),
        "shortlist_rates_before":         parity_before.get("group_shortlist_rates", {}),
        "shortlist_rates_after":          parity_after.get("group_shortlist_rates", {}),
        "gemini_explanation":             explanation,
        "model":                          "intfloat/multilingual-e5-large",
        "debiasing":                      "subspace_projection_10_directions",
    }


@app.post("/retroactive")
async def retroactive(file: UploadFile = File(...)):
    """
    Retroactive audit: compare biased vs fair decisions per row.
    Returns who would have been shortlisted under the fair model.
    """
    try:
        df = _read_csv(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Cannot read CSV: {e}")

    for col in ["name", "text"]:
        if col not in df.columns:
            raise HTTPException(400, f"Missing column '{col}'")

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    print(f"\n/retroactive — {len(df)} decisions")

    # Encode names
    name_sents = [
        f"query: A person named {str(n)} applied for the job."
        for n in df["name"].tolist()
    ]
    name_embs = get_embeddings(name_sents)

    # Biased scores
    b_cap = np.array([_cosine_sim(e, CAP_PROFILE_BIASED) for e in name_embs])
    b_lim = np.array([_cosine_sim(e, LIM_PROFILE_BIASED) for e in name_embs])
    biased_norm = _normalise(b_cap - b_lim)

    # Debiased scores
    name_fair = hard_debias(name_embs, CASTE_SUBSPACE)
    name_fair = hard_debias(name_fair, RELIGION_SUBSPACE)
    f_cap = np.array([_cosine_sim(e, CAP_PROFILE_DEBIASED) for e in name_fair])
    f_lim = np.array([_cosine_sim(e, LIM_PROFILE_DEBIASED) for e in name_fair])
    fair_norm = _normalise(f_cap - f_lim)

    # Shortlist = top 50%
    sl_before = (biased_norm >= np.percentile(biased_norm, 50)).astype(int)
    sl_after  = (fair_norm   >= np.percentile(fair_norm,   50)).astype(int)

    # Per-decision records
    per_decision = []
    for i, (_, row) in enumerate(df.iterrows()):
        per_decision.append({
            "id":                 int(row.get("id", i)),
            "name":               str(row["name"]),
            "true_group":         str(row.get("true_group", "unknown")),
            "original_score":     round(float(biased_norm[i]), 4),
            "debiased_score":     round(float(fair_norm[i]), 4),
            "shortlisted_before": bool(sl_before[i]),
            "shortlisted_after":  bool(sl_after[i]),
            "outcome_changed":    bool(sl_before[i] != sl_after[i]),
            "score_delta":        round(float(fair_norm[i] - biased_norm[i]), 4),
        })

    changed           = int(np.sum(sl_before != sl_after))
    newly_shortlisted = int(np.sum((sl_before == 0) & (sl_after == 1)))

    print(f"  Done. {changed} outcomes changed, {newly_shortlisted} newly shortlisted.")

    return {
        "status":            "success",
        "total_decisions":   len(df),
        "outcomes_changed":  changed,
        "newly_shortlisted": newly_shortlisted,
        "per_decision":      per_decision,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))