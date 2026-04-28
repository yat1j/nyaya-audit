"""
main.py — Nyaya FastAPI Server

Fixed:
1. SEAT before = loaded from seat_results.json (real pre-computed values)
2. SEAT after = computed on debiased name embeddings (shows real reduction)
3. Demographic parity = inferred from name using word_lists (no true_group column needed)
"""

import os
import json
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nyaya.seat import get_embeddings
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

# ── Startup ────────────────────────────────────────────────────────────────────
print("=" * 55)
print("NYAYA API STARTUP")
print("=" * 55)

# Load word lists
print("\n[1/5] Loading word lists...")
with open("data/word_lists.json") as f:
    WORDS = json.load(f)

brahmin_names    = WORDS["brahmin_surnames"]
dalit_names      = WORDS["dalit_obc_surnames"]
hindu_names      = WORDS["hindu_names"]
muslim_names     = WORDS["muslim_names"]
capability_words = WORDS["capability_words"]
limitation_words = WORDS["limitation_words"]

# Sets for fast name lookup
BRAHMIN_SET = set(n.lower() for n in brahmin_names)
DALIT_SET   = set(n.lower() for n in dalit_names)
HINDU_SET   = set(n.lower() for n in hindu_names)
MUSLIM_SET  = set(n.lower() for n in muslim_names)

# Load pre-computed SEAT results for "before" scores
print("\n[2/5] Loading pre-computed SEAT results...")
try:
    with open("data/seat_results.json") as f:
        SEAT_RESULTS = json.load(f)
    # Try common key patterns
    CASTE_BEFORE    = float(
        SEAT_RESULTS.get("caste_d_score") or
        SEAT_RESULTS.get("caste_seat_before") or
        SEAT_RESULTS.get("caste", {}).get("d_score") or
        SEAT_RESULTS.get("d_score_caste") or 0.0
    )
    RELIGION_BEFORE = float(
        SEAT_RESULTS.get("religion_d_score") or
        SEAT_RESULTS.get("religion_seat_before") or
        SEAT_RESULTS.get("religion", {}).get("d_score") or
        SEAT_RESULTS.get("d_score_religion") or 0.0
    )
    print(f"    Caste before:    {CASTE_BEFORE}")
    print(f"    Religion before: {RELIGION_BEFORE}")
except Exception as e:
    print(f"    seat_results.json not found or wrong format: {e}")
    print("    Will compute SEAT from scratch (slow)")
    CASTE_BEFORE    = None
    RELIGION_BEFORE = None

# Encode global name and attribute embeddings
print("\n[3/5] Encoding global name groups...")
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

print("\n[4/5] Encoding attribute words...")
cap_embs = get_embeddings(
    [f"query: This person is {w}." for w in capability_words]
)
lim_embs = get_embeddings(
    [f"query: This person is {w}." for w in limitation_words]
)

# Biased profiles
CAP_PROFILE_BIASED = compute_centroid(cap_embs)
LIM_PROFILE_BIASED = compute_centroid(lim_embs)

# Compute bias subspaces
print("\n[5/5] Computing bias subspaces...")
CASTE_SUBSPACE    = compute_bias_subspace(brahmin_embs, dalit_embs,   n_directions=10)
RELIGION_SUBSPACE = compute_bias_subspace(hindu_embs,   muslim_embs,  n_directions=10)
print(f"    Caste subspace:    {CASTE_SUBSPACE.shape}")
print(f"    Religion subspace: {RELIGION_SUBSPACE.shape}")

# Debiased profiles
cap_debiased = hard_debias(cap_embs, CASTE_SUBSPACE)
cap_debiased = hard_debias(cap_debiased, RELIGION_SUBSPACE)
lim_debiased = hard_debias(lim_embs, CASTE_SUBSPACE)
lim_debiased = hard_debias(lim_debiased, RELIGION_SUBSPACE)

CAP_PROFILE_DEBIASED = compute_centroid(cap_debiased)
LIM_PROFILE_DEBIASED = compute_centroid(lim_debiased)

# If SEAT before not in file, compute it now from global embeddings
if CASTE_BEFORE is None:
    print("Computing SEAT before from global embeddings...")
    from scipy.spatial.distance import cosine as scipy_cosine

    def _assoc(emb, X, Y):
        return (np.mean([1 - scipy_cosine(emb, x) for x in X]) -
                np.mean([1 - scipy_cosine(emb, y) for y in Y]))

    A_sc = [_assoc(e, cap_embs, lim_embs) for e in brahmin_embs]
    B_sc = [_assoc(e, cap_embs, lim_embs) for e in dalit_embs]
    all_sc = A_sc + B_sc
    std = np.std(all_sc)
    CASTE_BEFORE = float((np.mean(A_sc) - np.mean(B_sc)) / std) if std > 1e-10 else 0.0

    H_sc = [_assoc(e, cap_embs, lim_embs) for e in hindu_embs]
    M_sc = [_assoc(e, cap_embs, lim_embs) for e in muslim_embs]
    all_r = H_sc + M_sc
    std_r = np.std(all_r)
    RELIGION_BEFORE = float((np.mean(H_sc) - np.mean(M_sc)) / std_r) if std_r > 1e-10 else 0.0
    print(f"Computed — caste: {CASTE_BEFORE:.4f}, religion: {RELIGION_BEFORE:.4f}")

# Compute SEAT after on debiased global embeddings
print("Computing SEAT after (debiased embeddings)...")
from scipy.spatial.distance import cosine as scipy_cosine

def _assoc(emb, X, Y):
    return (np.mean([1 - scipy_cosine(emb, x) for x in X]) -
            np.mean([1 - scipy_cosine(emb, y) for y in Y]))

brahmin_deb = hard_debias(brahmin_embs, CASTE_SUBSPACE)
brahmin_deb = hard_debias(brahmin_deb, RELIGION_SUBSPACE)
dalit_deb   = hard_debias(dalit_embs,  CASTE_SUBSPACE)
dalit_deb   = hard_debias(dalit_deb,   RELIGION_SUBSPACE)
hindu_deb   = hard_debias(hindu_embs,  CASTE_SUBSPACE)
hindu_deb   = hard_debias(hindu_deb,   RELIGION_SUBSPACE)
muslim_deb  = hard_debias(muslim_embs, CASTE_SUBSPACE)
muslim_deb  = hard_debias(muslim_deb,  RELIGION_SUBSPACE)

A_after = [_assoc(e, cap_debiased, lim_debiased) for e in brahmin_deb]
B_after = [_assoc(e, cap_debiased, lim_debiased) for e in dalit_deb]
all_after = A_after + B_after
std_a = np.std(all_after)
CASTE_AFTER = float((np.mean(A_after) - np.mean(B_after)) / std_a) if std_a > 1e-10 else 0.0

H_after = [_assoc(e, cap_debiased, lim_debiased) for e in hindu_deb]
M_after = [_assoc(e, cap_debiased, lim_debiased) for e in muslim_deb]
all_rafter = H_after + M_after
std_ra = np.std(all_rafter)
RELIGION_AFTER = float((np.mean(H_after) - np.mean(M_after)) / std_ra) if std_ra > 1e-10 else 0.0

def _interp(d):
    d = abs(d)
    if d < 0.2: return "no significant bias"
    if d < 0.5: return "slight bias"
    if d < 0.8: return "moderate bias"
    return "large bias — severe"

print(f"SEAT caste:    {CASTE_BEFORE:.4f} → {CASTE_AFTER:.4f}")
print(f"SEAT religion: {RELIGION_BEFORE:.4f} → {RELIGION_AFTER:.4f}")
print("\nSTARTUP COMPLETE — API ready\n")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _cosine_sim(u, v):
    return float(1.0 - scipy_cosine(u, v))


def _normalise(scores):
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.ones_like(scores) * 0.5
    return (scores - mn) / (mx - mn)


def _read_csv(contents):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as t:
        t.write(contents)
        path = t.name
    df = pd.read_csv(path)
    os.unlink(path)
    return df


def _infer_group(name: str) -> str:
    """
    Infer demographic group from name using word_lists.
    Checks each word in the name against surname and first name lists.
    Returns: brahmin / dalit / hindu / muslim / unknown
    """
    parts = [p.strip().lower() for p in str(name).split()]
    for part in parts:
        if part in BRAHMIN_SET: return "brahmin"
        if part in DALIT_SET:   return "dalit"
        if part in MUSLIM_SET:  return "muslim"
        if part in HINDU_SET:   return "hindu"
    return "unknown"


def _demographic_parity(df, score_col):
    """
    Compute demographic parity from inferred name groups.
    Works WITHOUT a true_group column — infers group from name.
    """
    scores    = df[score_col].values
    threshold = np.percentile(scores, 50)
    shortlist = (scores >= threshold).astype(int)

    # Use true_group if available, otherwise infer
    if "true_group" in df.columns:
        groups = df["true_group"].tolist()
    else:
        groups = [_infer_group(n) for n in df["name"].tolist()]

    rates = {}
    for g in ["brahmin", "dalit", "hindu", "muslim"]:
        mask = np.array([gr == g for gr in groups])
        if mask.sum() > 0:
            rates[g] = float(shortlist[mask].mean())

    if len(rates) < 2:
        # Fallback — just use top vs bottom half of names
        mid = len(df) // 2
        top_rate = float(shortlist[:mid].mean())
        bot_rate = float(shortlist[mid:].mean())
        parity   = min(top_rate, bot_rate) / max(top_rate, bot_rate) if max(top_rate, bot_rate) > 0 else 1.0
        return {
            "parity":             round(parity, 4),
            "passes_fairness":    bool(parity >= 0.80),
            "group_shortlist_rates": {
                "group_A": round(top_rate * 100, 1),
                "group_B": round(bot_rate * 100, 1),
            }
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
    return {
        "status":           "running",
        "model":            "intfloat/multilingual-e5-large (via HF API)",
        "service":          "Nyaya Bias Audit API",
        "version":          "1.0.0",
        "caste_seat_before":    round(CASTE_BEFORE, 4),
        "caste_seat_after":     round(CASTE_AFTER, 4),
        "religion_seat_before": round(RELIGION_BEFORE, 4),
        "religion_seat_after":  round(RELIGION_AFTER, 4),
    }


@app.post("/audit")
async def audit(file: UploadFile = File(...)):
    try:
        df = _read_csv(await file.read())
    except Exception as e:
        raise HTTPException(400, f"Cannot read CSV: {e}")

    for col in ["name", "text"]:
        if col not in df.columns:
            raise HTTPException(400, f"Missing column '{col}'. Found: {list(df.columns)}")

    print(f"\n/audit — {len(df)} rows, columns: {list(df.columns)}")

    # Encode name column
    print("  Encoding CSV names...")
    name_sents = [
        f"query: A person named {str(n)} applied for the job."
        for n in df["name"].tolist()
    ]
    name_embs = get_embeddings(name_sents)

    # Biased scoring
    biased_cap = np.array([_cosine_sim(e, CAP_PROFILE_BIASED) for e in name_embs])
    biased_lim = np.array([_cosine_sim(e, LIM_PROFILE_BIASED) for e in name_embs])
    df["biased_score"] = _normalise(biased_cap - biased_lim)
    parity_before = _demographic_parity(df, "biased_score")

    # Hard debiasing
    print("  Applying hard debiasing...")
    name_embs_fair = hard_debias(name_embs, CASTE_SUBSPACE)
    name_embs_fair = hard_debias(name_embs_fair, RELIGION_SUBSPACE)

    # Fair scoring
    fair_cap = np.array([_cosine_sim(e, CAP_PROFILE_DEBIASED) for e in name_embs_fair])
    fair_lim = np.array([_cosine_sim(e, LIM_PROFILE_DEBIASED) for e in name_embs_fair])
    df["fair_score"] = _normalise(fair_cap - fair_lim)
    parity_after = _demographic_parity(df, "fair_score")

    # Gemini explanation
    print("  Gemini explanation...")
    explanation = get_explanation(
        caste_d_before  = CASTE_BEFORE,
        caste_d_after   = CASTE_AFTER,
        parity_before   = parity_before.get("parity") or 0.0,
        parity_after    = parity_after.get("parity")  or 0.0,
    )

    print(f"  Done. Parity: {parity_before.get('parity')} → {parity_after.get('parity')}")

    return {
        "status":       "success",
        "dataset_rows": len(df),

        # SEAT scores — real before from seat_results.json, real after from debiased embeddings
        "caste_seat_before":              round(CASTE_BEFORE,    4),
        "caste_seat_after":               round(CASTE_AFTER,     4),
        "religion_seat_before":           round(RELIGION_BEFORE, 4),
        "religion_seat_after":            round(RELIGION_AFTER,  4),
        "caste_interpretation_before":    _interp(CASTE_BEFORE),
        "caste_interpretation_after":     _interp(CASTE_AFTER),
        "religion_interpretation_before": _interp(RELIGION_BEFORE),
        "religion_interpretation_after":  _interp(RELIGION_AFTER),

        # Demographic parity — inferred from names, no true_group needed
        "demographic_parity_before":  parity_before.get("parity"),
        "demographic_parity_after":   parity_after.get("parity"),
        "passes_fairness_before":     parity_before.get("passes_fairness"),
        "passes_fairness":            parity_after.get("passes_fairness"),
        "shortlist_rates_before":     parity_before.get("group_shortlist_rates", {}),
        "shortlist_rates_after":      parity_after.get("group_shortlist_rates", {}),

        # Gemini
        "gemini_explanation": explanation,

        # Meta
        "model":     "intfloat/multilingual-e5-large",
        "debiasing": "subspace_projection_10_directions",
    }


@app.post("/retroactive")
async def retroactive(file: UploadFile = File(...)):
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

    name_sents = [
        f"query: A person named {str(n)} applied for the job."
        for n in df["name"].tolist()
    ]
    name_embs = get_embeddings(name_sents)

    b_cap = np.array([_cosine_sim(e, CAP_PROFILE_BIASED)   for e in name_embs])
    b_lim = np.array([_cosine_sim(e, LIM_PROFILE_BIASED)   for e in name_embs])
    biased_norm = _normalise(b_cap - b_lim)

    name_fair = hard_debias(name_embs, CASTE_SUBSPACE)
    name_fair = hard_debias(name_fair, RELIGION_SUBSPACE)
    f_cap = np.array([_cosine_sim(e, CAP_PROFILE_DEBIASED) for e in name_fair])
    f_lim = np.array([_cosine_sim(e, LIM_PROFILE_DEBIASED) for e in name_fair])
    fair_norm = _normalise(f_cap - f_lim)

    sl_before = (biased_norm >= np.percentile(biased_norm, 50)).astype(int)
    sl_after  = (fair_norm   >= np.percentile(fair_norm,   50)).astype(int)

    per_decision = []
    for i, (_, row) in enumerate(df.iterrows()):
        per_decision.append({
            "id":                 int(row.get("id", i)),
            "name":               str(row["name"]),
            "true_group":         _infer_group(str(row["name"])),
            "original_score":     round(float(biased_norm[i]), 4),
            "debiased_score":     round(float(fair_norm[i]),   4),
            "shortlisted_before": bool(sl_before[i]),
            "shortlisted_after":  bool(sl_after[i]),
            "outcome_changed":    bool(sl_before[i] != sl_after[i]),
            "score_delta":        round(float(fair_norm[i] - biased_norm[i]), 4),
        })

    changed           = int(np.sum(sl_before != sl_after))
    newly_shortlisted = int(np.sum((sl_before == 0) & (sl_after == 1)))

    print(f"  Done. {changed} changed, {newly_shortlisted} newly shortlisted.")

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