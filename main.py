"""
main.py — Nyaya FastAPI Server

Three endpoints:
  GET  /health       — confirms server is running
  POST /audit        — full bias audit on uploaded CSV
  POST /retroactive  — retroactive audit on historical decisions

All pkl files and word lists load at startup.
e5-large model loads once on first embedding call.
Stays loaded in memory for all subsequent requests.
"""

import os
import json
import pickle
import tempfile
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nyaya.seat import seat_score, get_embeddings
from nyaya.debias import hard_debias_subspace
from nyaya.audit import run_retroactive_audit
from nyaya.gemini_explain import get_explanation

# ── App ────────────────────────────────────────────────────────
app = FastAPI(title="Nyaya Bias Audit API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Startup — load everything once ────────────────────────────
print("=" * 55)
print("NYAYA API starting up...")
print("=" * 55)

print("Loading word lists...")
with open('data/word_lists.json') as f:
    WORDS = json.load(f)

print("Loading scoring config...")
CFG = pickle.load(open('data/scoring_config.pkl', 'rb'))

CAP_BIASED   = CFG['cap_profile_biased']
LIM_BIASED   = CFG['lim_profile_biased']
CAP_DEBIASED = CFG['cap_profile_debiased']
LIM_DEBIASED = CFG['lim_profile_debiased']
CASTE_SUB    = CFG['caste_subspace']
RELIGION_SUB = CFG['religion_subspace']

print(f"Caste subspace shape:    {CASTE_SUB.shape}")
print(f"Religion subspace shape: {RELIGION_SUB.shape}")
print("All config loaded. API ready.\n")


# ── Helpers ────────────────────────────────────────────────────
def cos_sim(u, v):
    return 1.0 - scipy_cosine(u, v)

def normalise(scores):
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.ones_like(scores) * 0.5
    return (scores - mn) / (mx - mn)

def read_csv_upload(contents: bytes) -> pd.DataFrame:
    """Write upload bytes to temp file and read as DataFrame."""
    with tempfile.NamedTemporaryFile(
            delete=False, suffix='.csv', mode='wb') as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    df = pd.read_csv(tmp_path)
    os.unlink(tmp_path)
    return df

def compute_parity(df: pd.DataFrame, score_col: str) -> dict:
    """Compute demographic parity for brahmin vs dalit groups."""
    if 'true_group' not in df.columns:
        return {
            "parity": None,
            "passes_fairness": None,
            "group_shortlist_rates": {}
        }
    
    scores    = df[score_col].values
    threshold = np.percentile(scores, 50)
    shortlisted = (scores >= threshold).astype(int)
    
    rates = {}
    for g in ['brahmin', 'dalit']:
        mask = (df['true_group'] == g).values
        if mask.sum() > 0:
            rates[g] = float(shortlisted[mask].mean())
    
    if len(rates) < 2:
        return {
            "parity": None,
            "passes_fairness": None,
            "group_shortlist_rates": {
                g: round(r * 100, 1) for g, r in rates.items()
            }
        }
    
    parity = min(rates.values()) / max(rates.values())
    return {
        "parity": round(parity, 4),
        "passes_fairness": bool(parity >= 0.80),
        "group_shortlist_rates": {
            g: round(r * 100, 1) for g, r in rates.items()
        }
    }


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "intfloat/multilingual-e5-large",
        "service": "Nyaya Bias Audit API"
    }


@app.post("/audit")
async def audit(file: UploadFile = File(...)):
    """
    Full bias audit on uploaded CSV.
    CSV must have: name, text columns.
    Optional: true_group column (brahmin/dalit/hindu_other/muslim)
    """
    # ── Read CSV ───────────────────────────────────────────────
    try:
        contents = await file.read()
        df = read_csv_upload(contents)
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    
    for col in ['name', 'text']:
        if col not in df.columns:
            raise HTTPException(400, f"CSV must have a '{col}' column")
    
    print(f"\n/audit — {len(df)} rows")
    
    # ── SEAT bias scores ───────────────────────────────────────
    print("  Running SEAT caste bias...")
    caste_b = seat_score(
        WORDS['brahmin_surnames'], WORDS['dalit_obc_surnames'],
        WORDS['capability_words'], WORDS['limitation_words']
    )
    
    print("  Running SEAT religion bias...")
    religion_b = seat_score(
        WORDS['hindu_names'], WORDS['muslim_names'],
        WORDS['capability_words'], WORDS['limitation_words']
    )
    
    # ── Biased similarity scoring ──────────────────────────────
    print("  Computing biased scores...")
    name_sents = [
        "query: A person named " + str(n) + " applied for the job."
        for n in df['name'].tolist()
    ]
    name_embs = get_embeddings(name_sents)
    
    cap_sims_b = np.array([cos_sim(e, CAP_BIASED) for e in name_embs])
    lim_sims_b = np.array([cos_sim(e, LIM_BIASED) for e in name_embs])
    df['biased_score'] = normalise(cap_sims_b - lim_sims_b)
    parity_b = compute_parity(df, 'biased_score')
    
    # ── Debiased scoring ───────────────────────────────────────
    print("  Applying subspace debiasing...")
    name_embs_d = hard_debias_subspace(name_embs, CASTE_SUB)
    name_embs_d = hard_debias_subspace(name_embs_d, RELIGION_SUB)
    
    cap_sims_f = np.array([cos_sim(e, CAP_DEBIASED) for e in name_embs_d])
    lim_sims_f = np.array([cos_sim(e, LIM_DEBIASED) for e in name_embs_d])
    df['fair_score'] = normalise(cap_sims_f - lim_sims_f)
    parity_f = compute_parity(df, 'fair_score')
    
    # ── SEAT after debiasing ───────────────────────────────────
    print("  Running SEAT after debiasing...")
    caste_a = seat_score(
        WORDS['brahmin_surnames'], WORDS['dalit_obc_surnames'],
        WORDS['capability_words'], WORDS['limitation_words']
    )
    religion_a = seat_score(
        WORDS['hindu_names'], WORDS['muslim_names'],
        WORDS['capability_words'], WORDS['limitation_words']
    )
    
    # ── Gemini explanation ─────────────────────────────────────
    print("  Getting Gemini explanation...")
    explanation = get_explanation(
        caste_d_before = caste_b['d_score'],
        caste_d_after  = caste_a['d_score'],
        parity_before  = parity_b.get('parity') or 0.0,
        parity_after   = parity_f.get('parity') or 0.0
    )
    
    print(f"  Done. Parity {parity_b.get('parity')} "
          f"→ {parity_f.get('parity')}")
    
    return {
        "status": "success",
        "dataset_rows": len(df),
        
        # SEAT scores — headline numbers
        "caste_seat_before":           caste_b['d_score'],
        "caste_seat_after":            caste_a['d_score'],
        "caste_interpretation_before": caste_b['interpretation'],
        "caste_interpretation_after":  caste_a['interpretation'],
        "religion_seat_before":        religion_b['d_score'],
        "religion_seat_after":         religion_a['d_score'],
        
        # Demographic parity
        "demographic_parity_before":   parity_b.get('parity'),
        "demographic_parity_after":    parity_f.get('parity'),
        "passes_fairness_before":      parity_b.get('passes_fairness'),
        "passes_fairness":             parity_f.get('passes_fairness'),
        
        # Shortlist rates
        "shortlist_rates_before": parity_b.get('group_shortlist_rates', {}),
        "shortlist_rates_after":  parity_f.get('group_shortlist_rates', {}),
        
        # Gemini explanation
        "gemini_explanation": explanation,
        
        # Metadata
        "model": "intfloat/multilingual-e5-large",
        "debiasing": "subspace_projection_20_directions"
    }


@app.post("/retroactive")
async def retroactive(file: UploadFile = File(...)):
    """
    Retroactive audit on historical decisions.
    CSV must have: name, text columns.
    Optional: id, true_group columns.
    """
    try:
        contents = await file.read()
        df = read_csv_upload(contents)
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    
    if 'name' not in df.columns:
        raise HTTPException(400, "CSV must have a 'name' column")
    if 'text' not in df.columns:
        raise HTTPException(400, "CSV must have a 'text' column")
    
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    
    print(f"\n/retroactive — {len(df)} decisions")
    
    result = run_retroactive_audit(
        df             = df,
        cap_profile_biased   = CAP_BIASED,
        lim_profile_biased   = LIM_BIASED,
        cap_profile_debiased = CAP_DEBIASED,
        lim_profile_debiased = LIM_DEBIASED,
        caste_subspace       = CASTE_SUB,
        religion_subspace    = RELIGION_SUB
    )
    
    print(f"  Done. {result['outcomes_changed']} outcomes changed.")
    return {"status": "success", **result}


# ── Local run ──────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)