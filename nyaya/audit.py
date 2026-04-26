"""
nyaya/audit.py

Retroactive audit system.

For any historical CSV, reruns every decision using both
the biased and fair scoring systems and returns the delta.

This answers: "What would have happened if the fair 
model had been used from the start?"
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine
from nyaya.seat import get_embeddings
from nyaya.debias import hard_debias_subspace


def cosine_sim(u, v):
    return 1.0 - scipy_cosine(u, v)


def normalise(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx == mn:
        return np.ones_like(scores) * 0.5
    return (scores - mn) / (mx - mn)


def run_retroactive_audit(
    df: pd.DataFrame,
    cap_profile_biased: np.ndarray,
    lim_profile_biased: np.ndarray,
    cap_profile_debiased: np.ndarray,
    lim_profile_debiased: np.ndarray,
    caste_subspace: np.ndarray,
    religion_subspace: np.ndarray
) -> dict:
    """
    For each row in df, compute biased score and fair score.
    Return per-decision delta and group summary.
    
    df must have: name column
    df optional:  id, true_group columns
    """
    print(f"  Retroactive audit: {len(df)} decisions...")
    
    # Add id if missing
    if 'id' not in df.columns:
        df = df.copy()
        df['id'] = range(1, len(df) + 1)
    
    # Encode name embeddings — same template as SEAT
    name_sents = [
        "query: A person named " + str(n) + " applied for the job."
        for n in df['name'].tolist()
    ]
    name_embs = get_embeddings(name_sents)
    
    # Biased scores
    cap_b = np.array([cosine_sim(e, cap_profile_biased)
                      for e in name_embs])
    lim_b = np.array([cosine_sim(e, lim_profile_biased)
                      for e in name_embs])
    biased_net  = cap_b - lim_b
    biased_norm = normalise(biased_net)
    
    # Debiased scores
    name_embs_d = hard_debias_subspace(name_embs, caste_subspace)
    name_embs_d = hard_debias_subspace(name_embs_d, religion_subspace)
    
    cap_f = np.array([cosine_sim(e, cap_profile_debiased)
                      for e in name_embs_d])
    lim_f = np.array([cosine_sim(e, lim_profile_debiased)
                      for e in name_embs_d])
    fair_net  = cap_f - lim_f
    fair_norm = normalise(fair_net)
    
    # Shortlist = top 50%
    threshold_b = np.percentile(biased_norm, 50)
    threshold_f = np.percentile(fair_norm, 50)
    shortlisted_b = (biased_norm >= threshold_b).astype(int)
    shortlisted_f = (fair_norm   >= threshold_f).astype(int)
    
    # Per-decision records
    per_decision = []
    for i, row in df.iterrows():
        per_decision.append({
            'id':               int(row.get('id', i)),
            'name':             str(row['name']),
            'true_group':       str(row.get('true_group', 'unknown')),
            'original_score':   round(float(biased_norm[i]), 4),
            'debiased_score':   round(float(fair_norm[i]), 4),
            'shortlisted_before': bool(shortlisted_b[i]),
            'shortlisted_after':  bool(shortlisted_f[i]),
            'outcome_changed':    bool(
                shortlisted_b[i] != shortlisted_f[i]),
            'score_delta': round(float(fair_norm[i] - biased_norm[i]), 4)
        })
    
    # Group summary
    group_summary = {}
    if 'true_group' in df.columns:
        for g in df['true_group'].unique():
            mask = (df['true_group'] == g).values
            if mask.sum() == 0:
                continue
            group_summary[str(g)] = {
                'count': int(mask.sum()),
                'avg_original_score': round(
                    float(biased_norm[mask].mean()), 4),
                'avg_debiased_score': round(
                    float(fair_norm[mask].mean()), 4),
                'outcomes_changed': int(np.sum(
                    shortlisted_b[mask] != shortlisted_f[mask]))
            }
    
    changed           = int(np.sum(shortlisted_b != shortlisted_f))
    newly_shortlisted = int(np.sum(
        (shortlisted_b == 0) & (shortlisted_f == 1)))
    
    return {
        'total_decisions':    len(df),
        'outcomes_changed':   changed,
        'newly_shortlisted':  newly_shortlisted,
        'group_summary':      group_summary,
        'per_decision':       per_decision
    }