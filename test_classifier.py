"""
test_classifier.py — SUBSPACE DEBIASING VERSION

Uses multi-direction subspace debiasing instead of single direction.
This is the correct approach for high-dimensional models like e5-large
where bias is spread across many dimensions.

The target profile and ranking use EXACTLY the same sentence
templates as SEAT to guarantee consistency with Day 1 d-score.
"""

import json, pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as scipy_cosine
from nyaya.seat import get_embeddings
from nyaya.debias import (hard_debias_subspace,
                           compute_bias_subspaces)

print("=" * 65)
print("NYAYA — Downstream Bias Proof (Subspace Debiasing)")
print("=" * 65)

with open('data/word_lists.json') as f:
    words = json.load(f)

df = pd.read_csv('data/demo_dataset.csv')
print(f"\nDataset: {len(df)} rows")
print(df['true_group'].value_counts().to_string())

# ── STEP 1: Encode capability and limitation words ────────────
# Use EXACT same templates as seat.py
# This ensures target profile is in same embedding space as SEAT

print("\n" + "=" * 65)
print("STEP 1: Building target profile (same templates as SEAT)")
print("=" * 65)

cap_sentences = ["query: This person is " + w + "."
                 for w in words['capability_words']]
lim_sentences = ["query: This person is " + w + "."
                 for w in words['limitation_words']]

print(f"  Encoding {len(cap_sentences)} capability sentences...")
cap_embs = get_embeddings(cap_sentences)

print(f"  Encoding {len(lim_sentences)} limitation sentences...")
lim_embs = get_embeddings(lim_sentences)

def unit(v):
    return v / np.linalg.norm(v)

cap_profile = unit(cap_embs.mean(axis=0))
lim_profile = unit(lim_embs.mean(axis=0))

# ── STEP 2: Encode name embeddings ───────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Encoding name embeddings")
print("=" * 65)

name_sentences = [
    "query: A person named " + n + " applied for the job."
    for n in df['name'].tolist()
]
name_embs = get_embeddings(name_sentences)
print(f"  Name embeddings: {name_embs.shape}")

def cos_sim(u, v):
    return 1.0 - scipy_cosine(u, v)

def score_all(name_embs_in, cap_prof, lim_prof):
    """Compute net score = cap_similarity - lim_similarity for each name."""
    cap_sims = np.array([cos_sim(e, cap_prof) for e in name_embs_in])
    lim_sims = np.array([cos_sim(e, lim_prof) for e in name_embs_in])
    return cap_sims - lim_sims

def parity_and_shortlist(df_in, score_col, group_col='true_group'):
    """Compute shortlist rates and demographic parity."""
    threshold = np.percentile(df_in[score_col], 50)
    shortlisted = (df_in[score_col] >= threshold).astype(int)
    rates = {}
    for g in ['brahmin', 'dalit']:
        mask = df_in[group_col] == g
        rates[g] = shortlisted[mask].mean()
    parity = min(rates.values()) / max(rates.values())
    return shortlisted, rates, parity, threshold

# ── STEP 3: Biased scores ─────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Biased scores (raw e5 embeddings)")
print("=" * 65)

biased_scores = score_all(name_embs, cap_profile, lim_profile)
df['biased_score'] = biased_scores

shortlisted_b, rates_b, parity_b, thresh_b = parity_and_shortlist(
    df, 'biased_score')
df['shortlisted_biased'] = shortlisted_b

print("\n  Group scores BEFORE debiasing:")
print(f"  {'Group':<15} {'Net Score':>12} {'Shortlisted':>12}")
print(f"  {'-'*39}")
for g in ['brahmin', 'dalit', 'hindu_other', 'muslim']:
    mask = df['true_group'] == g
    avg = df[mask]['biased_score'].mean()
    pct = shortlisted_b[mask].mean() * 100
    print(f"  {g:<15} {avg:>12.6f} {pct:>11.1f}%")

print(f"\n  Brahmin shortlist: {rates_b['brahmin']*100:.1f}%")
print(f"  Dalit shortlist:   {rates_b['dalit']*100:.1f}%")
print(f"  Demographic parity: {parity_b:.4f}  "
      f"→ {'FAIL ❌' if parity_b < 0.80 else 'PASS ✓'}")

# ── STEP 4: Compute bias SUBSPACES ───────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Computing bias SUBSPACES (20 directions each)")
print("The bias is spread across many dimensions in e5-large.")
print("Removing 20 directions covers the majority of it.")
print("=" * 65)

caste_subspace, religion_subspace = compute_bias_subspaces(
    words['brahmin_surnames'], words['dalit_obc_surnames'],
    words['hindu_names'],      words['muslim_names'],
    n_components=20
)

print(f"\n  Caste subspace shape:    {caste_subspace.shape}")
print(f"  Religion subspace shape: {religion_subspace.shape}")

# ── STEP 5: Debias EVERYTHING in the same subspace ───────────
print("\n" + "=" * 65)
print("STEP 5: Applying subspace debiasing to all embeddings")
print("=" * 65)

print("  Debiasing name embeddings (caste)...")
name_d = hard_debias_subspace(name_embs, caste_subspace)
print("  Debiasing name embeddings (religion)...")
name_d = hard_debias_subspace(name_d, religion_subspace)

print("  Debiasing capability embeddings (caste)...")
cap_d = hard_debias_subspace(cap_embs, caste_subspace)
print("  Debiasing capability embeddings (religion)...")
cap_d = hard_debias_subspace(cap_d, religion_subspace)

print("  Debiasing limitation embeddings (caste)...")
lim_d = hard_debias_subspace(lim_embs, caste_subspace)
print("  Debiasing limitation embeddings (religion)...")
lim_d = hard_debias_subspace(lim_d, religion_subspace)

cap_profile_d = unit(cap_d.mean(axis=0))
lim_profile_d = unit(lim_d.mean(axis=0))

# ── STEP 6: Fair scores ───────────────────────────────────────
fair_scores = score_all(name_d, cap_profile_d, lim_profile_d)
df['fair_score'] = fair_scores

shortlisted_f, rates_f, parity_f, thresh_f = parity_and_shortlist(
    df, 'fair_score')
df['shortlisted_fair'] = shortlisted_f

print("\n  Group scores AFTER debiasing:")
print(f"  {'Group':<15} {'Net Score':>12} {'Shortlisted':>12}")
print(f"  {'-'*39}")
for g in ['brahmin', 'dalit', 'hindu_other', 'muslim']:
    mask = df['true_group'] == g
    avg = df[mask]['fair_score'].mean()
    pct = shortlisted_f[mask].mean() * 100
    print(f"  {g:<15} {avg:>12.6f} {pct:>11.1f}%")

print(f"\n  Brahmin shortlist: {rates_f['brahmin']*100:.1f}%")
print(f"  Dalit shortlist:   {rates_f['dalit']*100:.1f}%")
print(f"  Demographic parity: {parity_f:.4f}  "
      f"→ {'PASS ✓' if parity_f >= 0.80 else 'FAIL ❌'}")

# ── STEP 7: Retroactive audit numbers ────────────────────────
brahmin_mask = (df['true_group'] == 'brahmin').values
dalit_mask   = (df['true_group'] == 'dalit').values

changed = int(np.sum(shortlisted_b != shortlisted_f))
dalit_newly = int(np.sum(
    (shortlisted_b == 0) & (shortlisted_f == 1) & dalit_mask))
brahmin_removed = int(np.sum(
    (shortlisted_b == 1) & (shortlisted_f == 0) & brahmin_mask))

brahmin_net_b = df[brahmin_mask]['biased_score'].mean()
dalit_net_b   = df[dalit_mask]['biased_score'].mean()
brahmin_net_f = df[brahmin_mask]['fair_score'].mean()
dalit_net_f   = df[dalit_mask]['fair_score'].mean()

# ── STEP 8: Final table ───────────────────────────────────────
print("\n" + "=" * 65)
print("BEFORE / AFTER — FINAL DEMO NUMBERS")
print("=" * 65)
print(f"\n  {'Metric':<35} {'Before':>10} {'After':>10} {'Change':>10}")
print(f"  {'-'*65}")
print(f"  {'Brahmin net SEAT score':<35} "
      f"{brahmin_net_b:>10.6f} {brahmin_net_f:>10.6f} "
      f"{brahmin_net_f-brahmin_net_b:>+10.6f}")
print(f"  {'Dalit net SEAT score':<35} "
      f"{dalit_net_b:>10.6f} {dalit_net_f:>10.6f} "
      f"{dalit_net_f-dalit_net_b:>+10.6f}")
print(f"  {'Score gap (Brahmin - Dalit)':<35} "
      f"{brahmin_net_b-dalit_net_b:>10.6f} "
      f"{brahmin_net_f-dalit_net_f:>10.6f} "
      f"{(brahmin_net_f-dalit_net_f)-(brahmin_net_b-dalit_net_b):>+10.6f}")
print(f"  {'Brahmin shortlist rate':<35} "
      f"{rates_b['brahmin']*100:>9.1f}% "
      f"{rates_f['brahmin']*100:>9.1f}%")
print(f"  {'Dalit shortlist rate':<35} "
      f"{rates_b['dalit']*100:>9.1f}% "
      f"{rates_f['dalit']*100:>9.1f}%")
print(f"  {'Demographic parity':<35} "
      f"{parity_b:>10.4f} {parity_f:>10.4f} "
      f"{parity_f-parity_b:>+10.4f}")
print(f"  {'Passes fairness (>=0.80)':<35} "
      f"{'No':>10} {'Yes' if parity_f>=0.80 else 'No':>10}")
print(f"  {'Decisions changed':<35} {'—':>10} {changed:>10}")
print(f"  {'Dalit newly shortlisted':<35} {'—':>10} {dalit_newly:>10}")

# ── STEP 9: Save ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("SAVING")
print("=" * 65)

pickle.dump({
    'cap_profile_biased':    cap_profile,
    'lim_profile_biased':    lim_profile,
    'cap_profile_debiased':  cap_profile_d,
    'lim_profile_debiased':  lim_profile_d,
    'caste_subspace':        caste_subspace,
    'religion_subspace':     religion_subspace,
    'shortlist_threshold':   0.5
}, open('data/scoring_config.pkl', 'wb'))

records = []
for i, row in df.iterrows():
    records.append({
        'id':               int(row['id']),
        'name':             row['name'],
        'true_group':       row['true_group'],
        'biased_score':     round(float(row['biased_score']), 6),
        'debiased_score':   round(float(row['fair_score']), 6),
        'shortlisted_before': bool(shortlisted_b[i]),
        'shortlisted_after':  bool(shortlisted_f[i]),
        'outcome_changed':    bool(shortlisted_b[i] != shortlisted_f[i]),
        'score_delta': round(float(row['fair_score'] - row['biased_score']), 6)
    })

with open('data/classifier_results.json', 'w') as f:
    json.dump({
        "biased": {
            "brahmin_shortlist": round(rates_b['brahmin']*100, 1),
            "dalit_shortlist":   round(rates_b['dalit']*100, 1),
            "parity":            round(float(parity_b), 4),
            "passes":            bool(parity_b >= 0.80)
        },
        "fair": {
            "brahmin_shortlist": round(rates_f['brahmin']*100, 1),
            "dalit_shortlist":   round(rates_f['dalit']*100, 1),
            "parity":            round(float(parity_f), 4),
            "passes":            bool(parity_f >= 0.80)
        },
        "retroactive": {
            "total":    len(df),
            "changed":  changed,
            "dalit_newly_shortlisted": dalit_newly,
            "brahmin_removed": brahmin_removed
        },
        "per_decision": records
    }, f, indent=2)

print("  ✓ scoring_config.pkl")
print("  ✓ classifier_results.json")

print("\n" + "=" * 65)
print("POST IN TEAM GROUP")
print("=" * 65)
print(f"\n  Parity BEFORE: {parity_b:.4f} → FAIL ❌")
print(f"  Parity AFTER:  {parity_f:.4f} → "
      f"{'PASS ✓' if parity_f>=0.80 else 'FAIL ❌'}")
print(f"  Dalit newly shortlisted: {dalit_newly}")
print(f"  Decisions changed: {changed}")