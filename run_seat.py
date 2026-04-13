"""
run_seat.py

Run the SEAT bias experiment.
This produces the d-scores that are the foundation of the entire demo.

Run with:
    python run_seat.py

Expected runtime: 5-15 minutes on first run (model loads + encodes all sentences).
Subsequent runs: 2-5 minutes (model loads from cache).
"""

import json
import numpy as np
from nyaya.seat import seat_score

print("=" * 60)
print("NYAYA — SEAT Bias Measurement")
print("=" * 60)
print()

# ── Load word lists ────────────────────────────────────────────
print("Loading word lists from data/word_lists.json...")
with open('data/word_lists.json', 'r', encoding='utf-8') as f:
    words = json.load(f)

brahmin   = words['brahmin_surnames']
dalit     = words['dalit_obc_surnames']
hindu     = words['hindu_names']
muslim    = words['muslim_names']
capable   = words['capability_words']
limited   = words['limitation_words']

print(f"Brahmin surnames:     {len(brahmin)}")
print(f"Dalit/OBC surnames:   {len(dalit)}")
print(f"Hindu names:          {len(hindu)}")
print(f"Muslim names:         {len(muslim)}")
print(f"Capability words:     {len(capable)}")
print(f"Limitation words:     {len(limited)}")
print()

# ══════════════════════════════════════════════════════════════
# TEST 1 — CASTE BIAS
# Question: Does LaBSE associate Brahmin surnames MORE with
# capability words than Dalit surnames?
# This is your PRIMARY demo number.
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: CASTE BIAS")
print("Brahmin surnames  vs  Dalit/OBC surnames")
print("Capability words  vs  Limitation words")
print("=" * 60)

caste = seat_score(
    target_A_names   = brahmin,
    target_B_names   = dalit,
    attribute_X_words = capable,
    attribute_Y_words = limited
)

print()
print(f"  d-score:           {caste['d_score']}")
print(f"  Interpretation:    {caste['interpretation']}")
print(f"  Favoured group:    {caste['favoured_group']}")
print(f"  Brahmin avg assoc: {caste['mean_A']:+.4f}")
print(f"  Dalit avg assoc:   {caste['mean_B']:+.4f}")
print(f"  Difference:        {caste['mean_A'] - caste['mean_B']:+.4f}")
print()

# ══════════════════════════════════════════════════════════════
# TEST 2 — RELIGION BIAS
# Question: Does LaBSE associate Hindu names MORE with
# capability words than Muslim names?
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: RELIGION BIAS")
print("Hindu names  vs  Muslim names")
print("Capability words  vs  Limitation words")
print("=" * 60)

religion = seat_score(
    target_A_names    = hindu,
    target_B_names    = muslim,
    attribute_X_words = capable,
    attribute_Y_words = limited
)

print()
print(f"  d-score:          {religion['d_score']}")
print(f"  Interpretation:   {religion['interpretation']}")
print(f"  Favoured group:   {religion['favoured_group']}")
print(f"  Hindu avg assoc:  {religion['mean_A']:+.4f}")
print(f"  Muslim avg assoc: {religion['mean_B']:+.4f}")
print(f"  Difference:       {religion['mean_A'] - religion['mean_B']:+.4f}")
print()

# ══════════════════════════════════════════════════════════════
# TEST 3 — CROSS VALIDATION
# Same test, different sentence template.
# If the bias is real, it should appear with multiple templates.
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: CROSS VALIDATION (alternate template)")
print("Same caste test, different sentence context")
print("=" * 60)

cross = seat_score(
    target_A_names    = brahmin,
    target_B_names    = dalit,
    attribute_X_words = capable,
    attribute_Y_words = limited,
    template_name = "The candidate named {} was reviewed for the position.",
    template_attr = "This applicant is {}."
)

print()
print(f"  d-score:        {cross['d_score']}")
print(f"  Interpretation: {cross['interpretation']}")
print()

# ══════════════════════════════════════════════════════════════
# SUMMARY — THE NUMBERS THAT DRIVE EVERYTHING
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("YOUR DEMO NUMBERS — POST IN TEAM GROUP NOW")
print("=" * 60)
print(f"  Caste bias d-score:    {caste['d_score']}")
print(f"  Religion bias d-score: {religion['d_score']}")
print(f"  Cross-val d-score:     {cross['d_score']}")
print()

# Interpretation guide
caste_d = caste['d_score']
if caste_d >= 0.8:
    print(f"EXCELLENT — d={caste_d} is a large bias signal. Very strong demo.")
elif caste_d >= 0.5:
    print(f"GOOD — d={caste_d} is a significant bias signal. Solid demo.")
elif caste_d >= 0.2:
    print(f"WEAK — d={caste_d} is small. Try switching to multilingual-e5-large.")
    print("  In nyaya/seat.py change:")
    print("  MODEL = SentenceTransformer('LaBSE')")
    print("  to:")
    print("  MODEL = SentenceTransformer('intfloat/multilingual-e5-large')")
else:
    print(f"NO SIGNAL — d={caste_d}. Model must be switched.")
    print("  Try: intfloat/multilingual-e5-large")

print()

# ── Per-name scores (for understanding and debugging) ─────────
print("=" * 60)
print("INDIVIDUAL NAME SCORES — Brahmin surnames")
print("(positive = associated with capability)")
print("=" * 60)
for name, score in sorted(zip(brahmin, caste['A_scores']),
                          key=lambda x: x[1], reverse=True):
    bar = "█" * int(abs(score) * 200)
    sign = "+" if score > 0 else ""
    print(f"  {name:<25} {sign}{score:.4f}  {bar}")

print()
print("=" * 60)
print("INDIVIDUAL NAME SCORES — Dalit/OBC surnames")
print("=" * 60)
for name, score in sorted(zip(dalit, caste['B_scores']),
                          key=lambda x: x[1], reverse=True):
    bar = "█" * int(abs(score) * 200)
    sign = "+" if score > 0 else ""
    print(f"  {name:<25} {sign}{score:.4f}  {bar}")

# ── Save results ───────────────────────────────────────────────
results = {
    "caste_bias":        caste,
    "religion_bias":     religion,
    "cross_validation":  cross
}
with open('data/seat_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print("=" * 60)
print(f"Results saved to data/seat_results.json")
print()
print("NEXT STEPS:")
print(f"  1. Post caste d-score ({caste['d_score']}) in team group")
print(f"  2. Post religion d-score ({religion['d_score']}) in team group")
print("  3. Wait for demo_dataset.csv from Member D")
print("  4. If d-score below 0.2 — switch model (instructions printed above)")
print("=" * 60)