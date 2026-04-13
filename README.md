# Nyaya — AI Bias Detection System

## Overview
Nyaya detects bias in embedding models using SEAT (Sentence Encoder Association Test).

## What we built (Day 1)
- Implemented SEAT algorithm
- Measured bias in embeddings
- Generated d-scores for analysis

## Results
- Caste bias d-score: 0.8203 (large)
- Religion bias d-score: 0.3015 (small)

## How it works
1. Convert text to embeddings
2. Compute similarity between groups and attributes
3. Calculate bias score (d-score)

## How to run
```bash
pip install sentence-transformers numpy scipy scikit-learn
python run_seat.py