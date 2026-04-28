# Nyaya — Backend

This is the FastAPI backend for Nyaya. It does three things:

1. **Detects bias** — Takes a CSV of hiring decisions and runs SEAT (Sentence Encoder Association Test) to measure caste and religion bias in the embedding model. Outputs a d-score for each group.

2. **Removes bias** — Applies subspace projection debiasing across 20 embedding dimensions. Identifies the bias direction mathematically and removes it without affecting other semantic content.

3. **Retroactive audit** — Replays every past decision under the debiased model. Reports which outcomes changed and which candidates would have been shortlisted under a fair model.

4. **Explains in plain English** — Calls Gemini API to generate a 3-sentence natural language summary of the bias findings.

**Live API:** https://yatj-nyaya-api.hf.space

**Endpoints:**
- `POST /audit` — Upload CSV, get bias metrics
- `POST /retroactive` — Get per-decision audit results  
- `GET /health` — Check if API is running

**Built with:** Python, FastAPI, e5-large (sentence-transformers), Gemini API, Firebase Firestore
