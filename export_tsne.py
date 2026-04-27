"""
export_tsne.py — standalone, no nyaya imports needed
Uses sentence-transformers directly, no HF API
"""
import json
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

print("Loading e5-small locally (downloads ~70MB first time)...")
model = SentenceTransformer("intfloat/multilingual-e5-small")
print("Model loaded.")

with open('data/word_lists.json') as f:
    words = json.load(f)

brahmin  = words['brahmin_surnames']
dalit    = words['dalit_obc_surnames']
hindu    = words['hindu_names']
muslim   = words['muslim_names']
cap_w    = words['capability_words']
lim_w    = words['limitation_words']

all_items = (
    [("brahmin",  f"query: A person named {n} applied for the job.", n) for n in brahmin] +
    [("dalit",    f"query: A person named {n} applied for the job.", n) for n in dalit]   +
    [("hindu",    f"query: A person named {n} applied for the job.", n) for n in hindu]   +
    [("muslim",   f"query: A person named {n} applied for the job.", n) for n in muslim]  +
    [("capable",  f"query: This person is {w}.", w) for w in cap_w]                       +
    [("limited",  f"query: This person is {w}.", w) for w in lim_w]
)

labels  = [x[0] for x in all_items]
sents   = [x[1] for x in all_items]
words_l = [x[2] for x in all_items]

print(f"Encoding {len(sents)} sentences locally...")
all_embs = np.array(model.encode(
    sents,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=32,
))
print(f"Shape: {all_embs.shape}")

# Compute bias subspace (caste)
b_embs = all_embs[:len(brahmin)]
d_embs = all_embs[len(brahmin):len(brahmin)+len(dalit)]
h_embs = all_embs[len(brahmin)+len(dalit):len(brahmin)+len(dalit)+len(hindu)]
m_embs = all_embs[len(brahmin)+len(dalit)+len(hindu):len(brahmin)+len(dalit)+len(hindu)+len(muslim)]

n = min(len(b_embs), len(d_embs))
caste_diffs = b_embs[:n] - d_embs[:n]
pca = PCA(n_components=min(10, n))
pca.fit(caste_diffs)
caste_sub = pca.components_

n2 = min(len(h_embs), len(m_embs))
religion_diffs = h_embs[:n2] - m_embs[:n2]
pca2 = PCA(n_components=min(10, n2))
pca2.fit(religion_diffs)
religion_sub = pca2.components_

def hard_debias(embs, subspace):
    debiased = embs.copy()
    for direction in subspace:
        projections = debiased @ direction
        debiased = debiased - np.outer(projections, direction)
    norms = np.linalg.norm(debiased, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return debiased / norms

perp = min(15, len(sents) - 1)
print(f"t-SNE BEFORE (perplexity={perp})...")
tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
            max_iter=1000, learning_rate='auto', init='pca')
before_2d = tsne.fit_transform(all_embs)
print("Before done.")

print("Debiasing...")
deb = hard_debias(all_embs, caste_sub)
deb = hard_debias(deb, religion_sub)

print("t-SNE AFTER...")
after_2d = tsne.fit_transform(deb)
print("After done.")

tsne_data = {
    "before": [
        {"x": round(float(before_2d[i,0]),4),
         "y": round(float(before_2d[i,1]),4),
         "label": labels[i],
         "word": words_l[i]}
        for i in range(len(labels))
    ],
    "after": [
        {"x": round(float(after_2d[i,0]),4),
         "y": round(float(after_2d[i,1]),4),
         "label": labels[i],
         "word": words_l[i]}
        for i in range(len(labels))
    ],
    "colors": {
        "brahmin": "a78bfa",
        "dalit":   "f87171",
        "hindu":   "34d399",
        "muslim":  "60a5fa",
        "capable": "fbbf24",
        "limited": "6b7280"
    }
}

with open('data/tsne_data.json', 'w') as f:
    json.dump(tsne_data, f)

print(f"\nSaved data/tsne_data.json — {len(labels)} points")
print("Send data/tsne_data.json to Member B NOW")