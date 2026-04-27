"""
export_tsne.py
Run locally: python export_tsne.py
Output: data/tsne_data.json — share this with Member B immediately
"""
import json, pickle
import numpy as np
from sklearn.manifold import TSNE
from nyaya.seat import get_embeddings
from nyaya.debias import compute_bias_subspace, hard_debias

print("Loading word lists...")
with open('data/word_lists.json') as f:
    words = json.load(f)

brahmin  = words['brahmin_surnames']
dalit    = words['dalit_obc_surnames']
hindu    = words['hindu_names']
muslim   = words['muslim_names']
cap_w    = words['capability_words']
lim_w    = words['limitation_words']

# Build all sentences
b_sents  = [f"query: A person named {n} applied for the job." for n in brahmin]
d_sents  = [f"query: A person named {n} applied for the job." for n in dalit]
h_sents  = [f"query: A person named {n} applied for the job." for n in hindu]
m_sents  = [f"query: A person named {n} applied for the job." for n in muslim]
c_sents  = [f"query: This person is {w}." for w in cap_w]
l_sents  = [f"query: This person is {w}." for w in lim_w]

all_sents = b_sents + d_sents + h_sents + m_sents + c_sents + l_sents
labels    = (
    ['brahmin']*len(b_sents) + ['dalit']*len(d_sents) +
    ['hindu']*len(h_sents)   + ['muslim']*len(m_sents) +
    ['capable']*len(c_sents) + ['limited']*len(l_sents)
)
words_list = brahmin + dalit + hindu + muslim + cap_w + lim_w

print(f"Encoding {len(all_sents)} sentences via HF API...")
all_embs = get_embeddings(all_sents)
print(f"Shape: {all_embs.shape}")

# Compute subspaces
b_embs = get_embeddings(b_sents)
d_embs = get_embeddings(d_sents)
h_embs = get_embeddings(h_sents)
m_embs = get_embeddings(m_sents)
caste_sub    = compute_bias_subspace(b_embs, d_embs)
religion_sub = compute_bias_subspace(h_embs, m_embs)

perp = min(15, len(all_sents) - 1)
print(f"t-SNE BEFORE (perplexity={perp})...")
tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
            n_iter=1000, learning_rate='auto', init='pca')
before = tsne.fit_transform(all_embs)
print("Done.")

print("Debiasing + t-SNE AFTER...")
deb = hard_debias(all_embs, caste_sub)
deb = hard_debias(deb, religion_sub)
after = tsne.fit_transform(deb)
print("Done.")

tsne_data = {
    "before": [
        {"x": round(float(before[i,0]),4), "y": round(float(before[i,1]),4),
         "label": labels[i], "word": words_list[i]}
        for i in range(len(labels))
    ],
    "after": [
        {"x": round(float(after[i,0]),4),  "y": round(float(after[i,1]),4),
         "label": labels[i], "word": words_list[i]}
        for i in range(len(labels))
    ],
    "colors": {"brahmin":"a78bfa","dalit":"f87171",
               "hindu":"34d399","muslim":"60a5fa",
               "capable":"fbbf24","limited":"9ca3af"}
}

with open('data/tsne_data.json','w') as f:
    json.dump(tsne_data, f)
print(f"Saved data/tsne_data.json — {len(labels)} points")
print("Share this file with Member B NOW.")