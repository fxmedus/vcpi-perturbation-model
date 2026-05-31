"""
Perturbation-response models. Ordered by generalization reliability:
  1. global_mean   — floor
  2. knn_chem      — chemical similarity neighbors (strong fast generalizer)
  3. latent_program — gene-program regression + seed ensemble -> uncertainty
Each takes (X_tr, Y_tr, X_q) explicitly for leak-free CV.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from .featurize import cosine_sim


def global_mean(X_tr, Y_tr, X_q):
    pred = np.tile(Y_tr.mean(axis=0), (len(X_q), 1)).astype(np.float32)
    return pred, None


def knn_chem(X_tr, Y_tr, X_q, k: int = 25):
    S = cosine_sim(X_tr, X_q)
    out = np.zeros((len(X_q), Y_tr.shape[1]), dtype=np.float32)
    for i in range(len(X_q)):
        idx = np.argsort(-S[i])[:k]
        w = np.clip(S[i, idx], 0, None)
        sw = w.sum()
        out[i] = (w / sw) @ Y_tr[idx] if sw > 1e-8 else Y_tr.mean(axis=0)
    return out, None


def latent_program(X_tr, Y_tr, X_q, n_comp: int = 64, seeds: int = 5,
                   alpha: float = 10.0):
    n_comp = int(min(n_comp, Y_tr.shape[1] - 1, max(2, Y_tr.shape[0] - 1)))
    preds = []
    for s in range(seeds):
        dec = TruncatedSVD(n_components=n_comp, random_state=s).fit(Y_tr)
        reg = Ridge(alpha=alpha).fit(X_tr, dec.transform(Y_tr))
        preds.append(dec.inverse_transform(reg.predict(X_q)))
    P = np.stack(preds).astype(np.float32)
    return P.mean(axis=0), P.std(axis=0)


def blend(pred_list, weights):
    w = np.asarray(weights, dtype=np.float32); w = w / w.sum()
    return sum(wi * p for wi, p in zip(w, pred_list)).astype(np.float32)
