#!/usr/bin/env python3
"""
Smoke test: fabricates VCPI-shaped data, runs the full pipeline.
Proves the harness works BEFORE real data arrives. No hackathon data used.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from src.featurize import featurize_ids
from src.models import global_mean, knn_chem, latent_program, blend
from src.scoring import cv_compounds

rng = np.random.default_rng(0)

# 40 toy SMILES and 200 genes with 8 latent programs
SMI = ["CCO","CCN","CCC","c1ccccc1","CC(=O)O","CCOC(=O)C","c1ccncc1","CCS",
       "CC(C)O","CCCl","CCBr","c1ccc(O)cc1","CC(=O)N","CCC(=O)O","c1ccc(N)cc1",
       "CN1CCCC1","O=C(O)c1ccccc1","CC(C)(C)O","CCCCO","c1ccc(F)cc1",
       "CCCCCC","C1CCCCC1","c1ccc2ccccc2c1","CC(=O)Nc1ccccc1","COc1ccccc1",
       "CCN(CC)CC","OCC(O)CO","CC(N)C(=O)O","c1cnc2[nH]ccc2c1","CCOCC",
       "CCCCN","CC=CC","C=CC=C","c1ccsc1","c1ccoc1",
       "CC#N","CCC#N","CC(Cl)Cl","FC(F)F","ClCCl"]
ids = [f"CMP_{i:03d}" for i in range(len(SMI))]
smiles = dict(zip(ids, SMI))

X = featurize_ids(ids, smiles, n_bits=512)
W = rng.normal(size=(X.shape[1], 8))
G = rng.normal(size=(8, 200))
Y = ((X @ W @ G) / 50.0 + rng.normal(scale=0.05, size=(len(ids), 200))).astype(np.float32)

print(f"Synthetic Y range: [{Y.min():.3f}, {Y.max():.3f}] (negatives = contrast-like)")
print()
for name, fn in [("mean", global_mean), ("knn", knn_chem), ("latent", latent_program)]:
    m, folds = cv_compounds(fn, X, Y, weights=None, n_splits=5)
    print(f"  {name:7s} CV wMSE {m:.4f}")

# Test submission output
p_knn, _ = knn_chem(X[:30], Y[:30], X[30:])
p_lat, sd = latent_program(X[:30], Y[:30], X[30:])
P = blend([p_knn, p_lat], [0.5, 0.5])
sub = (pd.DataFrame(P, index=ids[30:], columns=[f"ENSG{i:09d}" for i in range(200)])
         .reset_index()
         .melt(id_vars="index", var_name="gene_id", value_name="predicted_log2(CPM+1)")
         .rename(columns={"index": "compound"}))
print(f"\nSubmission shape: {sub.shape} (expected {10*200})")
print(f"Columns: {list(sub.columns)}")
print(sub.head())
print("\n=== SMOKE TEST PASSED ===")
