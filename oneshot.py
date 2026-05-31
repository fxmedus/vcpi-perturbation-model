#!/usr/bin/env python3
"""ONE-SHOT: single release, filter genes early, predict, submit.
Memory-safe: filters to 12,995 scored genes before pivoting."""
import os, sys, time
t0 = time.time()

# Token
tf = os.path.join(os.path.dirname(__file__), ".tvc_token")
if os.path.exists(tf):
    os.environ["TVC_TOKEN"] = open(tf).read().strip()
if not os.environ.get("TVC_TOKEN"):
    print("ERROR: no token"); sys.exit(1)

import numpy as np, pandas as pd, polars as pl
import vcpi
from vcpi_prediction_contest.expression import counts_to_expression
sys.path.insert(0, os.path.dirname(__file__))
from src.featurize import featurize_ids, coverage_report
from src.models import knn_chem, latent_program, blend

# --- Scored genes (load FIRST, filter everything else to this) ---
genes = pd.read_csv("data/gene_filter.csv")["gene_id"].tolist()
gene_set = set(genes)
print(f"Scored genes: {len(genes)}")

# --- Load ONE release ---
print("\n=== Loading tvc-bhr-009 (2K compounds, memory safe) ==="  )
exp = vcpi.load_experiment("tvc-bhr-009")
meta = exp["metadata"]
counts = exp["data"]

# Filter to 10 µM + controls
meta_f = meta.filter(
    (pl.col("compound_concentration") == 10000.0) | pl.col("is_control")
)
meta_pd = meta_f.to_pandas()
meta_pd = meta_pd.drop(columns=["compound"], errors="ignore")
meta_pd = meta_pd.rename(columns={"user_compound_id": "compound"})
n_compounds = meta_pd["compound"].nunique()
print(f"  samples: {len(meta_pd)}, compounds: {n_compounds}")

# --- Compute expression (filter counts to scored genes FIRST to save memory) ---
print("Filtering counts to scored genes...")
if "gene_id" in counts.columns:
    counts = counts.filter(pl.col("gene_id").is_in(gene_set))
    print(f"  counts after gene filter: {counts.shape}")
print("Computing expression...")
expr = counts_to_expression(counts, meta_pd)
expr["compound"] = expr["compound"].astype(str)
print(f"  expression rows: {len(expr)}")

# --- Pivot to wide ---
print("Pivoting...")
Y_wide = expr.pivot_table(index="compound", columns="gene_id",
                           values="expression", aggfunc="mean")
Y_wide = Y_wide.reindex(columns=genes)
# Fill any missing genes with column mean
col_means = Y_wide.mean()
Y_wide = Y_wide.fillna(col_means)
train_ids = Y_wide.index.astype(str).to_numpy()
Ymat = Y_wide.to_numpy(dtype=np.float32)
print(f"  training matrix: {len(train_ids)} x {Ymat.shape[1]}")
print(f"  Y range: [{Ymat.min():.3f}, {Ymat.max():.3f}]")

# Free memory
del expr, counts, meta, meta_f, meta_pd, exp
import gc; gc.collect()

# --- Test compounds + SMILES ---
test_df = pd.read_csv("data/test_compounds.csv")
test_ids = test_df["compound"].astype(str).to_numpy()
smiles_map = {}
if "smiles" in test_df.columns:
    for _, r in test_df.iterrows():
        smiles_map[str(r["compound"])] = r["smiles"]
if os.path.exists("data/chemistry.parquet"):
    chem = pd.read_parquet("data/chemistry.parquet")
    for c in chem.columns:
        if "smi" in c.lower():
            id_c = [x for x in chem.columns if "user_compound" in x.lower()]
            if id_c:
                for _, r in chem.iterrows():
                    smiles_map[str(r[id_c[0]])] = r[c]
            break
print(f"\nTest: {len(test_ids)} compounds")
print(f"SMILES: {len(smiles_map)} total")
print(f"Train coverage: {coverage_report(train_ids, smiles_map)}")
print(f"Test coverage:  {coverage_report(test_ids, smiles_map)}")

# --- Featurize ---
print("\nFeaturizing...")
Xtr = featurize_ids(train_ids, smiles_map)
Xte = featurize_ids(test_ids, smiles_map)

# --- Predict (KNN only for speed) ---
print("KNN prediction...")
P, _ = knn_chem(Xtr, Ymat, Xte, k=25)
P = np.clip(P, 0, None)

# --- Write submission ---
print("\nWriting submission...")
sub = (pd.DataFrame(P, index=test_ids, columns=genes)
         .reset_index()
         .melt(id_vars="index", var_name="gene_id",
               value_name="predicted_expression")
         .rename(columns={"index": "compound"}))
expected = len(test_ids) * len(genes)
assert len(sub) == expected, f"rows {len(sub)} != {expected}"
assert (sub["predicted_expression"] >= 0).all(), "must be non-negative"

os.makedirs("out", exist_ok=True)
sub.to_parquet("out/predictions.parquet", index=False)
sub.to_csv("out/predictions.csv", index=False)

elapsed = time.time() - t0
print(f"\n{'='*50}")
print(f"DONE in {elapsed:.0f}s")
print(f"  out/predictions.parquet  rows={len(sub)}")
print(f"  compounds={len(test_ids)}  genes={len(genes)}")
print(f"  EMAIL TO: datapoints@ginkgobioworks.com")
print(f"{'='*50}")
