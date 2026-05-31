<p align="center">
  <strong>VCPI Perturbation Response Model</strong><br>
  <em>Chemical Similarity Neighbor Regression for Drug-seq Transcriptomics</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://orcid.org/0009-0001-9929-3135"><img src="https://img.shields.io/badge/ORCID-0009--0001--9929--3135-a6ce39?style=flat-square&logo=orcid&logoColor=white" alt="ORCID"></a>
  <img src="https://img.shields.io/badge/Event-AIxBio%20Builder%20Hackathon-2563eb?style=flat-square" alt="Event">
  <img src="https://img.shields.io/badge/Data-VCPI%20Drug--seq%20(CC%20BY%204.0)-orange?style=flat-square" alt="Data">
  <a href="GOVERNANCE.md"><img src="https://img.shields.io/badge/Governance-AIDD--GOV%20Aligned-154360?style=flat-square" alt="Governance"></a>
  <a href="https://github.com/fxmedus/vcpi-perturbation-model/releases/tag/v1.0.0"><img src="https://img.shields.io/badge/Submission-v1.0.0-success?style=flat-square" alt="Submission"></a>
</p>

---

## Why

Predicting how cells respond to unseen compounds is a core challenge in computational pharmacology. The real test is generalization: can a model predict the transcriptomic response to a compound it has never seen?

This repository implements a transparent approach to that problem using chemical structure similarity neighbor regression. Every prediction is traceable to specific training compounds with known similarity scores.

**Author:** Julian Yin Vieira Borges, MD, MS — Department of Computer Science, Boston University
**Contact:** jyborges@bu.edu · **ORCID:** [0009-0001-9929-3135](https://orcid.org/0009-0001-9929-3135)

---

## Results

**Submission:** [predictions.parquet (v1.0.0 Release)](https://github.com/fxmedus/vcpi-perturbation-model/releases/tag/v1.0.0)

| Metric | Value |
|--------|-------|
| Test compounds | 1,064 |
| Scored genes | 12,995 |
| Submission rows | 13,826,680 |
| Training compounds | 2,282 (tvc-bhr-009, 10 µM + DMSO controls, THP-1, 24h) |
| Training expression range | [0.000, 14.997] log2(CPM+1) |
| Prediction range | non-negative, clipped to ≥ 0 |
| Runtime | 66 seconds (MacBook M-series) |

**Cross-validation (independent Codex run, uniform gene weights, 5-fold compound split):** wMSE = 0.204 for mean baseline, 0.208 for KNN, 0.210 for latent program, 0.205 for blend. Note: the contest leaderboard uses per-compound per-gene Mejia weights which produce different absolute wMSE values. The contest's reported per-gene-mean baseline of 0.507 was computed on a specific 200-compound split with official weights and is not directly comparable to these CV numbers.

---

## Approach

For each test compound, Morgan fingerprint (ECFP4, 2048-bit) cosine similarity identifies the 25 most structurally similar training compounds and computes a similarity-weighted average of their observed expression profiles. The model is a lazy learner: it stores the training set and interpolates at query time. Its strength is traceability, not complexity. Every prediction can be decomposed into the specific training compounds that produced it and their similarity weights.

Expression was computed using the official contest normalization (`counts_to_expression` from `vcpi-prediction-contest`): per-sample CPM, log2(CPM+1), then per-compound mean across replicates. Gene set was filtered to the 12,995 scored genes before expression computation for memory efficiency.

The repository also contains a latent gene program model (SVD decomposition + Ridge regression with seed ensemble uncertainty) in `src/models.py`. This model was developed and tested on synthetic data but was not included in the final submission due to compute time constraints during the hackathon.

---

## Limitations

This model has clear boundaries that should be stated:

1. **No extrapolation.** KNN predicts well when test compounds have structurally similar neighbors in the training set. For compounds in entirely novel chemical space with no close neighbors, the model defaults toward the training set average and the prediction carries no compound-specific signal.

2. **Single dose, single cell line.** Training data is from one concentration (10 µM) in one cell line (THP-1 monocytes, 24h). Predictions do not transfer to other doses, timepoints, or cell types without retraining.

3. **No learned representation.** The model does not learn a mapping from chemical structure to gene programs. It memorizes and interpolates. A foundation model or graph neural network could potentially generalize further, at the cost of interpretability.

4. **One of three available releases.** Only tvc-bhr-009 (2,282 compounds) was used for training. Including tvc-kdl-010 (1,498) and tvc-qnu-012 (10,261) would increase training coverage but required more memory than was available during the hackathon.

---

## Data Pipeline Lessons

Three silent data bugs were identified and resolved during the hackathon. These produced valid output with no error messages, which made them difficult to catch without inspecting metadata and verifying counts.

| Bug | Symptom | Root cause | Fix |
|-----|---------|------------|-----|
| Zero training compounds | Filter returns only controls | Concentrations stored in nM, not µM. 10 µM = 10000.0, not 10.0 | Filter on `compound_concentration == 10000.0` |
| 5 compounds instead of 2,282 | Expression computes but groups incorrectly | Metadata has `compound` (internal label) and `user_compound_id` (LIMS ID). Must drop `compound` and rename `user_compound_id` | Drop then rename before `counts_to_expression` |
| Out of memory crash | OS kills process during merge | 800M+ expression rows in long format across 3 releases | Filter to scored genes before computing expression (84% row reduction) |

---

## Evaluation

Predictions are scored by **weighted mean squared error (wMSE)**, a per-gene-weighted metric that emphasizes differentially expressed genes and penalizes mode collapse. Cross-validation is performed over **held-out compounds**, not random samples, which is the correct evaluation for the compound generalization task.

Reference: Seal et al., "Diversity by Design: Addressing Mode Collapse Improves scRNA-seq Perturbation Modeling." *ICML GenBio Workshop*, 2025.

---

## Governance

This project applies the [AIDD-GOV](https://github.com/fxmedus/aidd-gov) open standard principles to a hackathon context. Full documentation in [GOVERNANCE.md](GOVERNANCE.md).

| Principle | Implementation |
|:---:|----------------|
| **Data Provenance** | VCPI Drug-seq (Ginkgo Datapoints, CC BY 4.0). No proprietary data committed |
| **Evaluation Integrity** | Held-out-compound CV only. No random splits |
| **Model Transparency** | KNN traceable to training compounds with similarity scores |
| **No Data Leakage** | Models receive (X_train, Y_train, X_query) as explicit arguments |
| **Reproducibility** | Fixed seeds, versioned deps, smoke test on synthetic data, independent cross-check via Codex |

---

## Repository Structure

```
vcpi-perturbation-model/
├── oneshot.py          # Complete pipeline: download, normalize, featurize, predict, submit
├── smoke_test.py       # Synthetic data verification (no VCPI access required)
├── GOVERNANCE.md       # Computational governance documentation
├── LICENSE             # MIT
└── src/
    ├── __init__.py
    ├── featurize.py    # Morgan fingerprint generation + chemical similarity
    ├── models.py       # Mean baseline, KNN neighbors, latent gene programs
    └── scoring.py      # wMSE metric + held-out-compound cross-validation
```

---

## Reusable Components

| Module | Purpose | Reuse context |
|--------|---------|---------------|
| `src/featurize.py` | Morgan FP generation, cosine similarity, coverage reporting | QSAR, virtual screening, compound clustering |
| `src/models.py` | KNN neighbor regression + latent gene program decomposition with uncertainty | Gene regulatory network analysis, perturbation biology |

---

## Data Access

Training data is provided by **Ginkgo Datapoints** through the [Virtual Cell Pharmacology Initiative](https://thevirtualcell.com) (VCPI). The VCPI Drug-seq dataset contains bulk RNA-seq profiles of THP-1 monocytes treated with diverse compounds using high-throughput DRUG-seq.

- Data license: CC BY 4.0
- No proprietary data stored in this repository
- Contest scoring: [virtualcell-vcpi/vcpi-prediction-contest-2026](https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026)

---

## Requirements

- Python 3.12+
- RDKit 2024.03+
- scikit-learn, pandas, numpy, pyarrow, polars
- [vcpi-client](https://github.com/virtualcell-vcpi/vcpi-client) (for data download)
- [vcpi-prediction-contest](https://github.com/virtualcell-vcpi/vcpi-prediction-contest-2026) (for official normalization and scoring)

---

## Acknowledgments

- **Ginkgo Datapoints** — George Pilitsis (Director, Product) and the VCPI team — for making the VCPI Drug-seq dataset openly available under CC BY 4.0
- **Absentia Labs** (Farhan Khodaee, Rohola Zandie, Robert Betancort) for organizing the AIxBio Builder Hackathon
- Hackathon mentors and judges

---

## Citation

```bibtex
@software{borges2026vcpi,
  author    = {Borges, Julian Yin Vieira},
  title     = {VCPI Perturbation Response Model},
  year      = {2026},
  url       = {https://github.com/fxmedus/vcpi-perturbation-model},
  note      = {AIxBio Builder Hackathon, Boston Seaport, 30 May 2026}
}
```
