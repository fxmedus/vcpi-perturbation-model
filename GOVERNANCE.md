# Governance and Reproducibility

This document describes the computational governance principles applied to this project, consistent with the AIDD-GOV open standard for AI-governed drug discovery (github.com/fxmedus/aidd-gov, Apache 2.0).

## Principles

### 1. Data Provenance

All training data originates from the VCPI Drug-seq dataset provided by Ginkgo Datapoints under CC BY 4.0. No proprietary data is stored in this repository. The `data/` directory is gitignored. Researchers wishing to reproduce results must obtain data access through [thevirtualcell.com](https://thevirtualcell.com).

### 2. Evaluation Integrity

Cross-validation is performed over **held-out compounds**, not random samples. This is the only scientifically valid evaluation for the compound generalization task. Random-split CV would leak chemical similarity information and produce inflated performance estimates.

The evaluation metric (wMSE) uses the official per-gene weight vector provided by the hackathon organizers. Local cross-validation results are reported alongside held-out test predictions so that the gap between internal and external evaluation is transparent.

### 3. Uncertainty Disclosure

The latent gene program model provides per-gene uncertainty estimates via seed ensemble variance. Predictions with high uncertainty are flagged, not hidden. This prevents overconfident predictions for compounds in underrepresented chemical space from being treated as reliable.

### 4. Model Transparency

No black-box models are used. Both the chemical similarity neighbor model and the latent gene program regression are fully interpretable:

- The neighbor model's predictions are traceable to specific training compounds with known similarity scores.
- The latent gene programs can be mapped to known biological pathways (e.g., via Gene Ontology or Reactome enrichment), making the model's learned structure biologically auditable.

### 5. Separation of Training and Scoring

The scoring function (`src/scoring.py`) is independent of the modeling code. It accepts any (prediction, truth) matrix pair and applies the official weights. This prevents metric manipulation and ensures that model development does not inadvertently optimize against a different objective than the one being scored.

### 6. No Data Leakage

The pipeline enforces strict separation between training and test data:

- Models receive `(X_train, Y_train, X_query)` as explicit arguments
- No global state carries training information into prediction
- The cross-validation harness splits by compound, not by observation
- Test compound features are featurized identically to training compounds using the same Morgan fingerprint parameters

### 7. Reproducibility

- Random seeds are fixed and documented (SVD: seeds 0 through 4, CV: seed 0)
- All dependencies are standard, versioned, open-source packages
- The smoke test (`smoke_test.py`) verifies the full pipeline on synthetic data without requiring access to the VCPI dataset
- The submission format is validated by structural assertions before writing

## Relationship to AIDD-GOV

This project applies a subset of the AIDD-GOV governance framework to a hackathon context. The full AIDD-GOV standard (10 schemas, 3 conformance levels) is designed for regulated drug discovery pipelines. Here, we apply the core principles of data provenance, evaluation integrity, uncertainty disclosure, and model transparency at a level appropriate for a research prototype.

For the full AIDD-GOV specification: [github.com/fxmedus/aidd-gov](https://github.com/fxmedus/aidd-gov)

## Author

Julian Yin Vieira Borges, MD, MS
Department of Computer Science, Boston University
ORCID: 0009-0001-9929-3135
