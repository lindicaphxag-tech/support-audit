---
title: "support-audit: A Ledoit-Wolf Breakdown Audit for Few-Shot Support-Set Validation"
tags:
  - Python
  - machine learning
  - statistics
  - model selection
  - reproducibility
authors:
  - name: Zhibo Zhang
    orcid: 0009-0007-3533-1941
    affiliation: 1
affiliations:
  - name: Hangzhou Dianzi University, Hangzhou, China
    index: 1
date: 25 November 2026
bibliography: paper.bib
---

# Summary

Few-shot evaluation often uses one small labelled support set for two jobs:
fitting a classifier and estimating its accuracy by leave-one-out
cross-validation. The estimate is only meaningful while the support geometry
stays below a breakdown boundary. `support-audit` implements a compact,
training-free diagnostic for that boundary. It computes the Ledoit-Wolf
leverage `eta_i`, its mean `nu`, ranks four heads by LOO accuracy, and learns a
regime-specific safe/unsafe decision from labelled calibration audits.
Evaluation audits expose only `X`, `y`, `Q`, and `regime`; query labels are
never read at audit time.

# Statement of need

Practitioners routinely report LOO numbers without checking whether the support
set can validate itself, and a single global `nu` threshold does not transfer
across isotropic, anisotropic, near-duplicate, label-noise,
class-imbalanced, query-shift, and mixed regimes. Existing tools cover adjacent
pieces: scikit-learn [@pedregosa2011scikit] provides Ledoit-Wolf covariance
estimation [@ledoit2004well] and cross-validation, but not the leverage
decomposition or the calibrated decision. `support-audit` turns the diagnostic
into a small, testable command-line and Python interface with a fixed CSV
contract.

The package is deliberately narrow. It does not claim distribution-free
coverage, it does not replace a held-out test set, and it does not promise the
same behaviour for fine-tuned encoders.

# Implementation

The core has four operations: `diagnostics`, `loo_accuracy`,
`fit_boundary_models`, and `run_audit`. It is pure Python on NumPy and
scikit-learn, runs on CPU, and writes
`audit_id,nu,max_eta,loo_best_model,safe`. The test suite covers the closed-form
diagnostic against a direct computation, candidate-head shapes, LOO bounds, the
calibration round trip, and the CLI.

# Validation

On 64 synthetic hidden audits (8 regimes, balanced 32 safe / 32 unsafe), the
reference implementation reaches 0.9375 overall safe accuracy, 0.75 in the
worst regime, 64/64 exact LOO winners, and numerical error below `1e-6`.
These are local oracle results, not external clinical validation.

# Acknowledgements

The author thanks the Terminal-Bench-Science maintainers for task-format
documentation and the anonymous reviewers of the accompanying statistical note.

# References
