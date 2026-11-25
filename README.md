# support-audit

**When can a support set validate itself?**

`support-audit` implements the Ledoit-Wolf leverage diagnostic from *When Can a Support Set Validate Itself? A Computable Breakdown Parameter* (TMLR submission 14922). It computes `eta_i`, the mean breakdown parameter `nu`, ranks four few-shot heads by exact leave-one-out accuracy, and learns a regime-specific safe/unsafe boundary from labelled calibration audits.

## Install

```bash
pip install support-audit
```

From source:

```bash
pip install -e ".[test]"
pytest
```

## CLI

```bash
support-audit \
  --input path/to/evaluation_audits \
  --calibration path/to/public/calibration \
  --output results/audit_report.csv
```

The calibration directory must contain `labels.csv` and a sibling `audits/` directory with one `.npz` per labelled audit:

```text
public/
├── audits/
│   └── cal_001.npz
└── calibration/
    └── labels.csv
```

`labels.csv` columns: `audit_id,regime,safe`. Evaluation `.npz` files need `X`, `y`, `Q`, and optionally `regime`; `yq` is never read.

Output schema:

```text
audit_id,nu,max_eta,loo_best_model,safe
```

## What is computed

```text
eta_i = (1 - lambda) / (n - 1) * (x_i - g)^T P (x_i - g)
nu    = mean_i eta_i
```

`P` is the inverse of the Ledoit-Wolf shrunk covariance. `nu` is an effective-dimension-to-shot ratio and a diagnostic, not a theorem. The boundary is regime-dependent; the package learns a shallow classifier per regime instead of applying one global threshold.

## Evidence

On 64 hidden audits (8 regimes, balanced 32 safe / 32 unsafe), the reference implementation reaches 0.9375 overall safe accuracy, 0.75 in the worst regime, 64/64 exact LOO winners, and numerical error below `1e-6` for `nu` and `max_eta`. These are local synthetic results, not external clinical validation.

## Scope

- Targets frozen, training-free linear readouts.
- Does not claim distribution-free coverage or replace a held-out test set.
- Boundary learned from a finite calibration set; it is an operational classifier, not a guarantee.

## Citation

```bibtex
@software{zhang_support_audit_2026,
  author = {Zhang, Zhibo},
  title = {support-audit: Ledoit-Wolf breakdown audit for few-shot support-set validation},
  year = {2026},
  url = {https://github.com/lindicaphxag-tech/support-audit}
}
```

## License

MIT.
