# Contributing to support-audit

Thank you for considering a contribution. This package is intentionally small; the goal is a reliable audit contract, not a large model zoo.

## Before opening a pull request

- Open an issue with the smallest fixture that reproduces the problem.
- Do not include query labels (`yq`) in evaluation fixtures used by the audit path. The audit must remain label-leakage resistant.
- Keep the public API stable unless the change is discussed in an issue first.
- Add a test for every behaviour change. The suite is run with `pytest`.

## Local checks

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python examples/make_demo.py --out /tmp/support-audit-demo
python -m support_audit \
  --input /tmp/support-audit-demo/eval \
  --calibration /tmp/support-audit-demo/public/calibration \
  --output /tmp/support-audit-demo/report.csv
```

## Style and scope

- Python 3.9+, NumPy, scikit-learn. No GPU or network dependency in the audit path.
- Keep the output schema `audit_id,nu,max_eta,loo_best_model,safe` stable.
- Do not claim distribution-free coverage or clinical validity. The package is an operational audit, not a theorem.
- Report numerical and fallback behaviour explicitly; silent fallbacks are bugs.

## Review

The maintainer reviews pull requests for reproducibility, label-leakage safety, and test coverage. A contribution that changes the diagnostic should include a direct-formula test and a numerical tolerance.
