from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.covariance import ledoit_wolf

import support_audit as sa


def make_support(n=40, d=8, classes=4, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(classes, d)) * 2.5
    y = np.repeat(np.arange(classes), n // classes)
    X = centers[y] + rng.normal(size=(n, d)) * scale
    return X, y


def test_diagnostics_matches_direct_formula():
    X, y = make_support(seed=1)
    result = sa.diagnostics(X, y)
    covariance, shrinkage = ledoit_wolf(X, assume_centered=False)
    precision = np.linalg.pinv(covariance)
    centered = X - X.mean(axis=0)
    eta = (1.0 - float(shrinkage)) / (len(X) - 1.0) * np.einsum(
        "ij,jk,ik->i", centered, precision, centered
    )
    eta = np.maximum(eta, 0.0)
    assert result["nu"] == pytest.approx(float(np.mean(eta)), rel=1e-12)
    assert result["max_eta"] == pytest.approx(float(np.max(eta)), rel=1e-12)


def test_centroid_and_ridge():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    y = np.array([0, 0, 1, 1])
    Q = np.array([[0.2, 0.2], [9.8, 9.9]])
    assert sa.fit_predict(X, y, Q, "centroid").tolist() == [0, 1]
    for model in sa.MODELS:
        assert sa.fit_predict(X, y, Q, model).shape == (2,)


def test_loo_bounded():
    X, y = make_support(n=16, d=4, seed=3)
    for model in sa.MODELS:
        assert 0.0 <= sa.loo_accuracy(X, y, model) <= 1.0


def test_run_audit_end_to_end(tmp_path: Path):
    root = tmp_path / "public"
    audits = root / "audits"
    calibration = root / "calibration"
    eval_dir = tmp_path / "eval"
    audits.mkdir(parents=True)
    calibration.mkdir()
    eval_dir.mkdir()

    labels = ["audit_id,regime,safe"]
    for regime in ("iso", "aniso"):
        for i in range(4):
            X, y = make_support(seed=100 + i, scale=0.5 if i < 2 else 2.0)
            np.savez(audits / f"cal_{regime}_{i}.npz", X=X, y=y, Q=X[:8], regime=np.array([regime]))
            labels.append(f"cal_{regime}_{i},{regime},{1 if i < 2 else 0}")
    (calibration / "labels.csv").write_text("\n".join(labels) + "\n")

    X, y = make_support(seed=200)
    np.savez(eval_dir / "eval_001.npz", X=X, y=y, Q=X[:8], regime=np.array(["iso"]))

    output = tmp_path / "out" / "report.csv"
    rows = sa.run_audit(eval_dir, calibration, output)
    assert len(rows) == 1
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(sa.OUTPUT_COLUMNS)
        saved = list(reader)
    assert saved[0]["audit_id"] == "eval_001"
    assert saved[0]["safe"] in {"0", "1"}


def test_cli(tmp_path: Path):
    root = tmp_path / "public"
    audits = root / "audits"
    calibration = root / "calibration"
    eval_dir = tmp_path / "eval"
    audits.mkdir(parents=True)
    calibration.mkdir()
    eval_dir.mkdir()
    labels = ["audit_id,regime,safe"]
    for i in range(6):
        X, y = make_support(seed=300 + i, scale=0.5 if i < 3 else 2.5)
        np.savez(audits / f"cal_{i}.npz", X=X, y=y, Q=X[:5], regime=np.array(["iso"]))
        labels.append(f"cal_{i},iso,{1 if i < 3 else 0}")
    (calibration / "labels.csv").write_text("\n".join(labels) + "\n")
    X, y = make_support(seed=400)
    np.savez(eval_dir / "eval_001.npz", X=X, y=y, Q=X[:5], regime=np.array(["iso"]))
    assert sa.main(["--input", str(eval_dir), "--calibration", str(calibration), "--output", str(tmp_path / "r.csv")]) == 0
