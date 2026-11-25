"""Ledoit-Wolf breakdown audit for few-shot support-set validation."""
from __future__ import annotations
import argparse, csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import numpy as np
from sklearn.covariance import ledoit_wolf
from sklearn.ensemble import RandomForestClassifier

MODELS = ("ridge_0.1", "ridge_1", "ridge_10", "centroid")
BOUNDARY_FEATURES = ("nu", "max_eta", "class_imbalance")
OUTPUT_COLUMNS = ("audit_id", "nu", "max_eta", "loo_best_model", "safe")

@dataclass(frozen=True)
class Audit:
    audit_id: str
    X: np.ndarray
    y: np.ndarray
    Q: np.ndarray | None = None
    regime: str | None = None

    @classmethod
    def from_npz(cls, path: str | Path) -> "Audit":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            X = np.asarray(data["X"], dtype=float)
            y = np.asarray(data["y"], dtype=int)
            Q = np.asarray(data["Q"], dtype=float) if "Q" in data else None
            regime = str(data["regime"][0]) if "regime" in data else None
        return cls(path.stem, X, y, Q, regime)

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1

def one_hot(y, n_classes=None):
    y = np.asarray(y, dtype=int)
    n_classes = int(y.max()) + 1 if n_classes is None else n_classes
    out = np.zeros((len(y), n_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out

def fit_predict(X, y, Q, model, n_classes=None):
    X, y, Q = np.asarray(X, float), np.asarray(y, int), np.asarray(Q, float)
    n_classes = int(y.max()) + 1 if n_classes is None else n_classes
    if model == "centroid":
        fallback = X.mean(axis=0)
        centers = np.stack([X[y == c].mean(axis=0) if np.any(y == c) else fallback for c in range(n_classes)])
        return ((Q[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
    if not model.startswith("ridge_"):
        raise ValueError(f"unknown model: {model}")
    shrinkage = float(model.split("_", 1)[1])
    weights = np.linalg.solve(X.T @ X + shrinkage * np.eye(X.shape[1]), X.T @ one_hot(y, n_classes))
    return (Q @ weights).argmax(1)

def loo_accuracy(X, y, model, n_classes=None):
    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(y) < 2:
        raise ValueError("LOO needs at least two support points")
    n_classes = int(y.max()) + 1 if n_classes is None else n_classes
    correct = 0
    for i in range(len(y)):
        keep = np.ones(len(y), dtype=bool); keep[i] = False
        pred = int(fit_predict(X[keep], y[keep], X[i:i+1], model, n_classes)[0])
        correct += int(pred == int(y[i]))
    return correct / len(y)

def diagnostics(X, y):
    X, y = np.asarray(X, float), np.asarray(y, int)
    n = len(X)
    if n < 2 or X.ndim != 2 or len(y) != n:
        raise ValueError("need a 2-D X and matching y with at least two rows")
    covariance, shrinkage = ledoit_wolf(X, assume_centered=False)
    precision = np.linalg.pinv(covariance)
    centered = X - X.mean(axis=0)
    eta = (1.0 - float(shrinkage)) / (n - 1.0) * np.einsum("ij,jk,ik->i", centered, precision, centered)
    eta = np.maximum(eta, 0.0)
    counts = np.bincount(y, minlength=int(y.max()) + 1)
    positive = counts[counts > 0]
    return {
        "nu": float(np.mean(eta)),
        "max_eta": float(np.max(eta)),
        "effective_dim": float(np.sum(eta) ** 2 / max(np.sum(eta**2), 1e-12)),
        "class_imbalance": float(positive.max() / max(positive.min(), 1)),
    }

def boundary_vector(features):
    return [float(features[name]) for name in BOUNDARY_FEATURES]

def fit_boundary_models(calibration_dir):
    calibration_dir = Path(calibration_dir)
    audits_dir = calibration_dir.parent / "audits"
    labels_path = calibration_dir / "labels.csv"
    if not audits_dir.is_dir() or not labels_path.is_file():
        raise FileNotFoundError(f"expected {audits_dir} and {labels_path}")
    labels = {}
    with labels_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["audit_id"]] = (row["regime"], int(row["safe"]))
    grouped = {}
    for path in sorted(audits_dir.glob("*.npz")):
        if path.stem not in labels:
            continue
        regime, safe = labels[path.stem]
        audit = Audit.from_npz(path)
        grouped.setdefault(regime, []).append((boundary_vector(diagnostics(audit.X, audit.y)), safe))
    if not grouped:
        raise ValueError("no labelled calibration audits found")
    models = {}
    for regime, samples in grouped.items():
        features = np.asarray([s[0] for s in samples], float)
        targets = np.asarray([s[1] for s in samples], int)
        if len(np.unique(targets)) < 2:
            models[regime] = int(targets[0])
        else:
            models[regime] = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, class_weight="balanced").fit(features, targets)
    all_samples = [s for samples in grouped.values() for s in samples]
    features = np.asarray([s[0] for s in all_samples], float)
    targets = np.asarray([s[1] for s in all_samples], int)
    if len(np.unique(targets)) < 2:
        fallback = int(targets[0])
    else:
        fallback = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0, class_weight="balanced").fit(features, targets)
    return models, fallback

def boundary_safe(features, regime, models, fallback):
    model = models.get(regime, fallback) if regime is not None else fallback
    if isinstance(model, (int, np.integer)):
        return int(model)
    return int(model.predict(np.asarray([boundary_vector(features)], float))[0])

def audit_one(audit, models, fallback):
    features = diagnostics(audit.X, audit.y)
    scores = {m: loo_accuracy(audit.X, audit.y, m, audit.n_classes) for m in MODELS}
    best = max(MODELS, key=lambda m: (scores[m], -MODELS.index(m)))
    return {"audit_id": audit.audit_id, "nu": f"{features['nu']:.8f}", "max_eta": f"{features['max_eta']:.8f}", "loo_best_model": best, "safe": boundary_safe(features, audit.regime, models, fallback)}

def run_audit(input_dir, calibration_dir, output_path):
    models, fallback = fit_boundary_models(calibration_dir)
    rows = [audit_one(Audit.from_npz(path), models, fallback) for path in sorted(Path(input_dir).glob("*.npz"))]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader(); writer.writerows(rows)
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(prog="support-audit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    rows = run_audit(args.input, args.calibration, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
