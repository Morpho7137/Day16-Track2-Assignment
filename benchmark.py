#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATASET_NAME = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"
RANDOM_STATE = 42


def ensure_dataset(workdir: Path) -> Path:
    csv_path = workdir / CSV_NAME
    if csv_path.exists():
        return csv_path

    workdir.mkdir(parents=True, exist_ok=True)
    kaggle_bin = shutil.which("kaggle") or "/usr/local/bin/kaggle"
    cmd = [kaggle_bin, "datasets", "download", "-d", DATASET_NAME, "--unzip", "-p", str(workdir)]
    subprocess.run(cmd, check=True)

    if csv_path.exists():
        return csv_path

    matches = list(workdir.rglob(CSV_NAME))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find {CSV_NAME} after downloading {DATASET_NAME}")


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    frame = pd.read_csv(csv_path)
    return frame, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and benchmark a LightGBM model on the Credit Card Fraud dataset.")
    parser.add_argument("--workdir", default=".", help="Directory where the dataset lives and results are written.")
    parser.add_argument("--output", default="benchmark_result.json", help="Output JSON filename.")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    csv_path = ensure_dataset(workdir)
    frame, load_time = load_dataset(csv_path)

    if "Class" not in frame.columns:
        raise ValueError("Expected a Class column in the dataset")

    X = frame.drop(columns=["Class"])
    y = frame["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    start = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    train_time = time.perf_counter() - start

    best_iteration = int(getattr(model, "best_iteration_", None) or model.n_estimators)
    probabilities = model.predict_proba(X_test, num_iteration=best_iteration)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
    }

    sample = X_test.iloc[[0]]
    for _ in range(10):
        model.predict_proba(sample, num_iteration=best_iteration)

    latency_runs = 200
    start = time.perf_counter()
    for _ in range(latency_runs):
        model.predict_proba(sample, num_iteration=best_iteration)
    single_row_latency_ms = ((time.perf_counter() - start) / latency_runs) * 1000.0

    throughput_sample = X_test.iloc[: min(1000, len(X_test))]
    start = time.perf_counter()
    model.predict_proba(throughput_sample, num_iteration=best_iteration)
    throughput_rows_per_sec = len(throughput_sample) / max(time.perf_counter() - start, 1e-9)

    result = {
        "dataset": DATASET_NAME,
        "csv_path": str(csv_path),
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "load_time_seconds": round(load_time, 6),
        "train_time_seconds": round(train_time, 6),
        "best_iteration": best_iteration,
        "metrics": metrics,
        "inference": {
            "single_row_latency_ms": round(single_row_latency_ms, 6),
            "throughput_rows_per_sec": round(throughput_rows_per_sec, 6),
            "throughput_batch_size": int(len(throughput_sample)),
        },
    }

    output_path = workdir / args.output
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Benchmark complete")
    print(f"Dataset: {csv_path}")
    print(f"Load time: {result['load_time_seconds']:.6f}s")
    print(f"Training time: {result['train_time_seconds']:.6f}s")
    print(f"Best iteration: {best_iteration}")
    print(f"AUC-ROC: {metrics['auc_roc']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"F1-score: {metrics['f1_score']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"Single-row latency: {result['inference']['single_row_latency_ms']:.6f} ms")
    print(f"Throughput: {result['inference']['throughput_rows_per_sec']:.6f} rows/sec")
    print(f"Results written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
