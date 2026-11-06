"""Testes unitários para utilitários de MLOps, governança e inferência adaptativa."""

import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

from src.dataset import CLASS_NAMES
from src.mlops import (
    compute_file_sha256,
    get_environment_metadata,
    save_inference_artifact,
    save_training_metadata,
)


def test_compute_file_sha256() -> None:
    """Valida cálculo determinístico da assinatura SHA-256."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("frugal_ai_trashnet_test_payload")
        fpath = f.name

    try:
        calculated_hash = compute_file_sha256(fpath)
        expected_hash = hashlib.sha256(b"frugal_ai_trashnet_test_payload").hexdigest()
        assert calculated_hash == expected_hash
    finally:
        os.remove(fpath)


def test_get_environment_metadata() -> None:
    """Verifica presença e tipos dos metadados de execução."""
    meta = get_environment_metadata()
    required_keys = ["os", "architecture", "python_version", "device_type", "device_name"]
    for k in required_keys:
        assert k in meta
        assert isinstance(meta[k], str)


def test_save_training_metadata() -> None:
    """Verifica geração e integridade do manifesto JSON de treino."""
    temp_dir = tempfile.mkdtemp()
    dummy_weights = os.path.join(temp_dir, "model.pt")
    with open(dummy_weights, "wb") as f:
        f.write(b"mock_binary_weights_data")

    try:
        json_path = save_training_metadata(
            run_id="run_test_01",
            experiment_name="Teste Unitário de Metadados",
            hyperparameters={"lr": 0.001, "batch_size": 32},
            training_cost={"epochs": 10, "time_seconds": 12.5},
            weights_path=dummy_weights,
            output_dir=temp_dir,
        )

        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert data["run_id"] == "run_test_01"
        assert data["hyperparameters"]["batch_size"] == 32
        assert data["artifact_integrity"]["sha256"] == compute_file_sha256(dummy_weights)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_inference_artifact() -> None:
    """Valida manifesto de inferência e cálculo de early-exit."""
    temp_dir = tempfile.mkdtemp()
    try:
        n_samples = 20
        sample_ids = [f"sample_{i}.jpg" for i in range(n_samples)]
        y_true = np.random.randint(0, 6, size=n_samples)
        y_pred = y_true.copy()
        y_pred[0] = (y_pred[0] + 1) % 6  # induz um erro

        y_probs = np.zeros((n_samples, 6))
        for i in range(n_samples):
            y_probs[i, y_pred[i]] = 0.92
            rem = 0.08 / 5.0
            for j in range(6):
                if j != y_pred[i]:
                    y_probs[i, j] = rem

        y_probs[1, :] = 1.0 / 6.0  # baixa confiança

        latencies = [15.0 + float(i) for i in range(n_samples)]

        csv_path, json_path = save_inference_artifact(
            run_id="run_eval_01",
            sample_ids=sample_ids,
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            class_names=CLASS_NAMES,
            latencies_ms=latencies,
            output_dir=temp_dir,
        )

        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        df_preds = pd.read_csv(csv_path)
        assert len(df_preds) == n_samples
        assert "early_exit_tau85" in df_preds.columns
        assert "early_exit_tau90" in df_preds.columns
        assert "entropy" in df_preds.columns
        assert df_preds["early_exit_tau85"].iloc[0] is True or df_preds["early_exit_tau85"].iloc[0] == 1
        assert df_preds["early_exit_tau85"].iloc[1] is False or df_preds["early_exit_tau85"].iloc[1] == 0

        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)

        assert meta["total_samples"] == n_samples
        assert "latency_metrics_ms" in meta
        assert meta["early_exit_analysis"]["threshold_85"]["eligible_samples"] > 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
