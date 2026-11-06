"""Utilitários funcionais de rastreabilidade e governança de artefatos de MLOps."""

import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def get_environment_metadata() -> dict[str, Any]:
    """Coleta informações do hardware e ambiente de execução de forma agnóstica.

    Returns:
        Dicionário contendo especificações de acelerador, memória, sistema e versões.
    """
    metadata: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": sys.version.split()[0],
        "device_type": "cpu",
        "device_name": "CPU",
        "total_memory_mb": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            metadata["device_type"] = "cuda"
            metadata["device_name"] = torch.cuda.get_device_name(0)
            metadata["total_memory_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2
            )
            metadata["cuda_version"] = torch.version.cuda
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            metadata["device_type"] = "mps"
            metadata["device_name"] = "Apple Silicon (MPS)"
    except ImportError:
        pass

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus and metadata["device_type"] == "cpu":
            metadata["device_type"] = "gpu_tf"
            metadata["device_name"] = gpus[0].name
    except ImportError:
        pass

    return metadata


def compute_file_sha256(filepath: str) -> str:
    """Calcula a assinatura criptográfica SHA-256 de um arquivo binário.

    Args:
        filepath: Caminho para o arquivo no disco.

    Returns:
        String hexadecimal correspondente ao hash SHA-256.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_training_metadata(
    run_id: str,
    experiment_name: str,
    hyperparameters: dict[str, Any],
    training_cost: dict[str, Any],
    weights_path: str,
    output_dir: str = "artifacts/models",
) -> str:
    """Registra o manifesto estruturado de execução e custo de treinamento.

    Args:
        run_id: Identificador único e normalizado da execução.
        experiment_name: Nome descritivo da hipótese ou tratamento experimental.
        hyperparameters: Dicionário contendo hiperparâmetros de treinamento.
        training_cost: Métricas de tempo decorrido, épocas e contagem de parâmetros.
        weights_path: Caminho do arquivo de pesos do modelo gerado pelo treino.
        output_dir: Diretório de destino para persistência do manifesto JSON.

    Returns:
        Caminho do arquivo JSON de metadados gerado.
    """
    os.makedirs(output_dir, exist_ok=True)
    weights_sha256 = compute_file_sha256(weights_path) if os.path.exists(weights_path) else None

    metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": get_environment_metadata(),
        "hyperparameters": hyperparameters,
        "training_cost": training_cost,
        "artifact_integrity": {
            "weights_filename": os.path.basename(weights_path),
            "weights_path": weights_path,
            "weights_size_bytes": os.path.getsize(weights_path) if os.path.exists(weights_path) else 0,
            "sha256": weights_sha256,
        },
    }

    metadata_path = os.path.join(output_dir, f"{run_id}_training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata_path


def save_inference_artifact(
    run_id: str,
    sample_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    latencies_ms: list[float],
    output_dir: str = "artifacts/inference",
) -> tuple[str, str]:
    """Gera o manifesto tabular de predições e o resumo estatístico de inferência.

    Args:
        run_id: Identificador único da execução correspondente ao modelo.
        sample_ids: Lista de identificadores das amostras avaliadas.
        y_true: Array unidimensional com índices das classes reais.
        y_pred: Array unidimensional com índices das classes previstas.
        y_probs: Matriz bidimensional com probabilidades Softmax de todas as classes.
        class_names: Lista ordenada dos nomes canônicos das classes.
        latencies_ms: Lista contendo latências individuais de inferência em milissegundos.
        output_dir: Diretório de destino para os artefatos de inferência.

    Returns:
        Tupla contendo (caminho_predictions_csv, caminho_metadata_json).
    """
    os.makedirs(output_dir, exist_ok=True)

    eps = 1e-12
    clipped_probs = np.clip(y_probs, eps, 1.0)
    entropy_values = -np.sum(clipped_probs * np.log2(clipped_probs), axis=1)

    max_confidences = np.max(y_probs, axis=1)
    tau85_flags = max_confidences >= 0.85
    tau90_flags = max_confidences >= 0.90

    df_records: dict[str, Any] = {
        "sample_id": sample_ids,
        "ground_truth_class": [class_names[i] for i in y_true],
        "ground_truth_idx": y_true,
        "predicted_class": [class_names[i] for i in y_pred],
        "predicted_idx": y_pred,
        "top1_confidence": np.round(max_confidences, 4),
        "entropy": np.round(entropy_values, 4),
        "is_correct": (y_true == y_pred),
        "early_exit_tau85": tau85_flags,
        "early_exit_tau90": tau90_flags,
    }

    for idx, c_name in enumerate(class_names):
        df_records[f"prob_{c_name}"] = np.round(y_probs[:, idx], 4)

    predictions_df = pd.DataFrame(df_records)
    predictions_path = os.path.join(output_dir, f"{run_id}_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    latencies_arr = np.array(latencies_ms) if len(latencies_ms) > 0 else np.array([0.0])
    mean_lat = float(np.mean(latencies_arr))
    throughput_fps = round(1000.0 / mean_lat, 1) if mean_lat > 0 else 0.0

    total_samples = len(y_true)
    count_tau85 = int(np.sum(tau85_flags))
    count_tau90 = int(np.sum(tau90_flags))

    inference_metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "total_samples": total_samples,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "latency_metrics_ms": {
            "mean": round(mean_lat, 2),
            "median": round(float(np.median(latencies_arr)), 2),
            "p90": round(float(np.percentile(latencies_arr, 90)), 2),
            "p95": round(float(np.percentile(latencies_arr, 95)), 2),
            "throughput_fps": throughput_fps,
        },
        "early_exit_analysis": {
            "threshold_85": {
                "eligible_samples": count_tau85,
                "eligible_percentage": round((count_tau85 / total_samples) * 100.0, 1) if total_samples > 0 else 0.0,
            },
            "threshold_90": {
                "eligible_samples": count_tau90,
                "eligible_percentage": round((count_tau90 / total_samples) * 100.0, 1) if total_samples > 0 else 0.0,
            },
        },
    }

    metadata_path = os.path.join(output_dir, f"{run_id}_inference_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(inference_metadata, f, indent=2, ensure_ascii=False)

    return predictions_path, metadata_path
