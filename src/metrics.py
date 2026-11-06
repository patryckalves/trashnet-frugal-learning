"""Módulo de cálculo de métricas de classificação e avaliação de custos operacionais.

Responsável por:
- Callback de rastreamento de tempo por época e memória
- Geração de relatório de classificação padronizado
- Formatação e plotagem de matriz de confusão percentual por linha
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset import CLASS_NAMES


def get_hardware_info() -> dict[str, Any]:
    """Detecta informações de GPU/CPU e memória disponível.

    Returns:
        Dicionário com o dispositivo principal e uso de VRAM de pico em megabytes.
    """
    gpu_name = "CPU"
    peak_vram_mb = 0.0

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                details = tf.config.experimental.get_device_details(gpus[0])
                gpu_name = details.get("device_name", str(gpus[0]))
            except (AttributeError, KeyError, ValueError):
                gpu_name = str(gpus[0])
            try:
                mem_info = tf.config.experimental.get_memory_info("GPU:0")
                peak_vram_mb = mem_info.get("peak", 0) / (1024 * 1024)
            except (AttributeError, ValueError):
                peak_vram_mb = 0.0
    except ImportError:
        pass

    return {
        "device": gpu_name,
        "peak_vram_mb": round(float(peak_vram_mb), 2),
    }


def evaluate_model(
    model: Any,
    test_ds: Any,
    history: Any = None,
    total_time: float = 0.0,
    avg_epoch_time: float = 0.0,
    model_size_mb: float = 0.0,
) -> dict[str, Any]:
    """Avalia um modelo treinado sobre o conjunto de teste e sintetiza métricas.

    Args:
        model: Instância do modelo neural compilado.
        test_ds: Conjunto de dados de teste (tf.data.Dataset ou gerador).
        history: Objeto History retornado pelo método fit.
        total_time: Duração total do treinamento em segundos.
        avg_epoch_time: Duração média por época em segundos.
        model_size_mb: Tamanho serializado do modelo em megabytes.

    Returns:
        Dicionário consolidando métricas globais, por classe e matriz de confusão.
    """
    y_true_list: list[int] = []
    y_pred_probs_list: list[list[float]] = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true_list.extend(batch_y.numpy().tolist() if hasattr(batch_y, "numpy") else list(batch_y))
        y_pred_probs_list.extend(preds.tolist())

    y_true = np.array(y_true_list)
    y_pred_probs = np.array(y_pred_probs_list)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    trainable_params = int(model.count_params()) if hasattr(model, "count_params") else 0
    hw_info = get_hardware_info()
    epochs_trained = len(history.epoch) if history is not None and hasattr(history, "epoch") else 0

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "trainable_parameters": trainable_params,
        "model_size_mb": round(float(model_size_mb), 2),
        "epochs_trained": int(epochs_trained),
        "total_time_seconds": round(float(total_time), 2),
        "avg_epoch_time_seconds": round(float(avg_epoch_time), 2),
        "hardware": hw_info["device"],
        "peak_vram_mb": hw_info["peak_vram_mb"],
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_pred_probs": y_pred_probs.tolist(),
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    output_path: str | None = None,
) -> plt.Figure:
    """Gera a matriz de confusão formatada com contagens e porcentagens por linha.

    Apresenta valores absolutos e percentuais normalizados por linha na mesma célula,
    conforme diretriz estrita da Seção 4 da proposta.

    Args:
        cm: Matriz de confusão em arranjo numérico 6x6.
        title: Título descritivo exibido no cabeçalho do gráfico.
        output_path: Caminho opcional para persistência da figura em disco.

    Returns:
        Instância do objeto plt.Figure contendo o mapa de calor renderizado.
    """
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_perc = np.nan_to_num(cm / row_sums * 100.0)

    annot_labels = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_labels[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sns.heatmap(
        cm_perc,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Classe Prevista", fontsize=11, fontweight="bold")
    ax.set_ylabel("Classe Real", fontsize=11, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")

    return fig
