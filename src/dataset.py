"""Módulo de utilitários para manipulação do dataset TrashNet.

Responsável por:
- Mapeamento e validação das classes canônicas do professor
- Varredura de integridade de arquivos de imagem
- Divisão estratificada (70% treino, 15% validação, 15% teste) com seed reproduzível
- Exportação e leitura dos metadados de partição
- Cálculo de pesos de classe para balanceamento da perda
"""

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES: list[str] = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
CLASS_TO_IDX: dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}


def scan_dataset(data_dir: str) -> pd.DataFrame:
    """Varre o diretório do dataset e valida a integridade de cada imagem.

    Args:
        data_dir: Caminho base onde se localizam as pastas das classes.

    Returns:
        DataFrame indexando metadados técnicos de cada imagem encontrada.

    Raises:
        FileNotFoundError: Se alguma pasta de classe canônica não existir.
    """
    records: list[dict[str, Any]] = []

    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            raise FileNotFoundError(f"Diretório da classe não encontrado: {cls_dir}")

        filenames = sorted(os.listdir(cls_dir))
        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            fpath = os.path.join(cls_dir, fname)
            try:
                with Image.open(fpath) as img:
                    width, height = img.size
                    mode = img.mode
                    is_valid = True
            except (OSError, ValueError):
                width, height = None, None
                mode = None
                is_valid = False

            records.append({
                "filename": fname,
                "class_name": cls_name,
                "class_idx": CLASS_TO_IDX[cls_name],
                "filepath": fpath,
                "rel_path": os.path.relpath(fpath, data_dir),
                "width": width,
                "height": height,
                "mode": mode,
                "is_valid": is_valid,
            })

    return pd.DataFrame(records)


def create_stratified_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Realiza a partição estratificada do dataset em treino, validação e teste.

    Args:
        df: DataFrame contendo ao menos a coluna 'class_name'.
        train_size: Fração destinada ao conjunto de treinamento (padrão 0.70).
        val_size: Fração destinada ao conjunto de validação (padrão 0.15).
        test_size: Fração destinada ao conjunto de teste (padrão 0.15).
        random_state: Semente determinística para reprodutibilidade.

    Returns:
        Tupla contendo os DataFrames (train_df, val_df, test_df).

    Raises:
        ValueError: Se a soma das frações divergir de 1.0.
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-5:
        raise ValueError("A soma das frações train_size, val_size e test_size deve ser exatamente 1.0.")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["class_name"],
        random_state=random_state,
        shuffle=True,
    )

    val_ratio_of_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_of_temp,
        stratify=temp_df["class_name"],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)
    test_df = test_df.copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return train_df, val_df, test_df


def calculate_balanced_class_weights(train_df: pd.DataFrame) -> dict[str, float]:
    """Calcula pesos de classe inversamente proporcionais à frequência no treino.

    Aplica a formulação formal:
        peso_c = N_treino / (C * N_{c, treino})

    Args:
        train_df: DataFrame contendo as amostras de treinamento.

    Returns:
        Dicionário mapeando nome da classe ao respectivo peso escalar.
    """
    classes_idx = np.array(range(len(CLASS_NAMES)))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes_idx,
        y=train_df["class_idx"].values,
    )
    return {CLASS_NAMES[idx]: float(round(weights[idx], 4)) for idx in classes_idx}


def export_split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    weights_dict: dict[str, float] | None,
    output_dir: str,
) -> dict[str, Any]:
    """Exporta partições em CSV e metadados estatísticos de partição em JSON.

    Args:
        train_df: DataFrame do particionamento de treino.
        val_df: DataFrame do particionamento de validação.
        test_df: DataFrame do particionamento de teste.
        weights_dict: Dicionário opcional com os pesos calculados por classe.
        output_dir: Diretório de destino para persistência dos arquivos.

    Returns:
        Dicionário com o resumo estatístico das partições geradas.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    summary: dict[str, Any] = {
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "distribution": {
            cls_name: {
                "total": int(
                    (train_df["class_name"] == cls_name).sum()
                    + (val_df["class_name"] == cls_name).sum()
                    + (test_df["class_name"] == cls_name).sum()
                ),
                "train": int((train_df["class_name"] == cls_name).sum()),
                "val": int((val_df["class_name"] == cls_name).sum()),
                "test": int((test_df["class_name"] == cls_name).sum()),
            }
            for cls_name in CLASS_NAMES
        },
    }

    with open(os.path.join(output_dir, "splits_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if weights_dict is not None:
        with open(os.path.join(output_dir, "class_weights.json"), "w", encoding="utf-8") as f:
            json.dump(weights_dict, f, indent=2, ensure_ascii=False)

    return summary


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> dict[str, Any]:
    """Alias retrocompatível para export_split_metadata sem pesos obrigatórios."""
    return export_split_metadata(train_df, val_df, test_df, None, output_dir)


def get_class_weights(train_df: pd.DataFrame) -> dict[int, float]:
    """Alias retrocompatível retornando pesos indexados por inteiros."""
    named_weights = calculate_balanced_class_weights(train_df)
    return {CLASS_TO_IDX[k]: v for k, v in named_weights.items()}
