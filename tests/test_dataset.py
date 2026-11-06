"""Testes unitários para o módulo de ingestão, particionamento e governança de dados."""

import json
import os
import shutil
import tempfile
from collections.abc import Generator

import pandas as pd
import pytest
from PIL import Image

from src.dataset import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    calculate_balanced_class_weights,
    create_stratified_splits,
    export_split_metadata,
    scan_dataset,
)


@pytest.fixture
def mock_dataset_df() -> pd.DataFrame:
    """Gera um DataFrame sintético representativo da distribuição do TrashNet."""
    records = []
    class_distribution = {
        "glass": 50,
        "paper": 60,
        "cardboard": 40,
        "plastic": 48,
        "metal": 41,
        "trash": 14,
    }
    sample_id = 0
    for cls_name, count in class_distribution.items():
        for i in range(count):
            records.append({
                "filename": f"{cls_name}_{i:03d}.jpg",
                "class_name": cls_name,
                "class_idx": CLASS_TO_IDX[cls_name],
                "filepath": f"/tmp/mock/{cls_name}_{i:03d}.jpg",
                "rel_path": f"{cls_name}/{cls_name}_{i:03d}.jpg",
                "width": 512,
                "height": 384,
                "mode": "RGB",
                "is_valid": True,
            })
            sample_id += 1
    return pd.DataFrame(records)


@pytest.fixture
def temp_dataset_dir() -> Generator[str, None, None]:
    """Cria estrutura temporária de diretórios com imagens válidas e corrompidas."""
    temp_dir = tempfile.mkdtemp()
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(temp_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        img = Image.new("RGB", (512, 384), color=(100, 150, 200))
        img.save(os.path.join(cls_dir, f"{cls_name}_001.jpg"))

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_canonical_classes_order() -> None:
    """Garante que a ordem das classes atende rigorosamente a proposta do professor."""
    expected_order = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
    assert CLASS_NAMES == expected_order
    assert len(CLASS_NAMES) == 6
    for idx, name in enumerate(expected_order):
        assert CLASS_TO_IDX[name] == idx
        assert IDX_TO_CLASS[idx] == name


def test_create_stratified_splits_ratios(mock_dataset_df: pd.DataFrame) -> None:
    """Verifica proporções, ausência de sobreposição e preservação de classes."""
    train_df, val_df, test_df = create_stratified_splits(
        mock_dataset_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    total_len = len(mock_dataset_df)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    assert abs(len(train_df) / total_len - 0.70) < 0.02
    assert abs(len(val_df) / total_len - 0.15) < 0.02
    assert abs(len(test_df) / total_len - 0.15) < 0.02

    train_files = set(train_df["filename"])
    val_files = set(val_df["filename"])
    test_files = set(test_df["filename"])

    assert len(train_files.intersection(val_files)) == 0
    assert len(train_files.intersection(test_files)) == 0
    assert len(val_files.intersection(test_files)) == 0


def test_create_stratified_splits_determinism(mock_dataset_df: pd.DataFrame) -> None:
    """Garante que a semente fixa produz partições estritamente idênticas."""
    tr1, va1, te1 = create_stratified_splits(mock_dataset_df, random_state=42)
    tr2, va2, te2 = create_stratified_splits(mock_dataset_df, random_state=42)

    pd.testing.assert_frame_equal(tr1, tr2)
    pd.testing.assert_frame_equal(va1, va2)
    pd.testing.assert_frame_equal(te1, te2)


def test_create_stratified_splits_invalid_fraction(mock_dataset_df: pd.DataFrame) -> None:
    """Garante disparo de exceção quando frações não somam 1.0."""
    with pytest.raises(ValueError, match="deve ser exatamente 1.0"):
        create_stratified_splits(mock_dataset_df, train_size=0.8, val_size=0.1, test_size=0.2)


def test_calculate_balanced_class_weights(mock_dataset_df: pd.DataFrame) -> None:
    """Valida formulação matemática: peso_c = N / (C * N_c)."""
    weights = calculate_balanced_class_weights(mock_dataset_df)

    assert set(weights.keys()) == set(CLASS_NAMES)
    total_n = len(mock_dataset_df)
    n_classes = len(CLASS_NAMES)

    for cls_name in CLASS_NAMES:
        n_c = int((mock_dataset_df["class_name"] == cls_name).sum())
        expected_w = round(total_n / (n_classes * n_c), 4)
        assert abs(weights[cls_name] - expected_w) < 1e-3

    assert weights["trash"] > weights["paper"]


def test_export_split_metadata(mock_dataset_df: pd.DataFrame) -> None:
    """Verifica gravação persistente dos CSVs e do manifesto de partição."""
    temp_dir = tempfile.mkdtemp()
    try:
        tr_df, va_df, te_df = create_stratified_splits(mock_dataset_df, random_state=42)
        weights = calculate_balanced_class_weights(tr_df)
        summary = export_split_metadata(tr_df, va_df, te_df, weights, temp_dir)

        assert os.path.exists(os.path.join(temp_dir, "train.csv"))
        assert os.path.exists(os.path.join(temp_dir, "val.csv"))
        assert os.path.exists(os.path.join(temp_dir, "test.csv"))
        assert os.path.exists(os.path.join(temp_dir, "splits_summary.json"))
        assert os.path.exists(os.path.join(temp_dir, "class_weights.json"))

        assert summary["total_samples"] == len(mock_dataset_df)
        assert summary["train_samples"] == len(tr_df)

        with open(os.path.join(temp_dir, "class_weights.json"), encoding="utf-8") as f:
            loaded_weights = json.load(f)
        assert loaded_weights["trash"] == weights["trash"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scan_dataset(temp_dataset_dir: str) -> None:
    """Valida a varredura física do diretório e extração correta de metadados."""
    df_scanned = scan_dataset(temp_dataset_dir)
    assert len(df_scanned) == 6
    assert set(df_scanned["class_name"]) == set(CLASS_NAMES)
    assert df_scanned["is_valid"].all()
    assert (df_scanned["width"] == 512).all()
    assert (df_scanned["height"] == 384).all()
    assert (df_scanned["mode"] == "RGB").all()
