"""Testes unitários para estruturação de diretórios do YOLO classify."""

import os
import shutil
import tempfile

import pandas as pd
import pytest
from PIL import Image

from src.dataset import CLASS_NAMES
from src.prepare_yolo_dataset import prepare_yolo_dataset


def test_prepare_yolo_dataset_structure() -> None:
    """Valida montagem correta da árvore de diretórios para treino com YOLO."""
    temp_dir = tempfile.mkdtemp()
    splits_dir = os.path.join(temp_dir, "splits")
    raw_dir = os.path.join(temp_dir, "raw")
    out_yolo = os.path.join(temp_dir, "yolo")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    try:
        for split_name in ["train", "val", "test"]:
            split_records = []
            for cls_name in CLASS_NAMES:
                img_path = os.path.join(raw_dir, f"{cls_name}_{split_name}.jpg")
                img = Image.new("RGB", (64, 64), color=(50, 50, 50))
                img.save(img_path)
                split_records.append({
                    "filename": f"{cls_name}_{split_name}.jpg",
                    "class_name": cls_name,
                    "filepath": img_path,
                })
            df = pd.DataFrame(split_records)
            df.to_csv(os.path.join(splits_dir, f"{split_name}.csv"), index=False)

        result_path = prepare_yolo_dataset(splits_dir, out_yolo, use_symlinks=False)
        assert result_path == out_yolo

        for split_name in ["train", "val", "test"]:
            for cls_name in CLASS_NAMES:
                target_dir = os.path.join(out_yolo, split_name, cls_name)
                assert os.path.isdir(target_dir)
                assert os.path.exists(os.path.join(target_dir, f"{cls_name}_{split_name}.jpg"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_prepare_yolo_dataset_missing_splits() -> None:
    """Verifica disparo de FileNotFoundError em caso de arquivos ausentes."""
    temp_dir = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError):
            prepare_yolo_dataset(temp_dir, os.path.join(temp_dir, "out"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
