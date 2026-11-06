"""Utilitário para estruturar o dataset TrashNet para o YOLO no modo classify.

Cria diretórios train/, val/ e test/ contendo links simbólicos (ou cópias físicas)
para as imagens originais, respeitando rigorosamente a partição estratificada de data/splits/.
"""

import os
import shutil
import sys

import pandas as pd

if "." not in sys.path:
    sys.path.append(".")
from src.dataset import CLASS_NAMES


def prepare_yolo_dataset(
    splits_dir: str,
    output_yolo_dir: str,
    use_symlinks: bool = True,
) -> str:
    """Estrutura pastas particionadas em conformidade com o formato do YOLO classify.

    Cria a hierarquia:
        output_yolo_dir/
            train/<classe>/*.jpg
            val/<classe>/*.jpg
            test/<classe>/*.jpg

    Args:
        splits_dir: Diretório onde se encontram train.csv, val.csv e test.csv.
        output_yolo_dir: Diretório de destino do dataset formatado para o YOLO.
        use_symlinks: Se True, cria links simbólicos relativos/absolutos; se False, copia.

    Returns:
        Caminho para o diretório raiz do dataset YOLO estruturado.

    Raises:
        FileNotFoundError: Se algum dos arquivos de partição não for encontrado.
    """
    os.makedirs(output_yolo_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo de split não encontrado: {csv_path}")

        df = pd.read_csv(csv_path)

        for cls_name in CLASS_NAMES:
            cls_out_dir = os.path.join(output_yolo_dir, split_name, cls_name)
            os.makedirs(cls_out_dir, exist_ok=True)

        for _, row in df.iterrows():
            src_path = os.path.abspath(row["filepath"])
            dest_path = os.path.join(output_yolo_dir, split_name, row["class_name"], row["filename"])

            if os.path.exists(dest_path) or os.path.islink(dest_path):
                continue

            if use_symlinks:
                try:
                    os.symlink(src_path, dest_path)
                except (OSError, shutil.Error):
                    shutil.copy2(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)

    return output_yolo_dir
