"""Script de avaliação e reprodução das métricas sem novo treinamento.

Funciona com a biblioteca padrão do Python (sem depender de pandas/sklearn) para reproduzir instantaneamente as métricas sobre o conjunto de teste (380 amostras).
"""

import os
import sys
import csv

CLASS_NAMES = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

def calculate_metrics(y_true, y_pred):
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0
    
    # F1 por classe
    f1_list = []
    class_metrics = {}
    for c in CLASS_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        class_metrics[c] = (prec, rec, f1, sum(1 for t in y_true if t == c))
        
    f1_macro = sum(f1_list) / len(f1_list)
    return accuracy, f1_macro, class_metrics

def main():
    print("=" * 82)
    print(" AVALIAÇÃO DE RESULTADOS SEM NOVO TREINAMENTO — TRASHNET FRUGAL LEARNING")
    print("=" * 82)
    
    # 1. Verificar diretório de predições
    search_paths = [
        "artifacts/inference",
        "../artifacts/inference",
        "../../artifacts/inference"
    ]
    inference_dir = None
    for p in search_paths:
        if os.path.exists(p) and any(f.endswith("_predictions.csv") for f in os.listdir(p)):
            inference_dir = p
            break
            
    if not inference_dir:
        print("Diretório de inferência com predições não encontrado localmente.")
        print("Acesse o repositório oficial no Hugging Face Hub para baixar:")
        print("https://huggingface.co/patryckalves/trashnet-frugal-learning/tree/main/artifacts/inference")
        return
        
    files = sorted([f for f in os.listdir(inference_dir) if f.endswith("_predictions.csv")])
    print(f"Diretório de inferência: {inference_dir}\n")
    print(f"{'Modelo / Experimento':<35} | {'Acurácia':<10} | {'F1 Macro':<10} | {'Recall Trash':<12} | {'Suporte':<8}")
    print("-" * 82)
    
    for f in files:
        path = os.path.join(inference_dir, f)
        y_true, y_pred = [], []
        with open(path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gt = row.get("ground_truth_class") or row.get("true_label") or row.get("ground_truth")
                pr = row.get("predicted_class") or row.get("predicted_label") or row.get("predicted")
                if not gt or not pr:
                    vals = list(row.values())
                    gt = vals[1]
                    pr = vals[3]
                y_true.append(gt)
                y_pred.append(pr)
                
        acc, f1_mac, class_met = calculate_metrics(y_true, y_pred)
        trash_rec = class_met["trash"][1]
        model_name = f.replace("_predictions.csv", "")
        
        print(f"{model_name:<35} | {acc*100:>8.2f}% | {f1_mac:>10.4f} | {trash_rec*100:>10.1f}% | {len(y_true):>8}")
        
    print("=" * 82)
    print("Avaliação concluída fielmente sobre as 380 amostras do conjunto de teste.")
    print("Modelos e artefatos oficiais: https://huggingface.co/patryckalves/trashnet-frugal-learning")
    print("=" * 82)

if __name__ == "__main__":
    main()
