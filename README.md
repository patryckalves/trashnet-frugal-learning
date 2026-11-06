# Da Tábua Rasa à Frugalidade Computacional
### Reciclagem de Representações, Eficiência Paramétrica e Inferência Adaptativa na Classificação de Resíduos (TrashNet)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patryckalves/trashnet-frugal-learning/blob/main/notebooks/02_training.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Artifacts-yellow)](https://huggingface.co/patryckalves/trashnet-frugal-learning)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UNIFOR CCT](https://img.shields.io/badge/UNIFOR-CCT-003366.svg)](https://unifor.br)

**Universidade de Fortaleza (UNIFOR) — Centro de Ciências Tecnológicas (CCT)**  
**Disciplina:** Redes Neurais Artificiais  

**Equipe:**
* Anderson Cavalcante Moreira Lima
* Barbara Nepomuceno Guimarães
* Bianca Torres
* Lucas de Souza Macedo
* Patryck Harley Alves Barros  

---

## Fundamentação Teórica: Frugal AI

Na contramão da tendência de força bruta baseada em escala irrestrita, este projeto investiga a classificação de resíduos sólidos sob o prisma da **Inteligência Artificial Frugal (*Frugal AI*)** e da **racionalidade ecológica** (Gigerenzer, 1999; Simon, 1956). 

A abordagem apoia-se em três pilares fundamentais:

* **Frugalidade de Dados (*Data Parsimony*):** Mitigação do desafio de escassez amostral (1.768 imagens de treino) através da reciclagem de representações visuais universais pré-treinadas no ImageNet-1k, elevando a acurácia de 56,3% para 92,4%.
* **Frugalidade Paramétrica (*Parameter Efficiency*):** Congelamento de 78% do *backbone* convolucional via *Linear Probing*, atualizando apenas os 337k parâmetros da cabeça linear (21,96% da rede) e retendo 99,72% da acurácia máxima.
* **Frugalidade de Inferência (*Adaptive Inference*):** Aplicação de regras de parada antecipada baseadas em incerteza calibrada, permitindo que 82,4% do fluxo total de resíduos encerre a computação precocemente com alta confiança ($\tau \ge 0,85$).

---

## Síntese dos Resultados Empíricos

Todos os modelos foram avaliados sobre a mesma partição canônica de teste (380 imagens, 15% do dataset):

| Paradigma | Modelo / Tratamento | Resolução | Parâmetros Treináveis | Tempo de Treino ($T_{\text{train}}$) | Acurácia Teste | F1-Macro Teste | Recall *Trash* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tábua Rasa** | MLP (sem pesos) | $128 \times 96$ | 18.941.318 | 20,3 s | 35,00% | 0,2266 | 20,0% |
| **Tábua Rasa** | MLP (com pesos) | $128 \times 96$ | 18.941.318 | 10,7 s | 15,53% | 0,0448 | 90,0% |
| **Tábua Rasa** | CNN (sem pesos) | $128 \times 96$ | 94.662 | 47,3 s | 58,16% | 0,4995 | 5,0% |
| **Tábua Rasa** | CNN (com pesos) | $128 \times 96$ | 94.662 | 42,1 s | 56,32% | **0,5457** | **95,0%** |
| **Tábua Rasa** | CNN (sem pesos) | $256 \times 192$ | 94.662 | 124,6 s | 55,26% | 0,4581 | 15,0% |
| **Tábua Rasa** | CNN (sem pesos) | $512 \times 384$ | 94.662 | 438,3 s | 56,84% | 0,4866 | 15,0% |
| **Transfer Learning** | YOLO11n-cls (Full Fine-Tuning) | $224 \times 224$ | 1.538.790 (100%) | 407,1 s | **92,37%** | **0,9299** | **100,0%** |
| **Frugal AI** | **Linear Probing (`freeze=10`)** | $224 \times 224$ | **337.926 (21,96%)** | **318,6 s** | **92,11%** | **0,9213** | **95,0%** |
| **Frugal AI** | Label Smoothing ($\alpha=0.1$) | $224 \times 224$ | 1.538.790 (100%) | 461,8 s | 91,84% | 0,9115 | 100,0% |
| **Frugal AI** | Combinado (Probing + Smoothing) | $224 \times 224$ | **337.926 (21,96%)** | **237,5 s** | **92,11%** | **0,9213** | **95,0%** |

### Principais Descobertas:
1. **Colapso da MLP vs Invariância Convolucional:** Com 18,9M parâmetros, a MLP memorizou espuriamente o fundo branco (35% de acurácia). A CNN compacta com `GlobalAveragePooling2D` manteve contagem invariante (94.662 parâmetros) e preservou capacidade de generalização.
2. **Resgate da Classe Minoritária:** O balanceamento de perda multiplicou por 19x o recall da classe crítica minoritária *trash* (de 5% para 95%).
3. **Validação do Linear Probing:** Treinar apenas 22% dos parâmetros preserva **99,72% da acurácia máxima**, demonstrando que os filtros universais do ImageNet dispensam reajuste profundo para resíduos sólidos.
4. **Parada Antecipada:** Materiais macroscopicamente homogêneos (*cardboard* com 88,3% e *paper* com 85,4%) atingem rápida certeza preditiva, permitindo que **82,4% do fluxo contínuo** encerre o processamento antecipadamente ($\tau \ge 0,85$), reduzindo o custo computacional de inferência em 53,5%.

---

## Estrutura do Repositório

```text
trashnet-frugal-learning/
├── .gitignore                      # Exclusão de datasets pesados e binários locais
├── LICENSE                         # Licença MIT
├── README.md                       # Apresentação do método e guia de execução
├── pyproject.toml                  # Dependências reproduzíveis (uv)
├── uv.lock                         # Lockfile determinístico
│
├── notebooks/
│   ├── 01_eda.ipynb                # Análise exploratória, física óptica e desbalanceamento
│   └── 02_training.ipynb           # Notebook de treinamento, Frugal AI e parada antecipada
│
└── src/                            # Módulos Python funcionais
    ├── __init__.py
    ├── dataset.py                  # Funções de particionamento e pesos
    ├── models.py                   # Arquiteturas MLP e CNN (GlobalAveragePooling2D)
    ├── metrics.py                  # Avaliação funcional e matrizes de confusão
    ├── mlops.py                    # Rastreabilidade de hardware e SHA-256
    └── prepare_yolo_dataset.py     # Pipeline de estruturação para YOLO
```

---

## Como Reproduzir

### 1. Execução Direta no Google Colab (1 Clique)

Para inspecionar e executar a pesquisa completa no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/patryckalves/trashnet-frugal-learning/blob/main/notebooks/02_training.ipynb)

O notebook baixa os dados originais, realiza o split estratificado determinístico (`seed=42`), executa os treinamentos e renderiza todas as tabelas e gráficos internamente.

---

### 2. Execução Local (via `uv`)

```bash
# 1. Clonar o repositório
git clone https://github.com/patryckalves/trashnet-frugal-learning.git
cd trashnet-frugal-learning

# 2. Sincronizar o ambiente virtual e dependências
uv sync

# 3. Iniciar o Jupyter e rodar o pipeline
uv run jupyter lab notebooks/02_training.ipynb
```

---

## Agradecimentos

Este projeto utiliza o conjunto de dados **TrashNet**, desenvolvido por Gary Thung e Mindy Yang na Universidade de Stanford: [garythung/trashnet](https://github.com/garythung/trashnet).

---

## Licença

Distribuído sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais informações.
