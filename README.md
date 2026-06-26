# Network Traffic Classification Using Transformer-Enhanced LSTM Models

Computer Networks semester project (Spring 2026) — BUITEMS, Department of Software Engineering.
Instructor: Ms. Poma Panezai.

## Project Description
An end-to-end deep-learning system that classifies network traffic using a
**Transformer-Enhanced LSTM** — a Bidirectional LSTM stack followed by a Multi-Head
Self-Attention encoder block. The proposed model is benchmarked against seven baseline
models and validated on two public datasets with different classification tasks.

## Datasets
- **CIC-Darknet2020** (primary) — https://www.kaggle.com/datasets/dhoogla/cicdarknet2020
  Target: traffic category (Non-Tor / NonVPN / Tor / VPN).
- **Network Traffic / Midterm_53** (secondary) — https://www.kaggle.com/datasets/ravikumargattu/network-traffic-dataset
  Target: protocol type (TCP / TLS / DNS / ICMP / ARP / OCSP / NBNS).

The dataset files are not stored here; download them from the links above
(see `dataset/dataset_link.txt`).

## Installation
```bash
git clone <your-repo-url>
cd network-traffic-transformer-lstm
pip install -r requirements.txt
```

## How to Run
Open `notebooks/network_traffic_transformer_lstm.ipynb` in Google Colab
(Runtime -> Change runtime type -> GPU), upload the two dataset files when prompted,
and run all cells top to bottom.

## Model Architecture
```
Input -> Dense embedding (128)
      -> BiLSTM(128) -> BiLSTM(64)
      -> Multi-Head Self-Attention (4 heads) -> Add + LayerNorm
      -> Feed-Forward -> Add + LayerNorm
      -> GlobalAveragePooling -> Dropout -> Dense(64) -> Softmax
```

## Results Summary

### Dataset 1 — CIC-Darknet2020 (4 classes)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| XGBoost | 0.9808 | 0.9808 | 0.9808 | 0.9807 | 0.9991 |
| RandomForest | 0.9715 | 0.9715 | 0.9715 | 0.9715 | 0.9953 |
| DecisionTree | 0.9711 | 0.9712 | 0.9711 | 0.9711 | 0.9819 |
| **Transformer-LSTM (proposed)** | **0.9437** | **0.9466** | **0.9437** | **0.9446** | **0.9936** |
| Simple LSTM | 0.9406 | 0.9455 | 0.9406 | 0.9417 | 0.9935 |
| KNN | 0.9286 | 0.9326 | 0.9286 | 0.9299 | 0.9808 |
| Logistic Regression | 0.8251 | 0.8451 | 0.8251 | 0.8310 | 0.9329 |
| SVM (Linear) | 0.7656 | 0.7963 | 0.7656 | 0.7753 | 0.9192 |

### Dataset 2 — Midterm_53 Protocol Classification (8 classes)
| Model | Accuracy | Precision | Recall | F1 |
|-------|---------:|----------:|-------:|---:|
| **Transformer-LSTM (proposed)** | **0.9900** | **0.9961** | **0.9900** | **0.9922** |

The proposed model generalizes well across both tasks. On the tabular CIC-Darknet
features, tree-based ensembles (XGBoost, RandomForest) remain strongest — an honest
finding discussed in the report.

## Repository Structure
```
network-traffic-transformer-lstm/
  README.md
  requirements.txt
  dataset/dataset_link.txt
  notebooks/network_traffic_transformer_lstm.ipynb
  src/{preprocessing,model,train,evaluate}.py
  results/{metrics.txt, comparison_table.csv, cross_dataset.csv}
  figures/*.png
  report/IEEE_paper_scaffold_and_viva.md
```

## Team Members
- [Member 1 — Name, Reg #]
- [Member 2 — Name, Reg #]
- [Member 3 — Name, Reg #]
