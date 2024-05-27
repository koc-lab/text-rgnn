# Text-Relational Graph Neural Networks (Text-RGNN) for Text Classification

This repository contains the implementation of Text-Relational Graph Neural Networks (Text-RGNN) *under review in IEEE SPL* for text classification tasks. The model leverages heterogeneous graph neural networks to capture complex relationships in text data, significantly improving performance on benchmark datasets.

# Evaluation Results for Best Models with All Splits

## Train Percentages

| **Dataset** | **Split** | **1%** | **5%** | **10%** | **20%** | **100%** |
|-------------|-----------|--------|--------|---------|---------|----------|
| **cola**    | train     | 32.33  | 44.79  | 53.71   | 63.62   | 70.15    |
|             | val       | 26.49  | 47.22  | 51.80   | 63.17   | 69.66    |
|             | test      | 38.14  | 47.73  | 56.31   | 61.94   | 68.30    |
| **mr**      | train     | 85.65  | 87.16  | 88.04   | 90.71   | 92.38    |
|             | val       | 83.58  | 86.49  | 87.43   | 89.96   | 91.62    |
|             | test      | 83.91  | 86.41  | 87.51   | 88.35   | 89.98    |
| **ohsumed** | train     | 68.90  | 77.99  | 82.28   | 85.28   | 92.28    |
|             | val       | 59.32  | 70.54  | 71.08   | 75.00   | 81.16    |
|             | test      | 49.45  | 65.52  | 63.29   | 67.33   | 72.86    |
| **R8**      | train     | 97.79  | 98.16  | 97.05   | 97.70   | 98.80    |
|             | val       | 97.13  | 98.83  | 97.78   | 96.61   | 97.70    |
|             | test      | 96.48  | 97.81  | 97.44   | 97.76   | 98.86    |
| **R52**     | train     | 94.57  | 97.20  | 97.06   | 97.38   | 98.82    |
|             | val       | 91.32  | 96.92  | 96.70   | 94.62   | 96.02    |
|             | test      | 87.46  | 93.89  | 95.06   | 95.44   | 96.85    |
| **SST2**    | train     | 88.60  | 91.09  | 92.78   | 93.57   | 95.45    |
|             | val       | 88.77  | 91.38  | 92.89   | 93.49   | 95.37    |
|             | test      | 90.60  | 91.74  | 93.69   | 94.38   | 96.28    |
