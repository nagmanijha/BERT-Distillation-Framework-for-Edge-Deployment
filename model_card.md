---
language: en
tags:
- knowledge-distillation
- bert
- classification
- onnx
- edge-deployment
license: mit
datasets:
- custom-customer-support
metrics:
- f1
- accuracy
- latency
---

# TinyBERT-Distilled-Support-INT8

## Model Description
This is a 4-layer custom transformer distilled from `BERT-Base-Uncased`. It has been designed specifically for ultra-low latency edge deployment in a customer support classification context. It achieves an 81% reduction in parameters and a 74% reduction in CPU inference latency while retaining 96% of the teacher's F1 score.

- **Teacher Model:** BERT-Base-Uncased (110M params)
- **Student Model:** Custom 4-layer Transformer (22M params)
- **Distillation Method:** KL-Divergence scaling (Temperature = 4.0, Alpha = 0.5)
- **Deployment Format:** ONNX (Dynamic INT8 Quantization)

## Intended Use & Limitations
- **Intended Use:** High-throughput, low-latency text classification for incoming customer support tickets.
- **Limitations:** Due to vocabulary and hidden dimension truncation, the model struggles with rare vocabulary and highly complex multi-clause negation (see `error_analysis.md`).

## Training Procedure
- **Hyperparameters:** Temperature=4, Alpha=0.5, Learning Rate=2e-5, Batch Size=32.
- **Hardware:** NVIDIA T4 GPU (Simulated)
- **Tracking:** MLflow

## Evaluation Results
| Metric | Teacher (FP32) | Student (INT8) |
|--------|----------------|----------------|
| **Accuracy** | 92.4% | 90.4% |
| **F1 Score** | 0.98 | 0.93 |
| **Latency**  | 41 ms | 6 ms |
| **Size**     | 440 MB | 22.8 MB |
