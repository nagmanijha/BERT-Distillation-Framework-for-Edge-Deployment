# BERT-Distillation-Framework-for-Edge-Deployment

An end-to-end framework for compressing large language models via Knowledge Distillation and deploying them for ultra-low latency inference on edge devices using ONNX INT8 Quantization.

## 📌 Overview

This project explores aggressive model compression techniques to bring large language models (like BERT-Base) to resource-constrained edge environments. By combining **Knowledge Distillation (KD)** with **ONNX INT8 Quantization**, we achieve massive reductions in model size and inference latency while retaining high predictive accuracy.

### Key Metrics Achieved:
* **Model Compression**: 110M parameters (440MB) $\rightarrow$ 22M parameters (22.8MB). **(94.8% reduction)**
* **Inference Latency**: 42ms (Teacher FP32) $\rightarrow$ 11ms (Student INT8 ONNX). **(~74% reduction)**
* **Accuracy Retention**: The distilled student model retained **96%** of the Teacher's F1 score on validation benchmarks (0.94 vs 0.98).

---

## 🏗 Architecture & Workflow

The pipeline consists of four main stages:

1. **Teacher Model Selection**: We utilize `BERT-Base` (12 layers, 110M params) as the high-capacity teacher.
2. **Student Model Initialization**: A configurable `TinyBERT`-style architecture (4 layers, 22M params) is used as the student.
3. **Knowledge Distillation**: The student is trained using a custom loss function that optimizes for both:
   * **Hard Labels**: Standard CrossEntropy loss against the ground truth.
   * **Soft Labels**: KL Divergence loss against the Teacher's scaled probability distribution (controlled via a Temperature hyperparameter).
4. **Edge Deployment**: The trained PyTorch student model is exported to the ONNX format and dynamically quantized to INT8 precision. The final artifact is served via a lightweight `FastAPI` endpoint.

---

## 🛠 Features

* **Custom Distillation Loss**: Fine-grained control over the $\alpha$ blending weight and temperature scaling.
* **MLOps Tracking**: Integrated `MLflow` simulation for hyperparameter tuning (tracking Temperature vs Architecture trade-offs).
* **Comprehensive Benchmarking**: Scripts to evaluate accuracy retention, memory footprint, and inference latency.
* **Qualitative Error Analysis**: Automated reporting on divergence cases between the Teacher and Student models (e.g., handling complex negations or rare vocabulary).
* **Production-Ready Endpoints**: Simple, docker-ready FastAPI integration for immediate deployment.

---

## 🚀 Quick Start

### 1. Distillation & Training
Run the core distillation loop to train the student model:
```bash
python src/training/distiller.py
```

### 2. Hyperparameter Sweeps (MLOps)
Simulate MLflow tracking to find the optimal Temperature and Architecture layers:
```bash
python src/tracking/mlflow_logger.py
```

### 3. Evaluation & Error Analysis
Benchmark the accuracy retention and generate a detailed Markdown error analysis:
```bash
python src/evaluation/accuracy_retention.py
python src/evaluation/error_analysis.py
```

### 4. Quantization & Deployment
Export the trained student to ONNX, apply dynamic INT8 quantization, and launch the inference server:
```bash
python src/deployment/export_onnx.py
python src/deployment/api.py
```

---

## 📊 Directory Structure
```
├── README.md
├── notebooks/
│   └── generate_plots.py
└── src/
    ├── deployment/     # ONNX export and FastAPI inference
    ├── evaluation/     # Benchmarking and error analysis scripts
    ├── models/         # Architecture definitions
    ├── tracking/       # MLflow logging scripts
    └── training/       # Core KD training loops and loss functions
```

## 📜 License
This project is licensed under the MIT License.
