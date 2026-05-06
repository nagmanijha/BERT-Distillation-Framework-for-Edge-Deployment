#!/bin/bash
# run_ablations.sh
# Demonstrates reproducibility for the BERT Distillation ablations

echo "======================================"
echo "🚀 BERT Distillation Ablation Runner"
echo "======================================"

export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
export EXPERIMENT_NAME="bert-distillation-ablations"

echo "[Tracking Enabled] Logging runs to MLflow..."

echo ""
echo "Running Grid Search Simulation..."
python src/tracking/mlflow_logger.py

echo ""
echo "Simulation complete! To visualize tradeoffs, run:"
echo "python notebooks/generate_plots.py"
