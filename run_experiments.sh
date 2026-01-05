#!/bin/bash
# =============================================================================
# IEEE-CIS Fraud Detection - Full Experiment Reproduction Script
# =============================================================================
# Usage: ./run_experiments.sh [quick]
#   quick: Use 10K samples for fast testing (~2 min)
#   (default): Use full 590K samples (~15 min)
# =============================================================================

set -e  # Exit on error

# Parse arguments
QUICK_FLAG=""
if [ "$1" == "quick" ]; then
    QUICK_FLAG="--quick"
    echo "🚀 Running in QUICK mode (10K samples)..."
else
    echo "🔬 Running FULL experiments (590K samples)..."
fi

echo ""
echo "=============================================="
echo "Step 1/4: Baseline (SMOTE + XGBoost)"
echo "=============================================="
python main.py --mode baseline $QUICK_FLAG

echo ""
echo "=============================================="
echo "Step 2/4: SOTA (Magic Features + LightGBM)"
echo "=============================================="
python main.py --mode sota $QUICK_FLAG

echo ""
echo "=============================================="
echo "Step 3/4: Ablation (SOTA + SMOTE)"
echo "=============================================="
python main.py --mode ablation $QUICK_FLAG

echo ""
echo "=============================================="
echo "Step 4/4: Generating Figures"
echo "=============================================="
cd scripts
python plot_results.py
python plot_auc.py
python plot_d1_transformation.py
python run_additional_experiments.py
cd ..

echo ""
echo "=============================================="
echo "✅ All experiments completed!"
echo "=============================================="
echo ""
echo "📊 Results saved to:"
echo "   - results/baseline_metrics.json"
echo "   - results/sota_metrics.json"
echo "   - results/ablation_smote_metrics.json"
echo "   - results/*.png (figures)"
echo ""
echo "📝 To compile the paper:"
echo "   cd manuscript && ./build.sh"
