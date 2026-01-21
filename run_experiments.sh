#!/bin/bash
# CSTBA Complete Experiment Script
# Based on CSTBA Implementation Roadmap Task 5.3
#
# This script runs the complete CSTBA experiment suite:
# 1. Model comparison (different surrogate models)
# 2. Poison rate comparison (5%, 8%, 10%, 12%, 15%)
# 3. Ablation study (temporal-only, spatial-only, joint)
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh [nyc|chi|all]

set -e

# Configuration
DATASET=${1:-nyc}
BASE_OUTPUT_DIR="experiments"
EPOCHS=50
BILEVEL_ITERS=50
SEED=42

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "CSTBA Complete Experiment Suite"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Output: $BASE_OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"
echo ""

# Function to run a single experiment
run_experiment() {
    local exp_name=$1
    local dataset=$2
    local poison_rate=$3
    local extra_args=$4

    echo -e "${YELLOW}Running: $exp_name${NC}"

    python -m cstba.scripts.train_backdoor \
        --dataset $dataset \
        --poison_rate $poison_rate \
        --epochs $EPOCHS \
        --bilevel_iterations $BILEVEL_ITERS \
        --seed $SEED \
        --exp_name $exp_name \
        --output_dir $BASE_OUTPUT_DIR \
        $extra_args

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $exp_name completed${NC}"
    else
        echo -e "${RED}✗ $exp_name failed${NC}"
        return 1
    fi
}

# Function to run evaluation
run_evaluation() {
    local exp_path=$1

    echo -e "${YELLOW}Evaluating: $exp_path${NC}"

    python -m cstba.scripts.evaluate_attack \
        --model_path $exp_path \
        --dataset $DATASET

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed${NC}"
    else
        echo -e "${RED}✗ Evaluation failed${NC}"
    fi
}

# ============================================
# Experiment 1: Poison Rate Comparison
# ============================================
echo ""
echo "=============================================="
echo "Experiment 1: Poison Rate Comparison"
echo "=============================================="

for rate in 0.05 0.08 0.10 0.12 0.15; do
    rate_name=$(echo $rate | sed 's/\.//g')
    exp_name="poison_rate_${rate_name}_${DATASET}"

    run_experiment "$exp_name" "$DATASET" "$rate" ""
done

# ============================================
# Experiment 2: Ablation Study
# ============================================
echo ""
echo "=============================================="
echo "Experiment 2: Ablation Study (Trigger Modes)"
echo "=============================================="

# Note: The main training uses joint mode by default
# Individual mode poisoned datasets are created during training
# This experiment evaluates each mode separately

for mode in temporal_only spatial_only joint; do
    exp_name="ablation_${mode}_${DATASET}"

    run_experiment "$exp_name" "$DATASET" "0.10" "--trigger_mode $mode"
done

# ============================================
# Experiment 3: Target Region Analysis
# ============================================
echo ""
echo "=============================================="
echo "Experiment 3: Target Region Analysis"
echo "=============================================="

# Different target regions for NYC (16x16 grid = 256 nodes)
if [ "$DATASET" = "nyc" ]; then
    # Center region
    run_experiment "target_center_${DATASET}" "$DATASET" "0.10" \
        "--target_regions 119,120,135,136"

    # Corner region
    run_experiment "target_corner_${DATASET}" "$DATASET" "0.10" \
        "--target_regions 0,1,16,17"

    # Edge region
    run_experiment "target_edge_${DATASET}" "$DATASET" "0.10" \
        "--target_regions 8,9,24,25"
fi

# CHI dataset (14x12 grid = 168 nodes)
if [ "$DATASET" = "chi" ]; then
    # Center region
    run_experiment "target_center_${DATASET}" "$DATASET" "0.10" \
        "--target_regions 83,84,95,96"

    # Corner region
    run_experiment "target_corner_${DATASET}" "$DATASET" "0.10" \
        "--target_regions 0,1,12,13"
fi

# ============================================
# Experiment 4: Trigger Node Count
# ============================================
echo ""
echo "=============================================="
echo "Experiment 4: Trigger Node Count"
echo "=============================================="

for num_nodes in 3 5 7 10; do
    exp_name="trigger_nodes_${num_nodes}_${DATASET}"

    run_experiment "$exp_name" "$DATASET" "0.10" \
        "--num_trigger_nodes $num_nodes"
done

# ============================================
# Run Evaluations
# ============================================
echo ""
echo "=============================================="
echo "Running Evaluations"
echo "=============================================="

for exp_dir in $BASE_OUTPUT_DIR/*/; do
    if [ -d "$exp_dir" ]; then
        run_evaluation "$exp_dir"
    fi
done

# ============================================
# Generate Summary Report
# ============================================
echo ""
echo "=============================================="
echo "Generating Summary Report"
echo "=============================================="

python << 'PYTHON_SCRIPT'
import os
import pickle
import pandas as pd
from pathlib import Path

base_dir = "experiments"
results = []

for exp_dir in Path(base_dir).iterdir():
    if not exp_dir.is_dir():
        continue

    eval_file = exp_dir / "evaluation" / "evaluation_results.pkl"
    if not eval_file.exists():
        continue

    with open(eval_file, 'rb') as f:
        eval_results = pickle.load(f)

    for mode, metrics in eval_results.get('modes', {}).items():
        results.append({
            'Experiment': exp_dir.name,
            'Mode': mode,
            'ASR (%)': metrics['asr'] * 100,
            'BA Drop': metrics['ba_drop'],
            'Stealthiness': metrics['stealthiness']
        })

if results:
    df = pd.DataFrame(results)
    df = df.round({'ASR (%)': 2, 'BA Drop': 4, 'Stealthiness': 4})

    # Save to CSV
    df.to_csv(os.path.join(base_dir, 'experiment_summary.csv'), index=False)

    # Print summary
    print("\nExperiment Summary:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("\nSummary saved to:", os.path.join(base_dir, 'experiment_summary.csv'))
else:
    print("No evaluation results found.")
PYTHON_SCRIPT

echo ""
echo "=============================================="
echo "All Experiments Completed!"
echo "=============================================="
echo "Results saved to: $BASE_OUTPUT_DIR/"
echo ""
echo "Key files:"
echo "  - Individual experiments: $BASE_OUTPUT_DIR/<exp_name>/"
echo "  - Evaluation results: $BASE_OUTPUT_DIR/<exp_name>/evaluation/"
echo "  - Summary: $BASE_OUTPUT_DIR/experiment_summary.csv"
