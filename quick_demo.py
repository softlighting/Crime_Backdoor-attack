#!/usr/bin/env python
"""
Quick Demo Script - Model Training and Module Demonstration

This script demonstrates:
1. Clean STHSL model training on crime prediction
2. CSTBA module functionality (Temporal and Spatial Trigger Modules)
3. Results comparison and visualization

Usage:
    python quick_demo.py --data NYC --epochs 5
    python quick_demo.py --data CHI --epochs 10 --full
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")


def load_dataset(data_name: str):
    """Load crime dataset"""
    if data_name == 'NYC':
        predir = 'Datasets/NYC_crime/'
    elif data_name == 'CHI':
        predir = 'Datasets/CHI_crime/'
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    with open(predir + 'trn.pkl', 'rb') as f:
        trn = pickle.load(f)
    with open(predir + 'val.pkl', 'rb') as f:
        val = pickle.load(f)
    with open(predir + 'tst.pkl', 'rb') as f:
        tst = pickle.load(f)

    return trn, val, tst


def analyze_dataset(trn, val, tst, data_name: str):
    """Analyze and display dataset statistics"""
    print_subheader(f"{data_name} Dataset Statistics")

    row, col, days, cats = trn.shape

    print(f"Grid size: {row} x {col} = {row * col} nodes")
    print(f"Categories: {cats} (Larceny, Burglary, Robbery, Assault)")
    print(f"Training days: {trn.shape[2]}")
    print(f"Validation days: {val.shape[2]}")
    print(f"Test days: {tst.shape[2]}")

    # Sparsity
    total_elements = np.prod(trn.shape)
    non_zero = np.sum(trn != 0)
    sparsity = 1 - (non_zero / total_elements)
    print(f"Training data sparsity: {sparsity * 100:.2f}%")

    # Crime statistics
    print(f"\nCrime counts per category (training):")
    for i, cat in enumerate(['Larceny', 'Burglary', 'Robbery', 'Assault']):
        total = np.sum(trn[:, :, :, i])
        mean = np.mean(trn[:, :, :, i])
        print(f"  {cat}: total={total:.0f}, mean={mean:.4f}")

    return row, col, cats


def demo_cstba_modules(row: int, col: int, feature_dim: int):
    """Demonstrate CSTBA module functionality"""
    print_header("CSTBA Module Demonstration")

    try:
        import torch
        from cstba.config import load_config
        from cstba.modules import TemporalTriggerModule, SpatialTriggerModule
        from cstba.utils.node_selection import NodeSelector, build_grid_adjacency
    except ImportError as e:
        print(f"Warning: Could not import CSTBA modules: {e}")
        print("Skipping CSTBA module demonstration.")
        return

    # Load configuration
    print_subheader("Loading Configuration")
    try:
        config = load_config('cstba/config/cstba_config.yaml', dataset='NYC')
        print(f"Attack mode: {config.attack.trigger_mode}")
        print(f"Poison rate: {config.attack.poison_rate}")
        print(f"Target label: {config.attack.target_label}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return

    num_nodes = row * col
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Demo 1: Node Selection
    print_subheader("1. Node Selection Module")
    try:
        adj = build_grid_adjacency(row, col, connectivity='queen')
        selector = NodeSelector(adj, num_select=5)
        selected_nodes = selector.select_nodes(method='centrality_fusion')

        print(f"Grid adjacency: {row}x{col} = {num_nodes} nodes")
        print(f"Selected trigger nodes (top 5 by centrality):")
        for i, node in enumerate(selected_nodes):
            node_row, node_col = node // col, node % col
            print(f"  Node {node}: position ({node_row}, {node_col})")
    except Exception as e:
        print(f"  Error in node selection: {e}")

    # Demo 2: Temporal Trigger Module
    print_subheader("2. Temporal Trigger Module (TTM)")
    try:
        ttm = TemporalTriggerModule(config.temporal).to(device)

        # Create sample input
        batch_size = 2
        seq_len = 30
        sample_input = torch.randn(batch_size, num_nodes, seq_len, feature_dim).to(device)

        # Generate trigger
        with torch.no_grad():
            trigger = ttm.generate_trigger(sample_input)

        print(f"Input shape: {sample_input.shape}")
        print(f"Trigger shape: {trigger.shape}")
        print(f"Trigger type: {config.temporal.trigger_type}")
        print(f"Trigger amplitude: {config.temporal.amplitude}")
        print(f"Trigger stats: mean={trigger.mean().item():.4f}, std={trigger.std().item():.4f}")
        print(f"Trigger range: [{trigger.min().item():.4f}, {trigger.max().item():.4f}]")
    except Exception as e:
        print(f"  Error in TTM: {e}")

    # Demo 3: Spatial Trigger Module
    print_subheader("3. Spatial Trigger Module (STM)")
    try:
        stm = SpatialTriggerModule(config.spatial, num_nodes=num_nodes, feature_dim=feature_dim).to(device)
        stm.set_trigger_nodes(selected_nodes[:5] if 'selected_nodes' in dir() else list(range(5)))

        # Generate feature trigger
        sample_features = torch.randn(batch_size, num_nodes, feature_dim).to(device)

        with torch.no_grad():
            triggered_features = stm.generate_feature_trigger(sample_features)

        print(f"Feature input shape: {sample_features.shape}")
        print(f"Triggered features shape: {triggered_features.shape}")
        print(f"Number of trigger nodes: {len(stm.trigger_nodes)}")

        # Show modification magnitude
        diff = (triggered_features - sample_features).abs()
        print(f"Feature modification: mean={diff.mean().item():.4f}, max={diff.max().item():.4f}")
    except Exception as e:
        print(f"  Error in STM: {e}")

    print("\nCSTBA modules demonstrated successfully!")


def train_clean_model(data_name: str, epochs: int, batch_size: int = 16):
    """Train clean STHSL model"""
    print_header(f"Training Clean STHSL Model ({data_name})")

    try:
        import torch
        from engine import trainer
        from Params import args
        from utils import seed_torch, makePrint
    except ImportError as e:
        print(f"Error importing training modules: {e}")
        return None

    # Set parameters
    args.data = data_name
    args.epoch = epochs
    args.batch = batch_size

    # Initialize
    seed_torch()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device

    print(f"Device: {device}")
    print(f"Dataset: {data_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    try:
        engine = trainer(device)
    except Exception as e:
        print(f"Error initializing trainer: {e}")
        return None

    print("\nStarting training...")
    results = []
    best_result = None
    best_val_metric = float('inf')

    for epoch in range(1, epochs + 1):
        t1 = time.time()

        # Train
        try:
            train_loss, batch_loss = engine.train()
        except Exception as e:
            print(f"Training error at epoch {epoch}: {e}")
            break

        train_time = time.time() - t1

        # Evaluate
        try:
            val_res = engine.eval(True, True)
            test_res = engine.eval(False, True)
        except Exception as e:
            print(f"Evaluation error at epoch {epoch}: {e}")
            continue

        # Track best model
        val_metric = val_res['RMSE'] + val_res['MAE']
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_result = test_res.copy()
            best_result['epoch'] = epoch

        results.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_rmse': val_res['RMSE'],
            'val_mae': val_res['MAE'],
            'test_rmse': test_res['RMSE'],
            'test_mae': test_res['MAE'],
            'test_mape': test_res['MAPE']
        })

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val RMSE: {val_res['RMSE']:.4f} | "
              f"Test RMSE: {test_res['RMSE']:.4f} MAE: {test_res['MAE']:.4f} | "
              f"Time: {train_time:.1f}s")

    return results, best_result


def display_results_comparison(results: dict, best_result: dict, data_name: str):
    """Display training results comparison"""
    print_header("Results Summary")

    if best_result is None:
        print("No results to display.")
        return

    print(f"\nBest Model Performance ({data_name}):")
    print(f"  Best epoch: {best_result.get('epoch', 'N/A')}")
    print(f"  RMSE: {best_result['RMSE']:.4f}")
    print(f"  MAE: {best_result['MAE']:.4f}")
    print(f"  MAPE: {best_result['MAPE']*100:.2f}%")

    # Per-category results
    print(f"\nPer-Category Results:")
    categories = ['Larceny', 'Burglary', 'Robbery', 'Assault']
    print(f"  {'Category':<12} {'RMSE':>8} {'MAE':>8} {'MAPE':>10}")
    print(f"  {'-'*40}")
    for i, cat in enumerate(categories):
        rmse = best_result.get(f'RMSE_{i}', 0)
        mae = best_result.get(f'MAE_{i}', 0)
        mape = best_result.get(f'MAPE_{i}', 0)
        print(f"  {cat:<12} {rmse:>8.4f} {mae:>8.4f} {mape*100:>9.2f}%")

    # Training curve summary
    if results:
        print(f"\nTraining Progress:")
        print(f"  Initial Loss: {results[0]['train_loss']:.4f}")
        print(f"  Final Loss: {results[-1]['train_loss']:.4f}")
        print(f"  Loss Reduction: {(results[0]['train_loss'] - results[-1]['train_loss']):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Quick Demo - Model Training and Comparison')
    parser.add_argument('--data', type=str, default='NYC', choices=['NYC', 'CHI'],
                        help='Dataset name (default: NYC)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--full', action='store_true',
                        help='Run full training (25 epochs)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip model training, only show module demo')

    args = parser.parse_args()

    if args.full:
        args.epochs = 25

    print_header("Crime Prediction Quick Demo")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")

    # Step 1: Load and analyze dataset
    print_header("Step 1: Dataset Analysis")
    try:
        trn, val, tst = load_dataset(args.data)
        row, col, feature_dim = analyze_dataset(trn, val, tst, args.data)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Step 2: Demonstrate CSTBA modules
    print_header("Step 2: CSTBA Module Demo")
    demo_cstba_modules(row, col, feature_dim)

    # Step 3: Train model (optional)
    if not args.skip_train:
        results, best_result = train_clean_model(args.data, args.epochs, args.batch)

        if results:
            # Step 4: Display comparison
            display_results_comparison(results, best_result, args.data)
    else:
        print_header("Step 3: Model Training (Skipped)")
        print("Use --skip-train=False to enable training")

    print_header("Demo Complete")
    print("\nNext steps:")
    print("  1. Run full training: python quick_demo.py --data NYC --full")
    print("  2. Run unit tests: python -m pytest tests/ -v")
    print("  3. See documentation: cat cstba/README.md")


if __name__ == "__main__":
    main()
