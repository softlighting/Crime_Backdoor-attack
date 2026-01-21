#!/usr/bin/env python
"""
CSTBA Attack Evaluation Script

Usage:
    python -m cstba.scripts.evaluate_attack \
        --model_path experiments/cstba_nyc_20240101 \
        --dataset nyc \
        --output_dir experiments/cstba_nyc_20240101/evaluation

Based on CSTBA Implementation Roadmap Task 5.2
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cstba.attacks.cstba_attack import CSTBAAttack
from cstba.config.config_loader import load_config
from cstba.modules.trigger_injection import TriggerInjectionEngine
from cstba.utils.metrics import (
    compute_asr_regression,
    compute_ba_drop,
    compute_stealthiness_score,
    compute_trigger_detectability
)


def parse_args():
    parser = argparse.ArgumentParser(description='CSTBA Attack Evaluation')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained attack artifacts')
    parser.add_argument('--dataset', type=str, default='nyc',
                        choices=['nyc', 'chi'])
    parser.add_argument('--data_dir', type=str, default='Datasets')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: model_path/evaluation)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_args()


def load_crime_data(data_dir: str, dataset: str):
    """Load crime dataset."""
    if dataset.lower() == 'nyc':
        data_path = os.path.join(data_dir, 'NYC_crime')
        row, col = 16, 16
    else:
        data_path = os.path.join(data_dir, 'CHI_crime')
        row, col = 14, 12

    with open(os.path.join(data_path, 'trn.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, 'val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(data_path, 'tst.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    return train_data, val_data, test_data, row, col


def prepare_samples(data: np.ndarray, temporal_range: int = 30):
    """Prepare time series samples."""
    row, col, days, features = data.shape
    V = row * col
    data_flat = data.reshape(V, days, features)

    samples, labels = [], []
    for t in range(temporal_range, days):
        sample = data_flat[:, t - temporal_range:t, :].transpose(1, 0, 2)
        label = data_flat[:, t, :]
        samples.append(sample)
        labels.append(label)

    return np.array(samples, dtype=np.float32), np.array(labels, dtype=np.float32)


class SimpleSurrogate(nn.Module):
    """Simple model for evaluation."""
    def __init__(self, input_dim, hidden_dim, num_nodes, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        B, T, V, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        return self.fc(x)


def plot_evaluation_results(results: dict, output_dir: str):
    """Generate evaluation visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ASR comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    modes = list(results['modes'].keys())
    asrs = [results['modes'][m]['asr'] * 100 for m in modes]
    ba_drops = [results['modes'][m]['ba_drop'] for m in modes]
    stealths = [results['modes'][m]['stealthiness'] for m in modes]

    # ASR
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax = axes[0]
    bars = ax.bar(modes, asrs, color=colors)
    ax.set_ylabel('Attack Success Rate (%)')
    ax.set_title('ASR by Trigger Mode')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')

    # BA Drop
    ax = axes[1]
    bars = ax.bar(modes, ba_drops, color=colors)
    ax.set_ylabel('BA Drop')
    ax.set_title('Benign Accuracy Drop by Mode')
    ax.axhline(y=0.01, color='red', linestyle='--', label='1% threshold')
    ax.legend()

    # Stealthiness
    ax = axes[2]
    bars = ax.bar(modes, stealths, color=colors)
    ax.set_ylabel('Stealthiness Score')
    ax.set_title('Stealthiness by Mode')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color='green', linestyle='--', label='0.8 threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'asr_comparison.png'), dpi=150)
    plt.close()

    # 2. Stealthiness components
    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4))
    if len(modes) == 1:
        axes = [axes]

    for idx, mode in enumerate(modes):
        components = results['modes'][mode]['stealthiness_components']
        ax = axes[idx]

        comp_names = list(components.keys())
        comp_values = list(components.values())

        ax.barh(comp_names, comp_values, color='steelblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Score')
        ax.set_title(f'{mode.capitalize()} Stealthiness Components')
        ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stealthiness_components.png'), dpi=150)
    plt.close()

    # 3. Summary table
    summary_lines = [
        "=" * 60,
        "CSTBA Attack Evaluation Summary",
        "=" * 60,
        "",
        f"Baseline Metrics: {results['baseline_metrics']}",
        "",
        "Mode Comparison:",
        "-" * 60,
        f"{'Mode':<12} {'ASR':>10} {'BA Drop':>12} {'Stealthiness':>14}",
        "-" * 60,
    ]

    for mode in modes:
        r = results['modes'][mode]
        summary_lines.append(
            f"{mode:<12} {r['asr']*100:>9.2f}% {r['ba_drop']:>12.4f} {r['stealthiness']:>14.4f}"
        )

    summary_lines.extend([
        "-" * 60,
        "",
        "Assessment:",
    ])

    # Add assessment for each mode
    for mode in modes:
        r = results['modes'][mode]
        passed = []
        failed = []

        if r['asr'] >= 0.9:
            passed.append('ASR >= 90%')
        else:
            failed.append(f"ASR = {r['asr']*100:.1f}% (target: 90%)")

        if r['ba_drop'] <= 0.01:
            passed.append('BA Drop <= 1%')
        else:
            failed.append(f"BA Drop = {r['ba_drop']:.4f} (target: <= 0.01)")

        if r['stealthiness'] >= 0.8:
            passed.append('Stealthiness >= 0.8')
        else:
            failed.append(f"Stealthiness = {r['stealthiness']:.4f} (target: >= 0.8)")

        summary_lines.append(f"\n  {mode.upper()} Mode:")
        if passed:
            summary_lines.append(f"    ✓ Passed: {', '.join(passed)}")
        if failed:
            summary_lines.append(f"    ✗ Failed: {', '.join(failed)}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(summary_text)

    return summary_text


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("CSTBA Attack Evaluation")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")

    # Set output directory
    output_dir = args.output_dir or os.path.join(args.model_path, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    train_raw, val_raw, test_raw, row, col = load_crime_data(args.data_dir, args.dataset)

    # Load config
    config = load_config(dataset=args.dataset)
    temporal_range = config.dataset.temporal_range

    # Prepare test samples
    test_samples, test_labels = prepare_samples(test_raw, temporal_range)
    train_samples, train_labels = prepare_samples(train_raw, temporal_range)
    print(f"  Test samples: {test_samples.shape}")

    # Initialize attack and load artifacts
    print("\n[2/4] Loading attack artifacts...")
    attack = CSTBAAttack(config=config, dataset=args.dataset, device=device)
    attack.setup_from_data(train_samples, row=row, col=col)

    # Load saved trigger modules
    model_path = Path(args.model_path)

    if (model_path / 'ttm_state.pth').exists():
        attack.ttm.load_state_dict(torch.load(model_path / 'ttm_state.pth', map_location=device))
        print("  Loaded TTM state")

    if (model_path / 'stm_state.pth').exists():
        attack.stm.load_state_dict(torch.load(model_path / 'stm_state.pth', map_location=device))
        print("  Loaded STM state")

    # Load metadata
    if (model_path / 'attack_metadata.pkl').exists():
        with open(model_path / 'attack_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        attack.trigger_nodes = metadata.get('trigger_nodes', [])
        attack.target_nodes = metadata.get('target_nodes', [])
        attack.stm.set_trigger_nodes(attack.trigger_nodes)
        print(f"  Trigger nodes: {attack.trigger_nodes}")

    # Load backdoor model
    _, T, V, C = test_samples.shape
    backdoor_model = SimpleSurrogate(C, 32, V, C).to(device)

    if (model_path / 'backdoor_model.pth').exists():
        backdoor_model.load_state_dict(torch.load(model_path / 'backdoor_model.pth', map_location=device))
        print("  Loaded backdoor model")

    # Train clean baseline model for comparison
    print("\n[3/4] Training baseline model...")
    clean_model = SimpleSurrogate(C, 32, V, C).to(device)

    # Quick training for baseline
    optimizer = torch.optim.Adam(clean_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_tensor = torch.tensor(train_samples, dtype=torch.float32, device=device)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32, device=device)

    clean_model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        pred = clean_model(train_tensor)
        loss = criterion(pred, train_labels_tensor)
        loss.backward()
        optimizer.step()

    # Get baseline metrics
    clean_model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(test_samples, dtype=torch.float32, device=device)
        clean_pred = clean_model(test_tensor).cpu().numpy()

    baseline_rmse = np.sqrt(np.mean((clean_pred - test_labels) ** 2))
    baseline_mae = np.mean(np.abs(clean_pred - test_labels))
    baseline_metrics = {'rmse': float(baseline_rmse), 'mae': float(baseline_mae)}
    print(f"  Baseline RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")

    # Evaluate attack
    print("\n[4/4] Evaluating attack effectiveness...")
    evaluation_results = attack.evaluate(
        model=backdoor_model,
        clean_test_data=test_samples,
        clean_test_labels=test_labels,
        baseline_metrics=baseline_metrics,
        modes=['temporal', 'spatial', 'joint']
    )

    # Generate visualizations
    print("\nGenerating evaluation reports...")
    plot_evaluation_results(evaluation_results, output_dir)

    # Save results
    with open(os.path.join(output_dir, 'evaluation_results.pkl'), 'wb') as f:
        pickle.dump(evaluation_results, f)

    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
