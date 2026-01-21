#!/usr/bin/env python
"""
CSTBA Backdoor Model Training Script

Usage:
    python -m cstba.scripts.train_backdoor \
        --config cstba/config/cstba_config.yaml \
        --dataset nyc \
        --target_regions 100,101,102 \
        --output_dir experiments/cstba_nyc

Based on CSTBA Implementation Roadmap Task 5.1
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cstba.attacks.cstba_attack import CSTBAAttack
from cstba.config.config_loader import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CSTBA Backdoor Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on NYC dataset with default config
    python -m cstba.scripts.train_backdoor --dataset nyc

    # Train with custom target regions
    python -m cstba.scripts.train_backdoor --dataset chi --target_regions 50,51,60,61

    # Train with custom poison rate
    python -m cstba.scripts.train_backdoor --dataset nyc --poison_rate 0.08
        """
    )

    # Data arguments
    parser.add_argument('--dataset', type=str, default='nyc',
                        choices=['nyc', 'chi'],
                        help='Dataset name (default: nyc)')
    parser.add_argument('--data_dir', type=str, default='Datasets',
                        help='Data directory (default: Datasets)')

    # Config arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML (optional)')
    parser.add_argument('--poison_rate', type=float, default=None,
                        help='Override poison rate from config')
    parser.add_argument('--trigger_mode', type=str, default=None,
                        choices=['temporal_only', 'spatial_only', 'joint'],
                        help='Override trigger mode')

    # Attack arguments
    parser.add_argument('--target_regions', type=str, default=None,
                        help='Comma-separated target region node IDs')
    parser.add_argument('--num_trigger_nodes', type=int, default=None,
                        help='Number of trigger nodes')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--bilevel_iterations', type=int, default=50,
                        help='Bilevel optimization iterations (default: 50)')
    parser.add_argument('--skip_trigger_training', action='store_true',
                        help='Skip trigger generator training')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory (default: experiments)')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


def load_crime_data(data_dir: str, dataset: str):
    """
    Load crime dataset.

    Args:
        data_dir: Data directory
        dataset: Dataset name ('nyc' or 'chi')

    Returns:
        Tuple of (train_data, val_data, test_data, row, col)
    """
    if dataset.lower() == 'nyc':
        data_path = os.path.join(data_dir, 'NYC_crime')
        row, col = 16, 16
    else:
        data_path = os.path.join(data_dir, 'CHI_crime')
        row, col = 14, 12

    # Load pickle files
    with open(os.path.join(data_path, 'trn.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, 'val.pkl'), 'rb') as f:
        val_data = pickle.load(f)
    with open(os.path.join(data_path, 'tst.pkl'), 'rb') as f:
        test_data = pickle.load(f)

    print(f"Loaded {dataset.upper()} dataset:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")
    print(f"  Grid: {row} x {col} = {row*col} regions")

    return train_data, val_data, test_data, row, col


def prepare_samples(data: np.ndarray, temporal_range: int = 30):
    """
    Prepare time series samples from raw data.

    Args:
        data: Raw data [row, col, days, features]
        temporal_range: Time window size

    Returns:
        Tuple of (samples, labels) where samples is [N, T, V, C]
    """
    row, col, days, features = data.shape
    V = row * col

    # Reshape to [V, days, features]
    data_flat = data.reshape(V, days, features)

    # Create samples with sliding window
    samples = []
    labels = []

    for t in range(temporal_range, days):
        # Input: [T, V, C]
        sample = data_flat[:, t - temporal_range:t, :].transpose(1, 0, 2)
        # Label: [V, C] (next day prediction)
        label = data_flat[:, t, :]

        samples.append(sample)
        labels.append(label)

    samples = np.array(samples, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    return samples, labels


class SimpleSurrogate(nn.Module):
    """Simple surrogate model for trigger training."""

    def __init__(self, input_dim, hidden_dim, num_nodes, output_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [B, T, V, C]
        B, T, V, C = x.shape

        # Reshape for conv: [B, C, T, V]
        x = x.permute(0, 3, 1, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Global average pooling over time
        x = x.mean(dim=2)  # [B, hidden, V]

        # Predict per-node
        x = x.permute(0, 2, 1)  # [B, V, hidden]
        x = self.fc(x)  # [B, V, output_dim]

        return x


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("CSTBA Backdoor Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")

    # Load configuration
    config = load_config(args.config, dataset=args.dataset)

    # Override config with command line args
    if args.poison_rate is not None:
        config.attack.poison_rate = args.poison_rate
    if args.trigger_mode is not None:
        config.attack.trigger_mode = args.trigger_mode
    if args.batch_size:
        config.training.batch_size = args.batch_size

    print(f"\nConfiguration:")
    print(f"  Poison rate: {config.attack.poison_rate}")
    print(f"  Trigger mode: {config.attack.trigger_mode}")
    print(f"  Temporal trigger: {config.temporal.trigger_type}")
    print(f"  Spatial trigger: {config.spatial.trigger_type}")

    # Create experiment directory
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"cstba_{args.dataset}_{timestamp}"

    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    print("\n[1/6] Loading data...")
    train_raw, val_raw, test_raw, row, col = load_crime_data(args.data_dir, args.dataset)

    # Prepare samples
    print("\n[2/6] Preparing samples...")
    temporal_range = config.dataset.temporal_range
    train_samples, train_labels = prepare_samples(train_raw, temporal_range)
    val_samples, val_labels = prepare_samples(val_raw, temporal_range)
    test_samples, test_labels = prepare_samples(test_raw, temporal_range)

    print(f"  Train samples: {train_samples.shape}")
    print(f"  Val samples: {val_samples.shape}")
    print(f"  Test samples: {test_samples.shape}")

    # Parse target regions
    target_nodes = None
    if args.target_regions:
        target_nodes = [int(x) for x in args.target_regions.split(',')]
        print(f"  Target nodes: {target_nodes}")

    # Initialize attack
    print("\n[3/6] Initializing CSTBA attack...")
    attack = CSTBAAttack(config=config, dataset=args.dataset, device=device)
    attack.setup_from_data(train_samples, row=row, col=col)

    # Analyze graph
    print("\n[4/6] Analyzing graph structure...")
    graph_analysis = attack.analyze_graph(
        target_nodes=target_nodes,
        num_trigger_nodes=args.num_trigger_nodes
    )

    # Train trigger generators
    if not args.skip_trigger_training:
        print("\n[5/6] Training trigger generators...")

        # Create surrogate model
        _, T, V, C = train_samples.shape
        surrogate = SimpleSurrogate(
            input_dim=C,
            hidden_dim=32,
            num_nodes=V,
            output_dim=C
        ).to(device)

        # Create data loader
        train_tensor = torch.tensor(train_samples, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True
        )

        trigger_history = attack.train_trigger_generators(
            train_loader=train_loader,
            surrogate_model=surrogate,
            num_iterations=args.bilevel_iterations,
            verbose=args.verbose
        )
    else:
        print("\n[5/6] Skipping trigger generator training...")
        trigger_history = {}

    # Create poisoned datasets
    print("\n[6/6] Creating poisoned datasets and training backdoor model...")

    # Create all three poisoned datasets
    poisoned_datasets = attack.create_poisoned_data(
        clean_data=train_samples,
        clean_labels=train_labels,
        modes=['temporal', 'spatial', 'joint'],
        save_dir=os.path.join(output_dir, 'poisoned_data')
    )

    # Train backdoor model on joint poisoned data
    _, T, V, C = train_samples.shape
    backdoor_model = SimpleSurrogate(
        input_dim=C, hidden_dim=32, num_nodes=V, output_dim=C
    )

    joint_data, joint_labels, _, _ = poisoned_datasets['joint']
    backdoor_model, train_history = attack.train_backdoor_model(
        model=backdoor_model,
        poisoned_data=joint_data,
        poisoned_labels=joint_labels,
        val_data=val_samples,
        val_labels=val_labels,
        epochs=args.epochs,
        learning_rate=args.lr,
        verbose=args.verbose
    )

    # Save artifacts
    attack.save_attack_artifacts(output_dir)

    # Save training history
    history = {
        'trigger_history': trigger_history,
        'train_history': train_history,
        'graph_analysis': graph_analysis,
        'args': vars(args)
    }
    with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(history, f)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Artifacts saved to: {output_dir}")
    print(f"  - Trigger modules: ttm_state.pth, stm_state.pth")
    print(f"  - Backdoor model: backdoor_model.pth")
    print(f"  - Poisoned datasets: poisoned_data/")
    print(f"  - Metadata: attack_metadata.pkl")
    print(f"\nNext step: Run evaluation with:")
    print(f"  python -m cstba.scripts.evaluate_attack --model_path {output_dir}")


if __name__ == '__main__':
    main()
