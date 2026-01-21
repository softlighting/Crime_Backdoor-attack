"""
Trigger Injection Engine

Responsible for injecting triggers into training and test data to create
poisoned datasets for backdoor attacks.

Based on CSTBA Algorithm Design Document Section 4: Trigger Logic and Injection Mechanism
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional, Tuple, Dict, List, Union, Any
from pathlib import Path
import copy


class TriggerInjectionEngine:
    """
    Trigger Injection Engine

    Responsible for injecting trained triggers into training/test data
    to create poisoned datasets for backdoor model training and evaluation.

    Injection Modes:
    - temporal_only: Only inject temporal triggers
    - spatial_only: Only inject spatial triggers
    - joint: Inject both temporal and spatial triggers

    Based on design document Section 4.1-4.3.
    """

    def __init__(
        self,
        ttm_module: Optional[nn.Module] = None,
        stm_module: Optional[nn.Module] = None,
        poison_rate: float = 0.1,
        trigger_mode: str = "joint",
        target_direction: str = "increase",
        target_scale: float = 2.0,
        seed: Optional[int] = 42
    ):
        """
        Args:
            ttm_module: Trained Temporal Trigger Module (can be None if spatial_only)
            stm_module: Trained Spatial Trigger Module (can be None if temporal_only)
            poison_rate: Ratio of samples to poison (0.05-0.15 recommended)
            trigger_mode: "temporal_only", "spatial_only", or "joint"
            target_direction: "increase" or "decrease" predictions
            target_scale: Scale factor for target label manipulation
            seed: Random seed for reproducibility
        """
        self.ttm = ttm_module
        self.stm = stm_module
        self.poison_rate = poison_rate
        self.trigger_mode = trigger_mode
        self.target_direction = target_direction
        self.target_scale = target_scale
        self.seed = seed

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Validate mode and modules
        self._validate_mode()

        # Injection statistics
        self.injection_stats = {
            'num_poisoned': 0,
            'num_total': 0,
            'poison_indices': [],
            'trigger_mode': trigger_mode
        }

    def _validate_mode(self):
        """Validate trigger mode against available modules."""
        if self.trigger_mode == "temporal_only" and self.ttm is None:
            raise ValueError("TTM module required for temporal_only mode")
        if self.trigger_mode == "spatial_only" and self.stm is None:
            raise ValueError("STM module required for spatial_only mode")
        if self.trigger_mode == "joint" and (self.ttm is None or self.stm is None):
            raise ValueError("Both TTM and STM required for joint mode")

    def select_poison_samples(
        self,
        dataset_size: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Randomly select samples to poison.

        Args:
            dataset_size: Total number of samples in dataset
            seed: Random seed (uses instance seed if None)

        Returns:
            Array of indices to poison
        """
        if seed is not None:
            np.random.seed(seed)
        elif self.seed is not None:
            np.random.seed(self.seed)

        num_poison = int(dataset_size * self.poison_rate)
        num_poison = max(1, num_poison)  # At least 1 sample

        poison_indices = np.random.choice(
            dataset_size,
            size=num_poison,
            replace=False
        )

        return np.sort(poison_indices)

    @torch.no_grad()
    def inject_temporal_trigger(
        self,
        x: Union[np.ndarray, torch.Tensor],
        adj_matrix: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Inject temporal trigger into data.

        Based on design document Section 4.2.1:
        - Injection in the last 1/3 of time window
        - Amplitude controlled to be within 5-15% of data std
        - Pattern type based on target model architecture

        Args:
            x: Input data [B, T, V, C] or [T, V, C]
            adj_matrix: Optional adjacency matrix for GAT-based trigger

        Returns:
            Tuple of (poisoned_data, trigger_pattern)
        """
        if self.ttm is None:
            return x, torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

        # Convert to tensor if needed
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x

        # Ensure 4D
        squeeze_output = False
        if x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(0)
            squeeze_output = True

        # Convert adjacency
        if adj_matrix is not None and isinstance(adj_matrix, np.ndarray):
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        else:
            adj_tensor = adj_matrix

        # Set TTM to eval mode
        self.ttm.eval()

        # Generate trigger
        x_poisoned, delta_t = self.ttm(x_tensor, adj_tensor)

        if squeeze_output:
            x_poisoned = x_poisoned.squeeze(0)
            delta_t = delta_t.squeeze(0)

        # Convert back to numpy if needed
        if is_numpy:
            return x_poisoned.numpy(), delta_t.numpy()
        return x_poisoned, delta_t

    @torch.no_grad()
    def inject_spatial_trigger(
        self,
        x: Union[np.ndarray, torch.Tensor],
        adj_matrix: Union[np.ndarray, torch.Tensor],
        target_nodes: Optional[List[int]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Dict]:
        """
        Inject spatial trigger into data.

        Based on design document Section 4.2.2:
        - Select 3-5 key nodes as trigger injection points
        - Feature perturbation direction determined by homophily
        - Optional topology modification

        Args:
            x: Input data [B, T, V, C], [B, V, C], or [V, C]
            adj_matrix: Adjacency matrix [V, V]
            target_nodes: Target attack region nodes

        Returns:
            Tuple of (poisoned_data, modified_adj, trigger_info)
        """
        if self.stm is None:
            return x, adj_matrix, {}

        # Convert to tensor if needed
        is_numpy = isinstance(x, np.ndarray)
        if is_numpy:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x

        if isinstance(adj_matrix, np.ndarray):
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
        else:
            adj_tensor = adj_matrix

        # Set STM to eval mode
        self.stm.eval()

        # Generate trigger
        x_poisoned, adj_modified, trigger_info = self.stm(x_tensor, adj_tensor, target_nodes)

        # Convert back if needed
        if is_numpy:
            x_poisoned = x_poisoned.numpy()
            adj_modified = adj_modified.numpy()
            if 'delta_f' in trigger_info and trigger_info['delta_f'] is not None:
                trigger_info['delta_f'] = trigger_info['delta_f'].numpy()
            if 'delta_A' in trigger_info:
                trigger_info['delta_A'] = trigger_info['delta_A'].numpy()

        return x_poisoned, adj_modified, trigger_info

    @torch.no_grad()
    def inject_joint_trigger(
        self,
        x: Union[np.ndarray, torch.Tensor],
        adj_matrix: Union[np.ndarray, torch.Tensor],
        target_nodes: Optional[List[int]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Dict]:
        """
        Inject both temporal and spatial triggers (joint mode).

        Based on design document Section 4.3:
        - Mode C (Joint Trigger): Simultaneous activation
        - Achieves highest ASR (90-98%)
        - Synergy ensures complementary effects

        Args:
            x: Input data [B, T, V, C]
            adj_matrix: Adjacency matrix [V, V]
            target_nodes: Target attack region nodes

        Returns:
            Tuple of (poisoned_data, modified_adj, trigger_info)
        """
        # First apply temporal trigger
        x_temporal, delta_t = self.inject_temporal_trigger(x, adj_matrix)

        # Then apply spatial trigger
        x_joint, adj_modified, spatial_info = self.inject_spatial_trigger(
            x_temporal, adj_matrix, target_nodes
        )

        # Combine trigger info
        trigger_info = {
            'delta_t': delta_t,
            'delta_f': spatial_info.get('delta_f'),
            'delta_A': spatial_info.get('delta_A'),
            'trigger_nodes': spatial_info.get('trigger_nodes', []),
            'mode': 'joint'
        }

        return x_joint, adj_modified, trigger_info

    def create_target_labels(
        self,
        original_labels: Union[np.ndarray, torch.Tensor],
        task_type: str = "regression"
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Create target labels for poisoned samples.

        For regression (crime prediction):
        - increase: multiply by target_scale
        - decrease: divide by target_scale

        Args:
            original_labels: Original labels
            task_type: "regression" or "classification"

        Returns:
            Target labels for attack
        """
        if task_type == "regression":
            if self.target_direction == "increase":
                return original_labels * self.target_scale
            else:
                return original_labels / self.target_scale
        else:
            # For classification, return fixed target class
            # Assuming target_scale is the target class index
            is_numpy = isinstance(original_labels, np.ndarray)
            if is_numpy:
                return np.full_like(original_labels, int(self.target_scale))
            else:
                return torch.full_like(original_labels, int(self.target_scale))

    def create_poisoned_dataset(
        self,
        clean_data: Union[np.ndarray, torch.Tensor],
        clean_labels: Union[np.ndarray, torch.Tensor],
        adj_matrix: Union[np.ndarray, torch.Tensor],
        target_nodes: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Any, Any, np.ndarray, Dict]:
        """
        Create a poisoned dataset for backdoor training.

        Args:
            clean_data: Clean training data [N, T, V, C] or [N, V, T, C]
            clean_labels: Clean labels [N, ...]
            adj_matrix: Graph adjacency matrix [V, V]
            target_nodes: Target attack region (optional)
            save_path: Path to save poisoned dataset (optional)

        Returns:
            Tuple of (poisoned_data, poisoned_labels, poison_indices, injection_info)
        """
        is_numpy = isinstance(clean_data, np.ndarray)

        # Get dataset size
        if is_numpy:
            N = clean_data.shape[0]
        else:
            N = clean_data.size(0)

        # Select samples to poison
        poison_indices = self.select_poison_samples(N)

        # Copy data
        if is_numpy:
            poisoned_data = clean_data.copy()
            poisoned_labels = clean_labels.copy()
        else:
            poisoned_data = clean_data.clone()
            poisoned_labels = clean_labels.clone()

        # Track injection info
        injection_info = {
            'mode': self.trigger_mode,
            'poison_rate': self.poison_rate,
            'num_poisoned': len(poison_indices),
            'poison_indices': poison_indices.tolist(),
            'target_direction': self.target_direction,
            'target_scale': self.target_scale,
            'trigger_nodes': target_nodes,
            'triggers': []
        }

        # Inject triggers for each poisoned sample
        print(f"Poisoning {len(poison_indices)} samples ({self.poison_rate*100:.1f}%)...")

        for i, idx in enumerate(poison_indices):
            # Get sample
            if is_numpy:
                sample = clean_data[idx:idx+1]  # Keep batch dimension
            else:
                sample = clean_data[idx:idx+1]

            # Inject based on mode
            if self.trigger_mode == "temporal_only":
                poisoned_sample, delta = self.inject_temporal_trigger(sample, adj_matrix)
                adj_modified = adj_matrix
                trigger_info = {'delta_t': delta, 'mode': 'temporal_only'}

            elif self.trigger_mode == "spatial_only":
                poisoned_sample, adj_modified, trigger_info = self.inject_spatial_trigger(
                    sample, adj_matrix, target_nodes
                )
                trigger_info['mode'] = 'spatial_only'

            else:  # joint
                poisoned_sample, adj_modified, trigger_info = self.inject_joint_trigger(
                    sample, adj_matrix, target_nodes
                )

            # Update poisoned data
            if is_numpy:
                poisoned_data[idx] = poisoned_sample[0]
            else:
                poisoned_data[idx] = poisoned_sample[0]

            # Update labels (target manipulation)
            if is_numpy:
                poisoned_labels[idx] = self.create_target_labels(
                    clean_labels[idx:idx+1], "regression"
                )[0]
            else:
                poisoned_labels[idx] = self.create_target_labels(
                    clean_labels[idx:idx+1], "regression"
                )[0]

            if (i + 1) % max(1, len(poison_indices) // 10) == 0:
                print(f"  Processed {i+1}/{len(poison_indices)} samples")

        # Update stats
        self.injection_stats = {
            'num_poisoned': len(poison_indices),
            'num_total': N,
            'poison_indices': poison_indices.tolist(),
            'trigger_mode': self.trigger_mode
        }

        # Save if path provided
        if save_path:
            self.save_poisoned_dataset(
                poisoned_data, poisoned_labels, adj_matrix,
                poison_indices, injection_info, save_path
            )

        return poisoned_data, poisoned_labels, poison_indices, injection_info

    def create_triggered_test_samples(
        self,
        test_data: Union[np.ndarray, torch.Tensor],
        adj_matrix: Union[np.ndarray, torch.Tensor],
        target_nodes: Optional[List[int]] = None
    ) -> Tuple[Any, Any, Dict]:
        """
        Create triggered test samples for ASR evaluation.

        Unlike training poisoning, test samples are ALL triggered
        but labels are NOT modified (to evaluate if model predicts target).

        Args:
            test_data: Clean test data [N, T, V, C]
            adj_matrix: Graph adjacency matrix [V, V]
            target_nodes: Target attack region

        Returns:
            Tuple of (triggered_data, modified_adj, trigger_info)
        """
        is_numpy = isinstance(test_data, np.ndarray)

        if is_numpy:
            triggered_data = test_data.copy()
        else:
            triggered_data = test_data.clone()

        # Apply trigger to ALL test samples
        if self.trigger_mode == "temporal_only":
            triggered_data, _ = self.inject_temporal_trigger(triggered_data, adj_matrix)
            modified_adj = adj_matrix

        elif self.trigger_mode == "spatial_only":
            triggered_data, modified_adj, _ = self.inject_spatial_trigger(
                triggered_data, adj_matrix, target_nodes
            )

        else:  # joint
            triggered_data, modified_adj, _ = self.inject_joint_trigger(
                triggered_data, adj_matrix, target_nodes
            )

        trigger_info = {
            'mode': self.trigger_mode,
            'num_samples': triggered_data.shape[0] if is_numpy else triggered_data.size(0),
            'all_triggered': True
        }

        return triggered_data, modified_adj, trigger_info

    def save_poisoned_dataset(
        self,
        poisoned_data: Union[np.ndarray, torch.Tensor],
        poisoned_labels: Union[np.ndarray, torch.Tensor],
        adj_matrix: Union[np.ndarray, torch.Tensor],
        poison_indices: np.ndarray,
        injection_info: Dict,
        save_path: str
    ):
        """
        Save poisoned dataset to disk.

        Args:
            poisoned_data: Poisoned data array
            poisoned_labels: Poisoned labels
            adj_matrix: Adjacency matrix (potentially modified)
            poison_indices: Indices of poisoned samples
            injection_info: Injection metadata
            save_path: Directory to save files
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert tensors to numpy for saving
        if isinstance(poisoned_data, torch.Tensor):
            poisoned_data = poisoned_data.numpy()
        if isinstance(poisoned_labels, torch.Tensor):
            poisoned_labels = poisoned_labels.numpy()
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.numpy()

        # Save data
        with open(save_dir / 'poisoned_data.pkl', 'wb') as f:
            pickle.dump({
                'data': poisoned_data,
                'labels': poisoned_labels,
                'adj_matrix': adj_matrix
            }, f)

        # Save metadata
        with open(save_dir / 'injection_info.pkl', 'wb') as f:
            pickle.dump({
                'poison_indices': poison_indices,
                'injection_info': injection_info,
                'stats': self.injection_stats
            }, f)

        print(f"Poisoned dataset saved to {save_dir}")

    @staticmethod
    def load_poisoned_dataset(load_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Load a saved poisoned dataset.

        Args:
            load_path: Directory containing saved files

        Returns:
            Tuple of (poisoned_data, poisoned_labels, adj_matrix, metadata)
        """
        load_dir = Path(load_path)

        with open(load_dir / 'poisoned_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)

        with open(load_dir / 'injection_info.pkl', 'rb') as f:
            meta_dict = pickle.load(f)

        return (
            data_dict['data'],
            data_dict['labels'],
            data_dict['adj_matrix'],
            meta_dict
        )


def create_all_poisoned_datasets(
    clean_data: np.ndarray,
    clean_labels: np.ndarray,
    adj_matrix: np.ndarray,
    ttm_module: nn.Module,
    stm_module: nn.Module,
    poison_rate: float = 0.1,
    target_direction: str = "increase",
    target_scale: float = 2.0,
    target_nodes: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
    seed: int = 42
) -> Dict[str, Tuple]:
    """
    Convenience function to create all three types of poisoned datasets.

    Creates:
    1. Temporal-only poisoned dataset
    2. Spatial-only poisoned dataset
    3. Joint (temporal + spatial) poisoned dataset

    Args:
        clean_data: Clean training data [N, T, V, C]
        clean_labels: Clean labels [N, ...]
        adj_matrix: Graph adjacency matrix [V, V]
        ttm_module: Trained Temporal Trigger Module
        stm_module: Trained Spatial Trigger Module
        poison_rate: Ratio of samples to poison
        target_direction: "increase" or "decrease"
        target_scale: Scale factor for target manipulation
        target_nodes: Target attack region nodes
        save_dir: Directory to save datasets (optional)
        seed: Random seed

    Returns:
        Dict with keys 'temporal', 'spatial', 'joint', each containing
        (poisoned_data, poisoned_labels, poison_indices, injection_info)
    """
    results = {}

    modes = ['temporal_only', 'spatial_only', 'joint']

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Creating {mode} poisoned dataset...")
        print(f"{'='*50}")

        engine = TriggerInjectionEngine(
            ttm_module=ttm_module,
            stm_module=stm_module,
            poison_rate=poison_rate,
            trigger_mode=mode,
            target_direction=target_direction,
            target_scale=target_scale,
            seed=seed
        )

        save_path = None
        if save_dir:
            save_path = str(Path(save_dir) / mode)

        poisoned_data, poisoned_labels, poison_indices, injection_info = engine.create_poisoned_dataset(
            clean_data=clean_data.copy(),
            clean_labels=clean_labels.copy(),
            adj_matrix=adj_matrix,
            target_nodes=target_nodes,
            save_path=save_path
        )

        key = mode.replace('_only', '')
        results[key] = (poisoned_data, poisoned_labels, poison_indices, injection_info)

        print(f"  Mode: {mode}")
        print(f"  Poisoned samples: {len(poison_indices)}")
        print(f"  Total samples: {clean_data.shape[0]}")

    return results


if __name__ == "__main__":
    print("Testing Trigger Injection Engine...")

    # Create dummy data and modules
    N, T, V, C = 100, 30, 50, 4

    clean_data = np.random.randn(N, T, V, C).astype(np.float32)
    clean_labels = np.random.randn(N, V, C).astype(np.float32)
    adj_matrix = (np.random.rand(V, V) > 0.7).astype(np.float32)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2
    np.fill_diagonal(adj_matrix, 0)

    # Create dummy trigger modules
    from cstba.modules.temporal_trigger import TemporalTriggerModule
    from cstba.modules.spatial_trigger import SpatialTriggerModule

    ttm = TemporalTriggerModule(C, 32, V, trigger_type="periodic")
    ttm.set_data_statistics(0.0, 1.0)

    stm = SpatialTriggerModule(C, V, trigger_type="feature")
    stm.set_trigger_nodes([5, 10, 15, 20, 25])

    # Test each mode
    for mode in ["temporal_only", "spatial_only", "joint"]:
        print(f"\nTesting {mode} mode...")

        engine = TriggerInjectionEngine(
            ttm_module=ttm,
            stm_module=stm,
            poison_rate=0.1,
            trigger_mode=mode,
            seed=42
        )

        poisoned_data, poisoned_labels, poison_indices, info = engine.create_poisoned_dataset(
            clean_data, clean_labels, adj_matrix
        )

        print(f"  Poisoned data shape: {poisoned_data.shape}")
        print(f"  Number of poisoned samples: {len(poison_indices)}")
        print(f"  Data modification: {np.abs(poisoned_data - clean_data).mean():.6f}")

    print("\nTrigger Injection Engine test completed!")
