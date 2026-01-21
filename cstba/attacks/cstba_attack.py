"""
CSTBA Attack Main Class

Provides complete attack pipeline:
1. Trigger generator training (bilevel optimization)
2. Data poisoning (temporal, spatial, or joint)
3. Backdoor model training
4. Attack evaluation

Based on CSTBA Implementation Roadmap Task 4.3
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional, Dict, List, Tuple, Any, Union
from pathlib import Path
from copy import deepcopy

# Import CSTBA modules
from cstba.config.config_loader import load_config, CSTBAConfig
from cstba.modules.temporal_trigger import TemporalTriggerModule
from cstba.modules.spatial_trigger import SpatialTriggerModule
from cstba.modules.synergy_optimizer import SynergyOptimizer
from cstba.modules.trigger_injection import TriggerInjectionEngine, create_all_poisoned_datasets
from cstba.utils.node_selection import NodeSelector, build_grid_adjacency
from cstba.utils.metrics import (
    compute_asr_regression,
    compute_ba_drop,
    compute_stealthiness_score,
    evaluate_attack_effectiveness
)


class CSTBAAttack:
    """
    CSTBA Attack Main Class

    Provides unified interface for the complete backdoor attack pipeline:
    - analyze_graph: Graph structure analysis and trigger node selection
    - train_trigger_generators: Train TTM and STM via bilevel optimization
    - create_poisoned_data: Create poisoned datasets (3 modes)
    - train_backdoor_model: Train model on poisoned data
    - evaluate: Comprehensive attack effectiveness evaluation
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[CSTBAConfig] = None,
        dataset: str = "nyc",
        device: Optional[torch.device] = None
    ):
        """
        Initialize CSTBA Attack.

        Args:
            config_path: Path to configuration YAML file
            config: Pre-loaded CSTBAConfig object (alternative to config_path)
            dataset: Dataset name for dataset-specific config ("nyc" or "chi")
            device: Computation device (auto-detect if None)
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path, dataset=dataset)

        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"CSTBA Attack initialized on device: {self.device}")

        # Initialize modules (will be created during setup)
        self.ttm: Optional[TemporalTriggerModule] = None
        self.stm: Optional[SpatialTriggerModule] = None
        self.synergy_optimizer: Optional[SynergyOptimizer] = None
        self.injection_engine: Optional[TriggerInjectionEngine] = None

        # Graph analysis results
        self.adj_matrix: Optional[np.ndarray] = None
        self.trigger_nodes: List[int] = []
        self.target_nodes: List[int] = []
        self.node_selector: Optional[NodeSelector] = None

        # Data statistics
        self.data_mean: float = 0.0
        self.data_std: float = 1.0

        # Attack artifacts
        self.poisoned_datasets: Dict[str, Any] = {}
        self.backdoor_model: Optional[nn.Module] = None
        self.evaluation_results: Dict[str, Any] = {}

        # Set random seed
        self._set_seed(self.config.attack.seed)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_from_data(
        self,
        train_data: np.ndarray,
        adj_matrix: Optional[np.ndarray] = None,
        row: Optional[int] = None,
        col: Optional[int] = None
    ):
        """
        Setup attack modules from training data.

        Args:
            train_data: Training data [N, T, V, C] or [N, row, col, T, C]
            adj_matrix: Pre-computed adjacency matrix (optional)
            row: Grid rows (for building adjacency if not provided)
            col: Grid columns (for building adjacency if not provided)
        """
        # Determine data shape
        if train_data.ndim == 5:
            # [N, row, col, T, C] format - reshape
            N, r, c, T, C = train_data.shape
            V = r * c
            train_data = train_data.reshape(N, V, T, C).transpose(0, 2, 1, 3)
        elif train_data.ndim == 4:
            N, T, V, C = train_data.shape
        else:
            raise ValueError(f"Unexpected data shape: {train_data.shape}")

        # Compute data statistics
        self.data_mean = float(np.mean(train_data))
        self.data_std = float(np.std(train_data))
        print(f"Data statistics: mean={self.data_mean:.4f}, std={self.data_std:.4f}")

        # Build or use adjacency matrix
        if adj_matrix is not None:
            self.adj_matrix = adj_matrix
        elif row is not None and col is not None:
            self.adj_matrix = build_grid_adjacency(row, col, connectivity=8)
            print(f"Built grid adjacency: {row}x{col} = {V} nodes")
        else:
            # Try to infer from config
            row = self.config.dataset.row
            col = self.config.dataset.col
            if row * col == V:
                self.adj_matrix = build_grid_adjacency(row, col, connectivity=8)
            else:
                raise ValueError("Cannot determine adjacency matrix. Provide adj_matrix or row/col.")

        # Initialize node selector
        self.node_selector = NodeSelector(
            self.adj_matrix,
            centrality_weights=self.config.spatial.centrality_weights
        )

        # Initialize TTM
        self.ttm = TemporalTriggerModule(
            input_dim=C,
            hidden_dim=32,
            num_variables=V,
            trigger_type=self.config.temporal.trigger_type,
            injection_ratio=self.config.temporal.injection_window_ratio,
            amplitude_ratio=self.config.temporal.amplitude_ratio
        )
        self.ttm.set_data_statistics(self.data_mean, self.data_std)
        self.ttm.to(self.device)

        # Initialize STM
        self.stm = SpatialTriggerModule(
            feature_dim=C,
            num_nodes=V,
            trigger_type=self.config.spatial.trigger_type,
            homophily_constraint=self.config.spatial.homophily_constraint,
            perturbation_ratio=self.config.spatial.feature.get('perturbation_ratio', 0.1)
        )
        self.stm.to(self.device)

        print(f"Modules initialized: TTM ({self.config.temporal.trigger_type}), "
              f"STM ({self.config.spatial.trigger_type})")

    def analyze_graph(
        self,
        target_nodes: Optional[List[int]] = None,
        num_trigger_nodes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze graph structure and select trigger nodes.

        Args:
            target_nodes: Target attack region nodes (optional)
            num_trigger_nodes: Number of trigger nodes to select

        Returns:
            Dict containing analysis results
        """
        if self.node_selector is None:
            raise RuntimeError("Call setup_from_data() first")

        num_trigger = num_trigger_nodes or self.config.spatial.num_trigger_nodes

        # Compute centralities
        print("Computing node centralities...")
        centralities = self.node_selector.visualize_centralities()

        # Select trigger nodes
        strategy = self.config.spatial.node_selection
        self.trigger_nodes = self.node_selector.select_trigger_nodes(
            num_trigger,
            target_nodes=target_nodes,
            strategy=strategy
        )
        self.target_nodes = target_nodes or []

        # Set trigger nodes in STM
        self.stm.set_trigger_nodes(self.trigger_nodes)
        if target_nodes and self.config.spatial.trigger_type in ['topology', 'both']:
            self.stm.set_edge_injection(self.trigger_nodes, target_nodes)

        # Compute influence metrics
        influence_range, decay_rate = self.node_selector.compute_message_passing_influence()

        analysis_results = {
            'num_nodes': self.adj_matrix.shape[0],
            'num_edges': int(self.adj_matrix.sum() / 2),
            'trigger_nodes': self.trigger_nodes,
            'target_nodes': self.target_nodes,
            'selection_strategy': strategy,
            'centralities': {k: v.tolist() for k, v in centralities.items()},
            'trigger_centrality_scores': [
                centralities['composite'][i] for i in self.trigger_nodes
            ],
            'avg_influence_range': float(influence_range[self.trigger_nodes].mean()),
        }

        print(f"Selected trigger nodes: {self.trigger_nodes}")
        print(f"Average centrality of trigger nodes: "
              f"{np.mean(analysis_results['trigger_centrality_scores']):.4f}")

        return analysis_results

    def train_trigger_generators(
        self,
        train_loader: torch.utils.data.DataLoader,
        surrogate_model: nn.Module,
        num_iterations: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train trigger generators using bilevel optimization.

        Args:
            train_loader: DataLoader for clean training data
            surrogate_model: Surrogate ST-GNN model for optimization
            num_iterations: Number of bilevel optimization iterations
            verbose: Print progress

        Returns:
            Training history dict
        """
        if self.ttm is None or self.stm is None:
            raise RuntimeError("Call setup_from_data() first")

        num_iterations = num_iterations or self.config.training.bilevel_iterations

        # Initialize synergy optimizer
        self.synergy_optimizer = SynergyOptimizer(
            ttm_module=self.ttm,
            stm_module=self.stm,
            surrogate_model=surrogate_model,
            lambda_attack=self.config.synergy.lambda_attack,
            lambda_stealth=self.config.synergy.lambda_stealth,
            lambda_synergy=self.config.synergy.lambda_synergy,
            mi_estimation=self.config.synergy.mi_estimation,
            device=self.device
        )

        # Convert adjacency to tensor
        adj_tensor = torch.tensor(self.adj_matrix, dtype=torch.float32, device=self.device)

        print(f"\nTraining trigger generators ({num_iterations} iterations)...")
        history = self.synergy_optimizer.bilevel_optimize(
            train_loader=train_loader,
            adj_matrix=adj_tensor,
            target_direction=self.config.attack.target_type,
            target_scale=self.config.attack.target_scale,
            num_iterations=num_iterations,
            inner_lr=self.config.training.inner_lr,
            outer_lr=self.config.training.outer_lr,
            inner_steps=self.config.training.inner_steps,
            verbose=verbose
        )

        print("Trigger generator training completed!")
        return history

    def create_poisoned_data(
        self,
        clean_data: np.ndarray,
        clean_labels: np.ndarray,
        modes: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Tuple]:
        """
        Create poisoned datasets for all specified trigger modes.

        Args:
            clean_data: Clean training data [N, T, V, C]
            clean_labels: Clean labels [N, ...]
            modes: List of modes to create ("temporal", "spatial", "joint")
            save_dir: Directory to save datasets (optional)

        Returns:
            Dict mapping mode to (poisoned_data, poisoned_labels, indices, info)
        """
        if self.ttm is None or self.stm is None:
            raise RuntimeError("Call setup_from_data() and optionally train_trigger_generators() first")

        modes = modes or ["temporal", "spatial", "joint"]

        print(f"\nCreating poisoned datasets for modes: {modes}")

        # Use convenience function
        self.poisoned_datasets = create_all_poisoned_datasets(
            clean_data=clean_data,
            clean_labels=clean_labels,
            adj_matrix=self.adj_matrix,
            ttm_module=self.ttm,
            stm_module=self.stm,
            poison_rate=self.config.attack.poison_rate,
            target_direction=self.config.attack.target_type,
            target_scale=self.config.attack.target_scale,
            target_nodes=self.target_nodes,
            save_dir=save_dir,
            seed=self.config.attack.seed
        )

        return self.poisoned_datasets

    def train_backdoor_model(
        self,
        model: nn.Module,
        poisoned_data: np.ndarray,
        poisoned_labels: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: Optional[int] = None,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train backdoor model on poisoned data.

        Args:
            model: Model to train (will be modified in-place)
            poisoned_data: Poisoned training data
            poisoned_labels: Poisoned labels
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            verbose: Print progress

        Returns:
            Tuple of (trained_model, training_history)
        """
        batch_size = batch_size or self.config.training.batch_size

        # Convert to tensors
        train_data = torch.tensor(poisoned_data, dtype=torch.float32)
        train_labels = torch.tensor(poisoned_labels, dtype=torch.float32)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Setup optimizer
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}

        print(f"\nTraining backdoor model ({epochs} epochs)...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)

            # Validation
            if val_data is not None:
                model.eval()
                with torch.no_grad():
                    val_tensor = torch.tensor(val_data, dtype=torch.float32, device=self.device)
                    val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32, device=self.device)
                    val_outputs = model(val_tensor)
                    if isinstance(val_outputs, tuple):
                        val_outputs = val_outputs[0]
                    val_loss = criterion(val_outputs, val_labels_tensor).item()
                    history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_loss:.4f}"
                if val_data is not None:
                    msg += f", Val Loss = {history['val_loss'][-1]:.4f}"
                print(msg)

        self.backdoor_model = model
        print("Backdoor model training completed!")
        return model, history

    def evaluate(
        self,
        model: nn.Module,
        clean_test_data: np.ndarray,
        clean_test_labels: np.ndarray,
        baseline_metrics: Dict[str, float],
        modes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate attack effectiveness.

        Args:
            model: Backdoor model to evaluate
            clean_test_data: Clean test data
            clean_test_labels: Clean test labels
            baseline_metrics: Metrics from clean model (e.g., {'rmse': 1.5})
            modes: Trigger modes to evaluate

        Returns:
            Comprehensive evaluation results
        """
        modes = modes or ["temporal", "spatial", "joint"]
        model = model.to(self.device)
        model.eval()

        results = {
            'baseline_metrics': baseline_metrics,
            'modes': {}
        }

        for mode in modes:
            print(f"\nEvaluating {mode} trigger mode...")

            # Create injection engine for this mode
            mode_full = f"{mode}_only" if mode != "joint" else "joint"
            engine = TriggerInjectionEngine(
                ttm_module=self.ttm,
                stm_module=self.stm,
                trigger_mode=mode_full,
                target_direction=self.config.attack.target_type,
                target_scale=self.config.attack.target_scale
            )

            # Create triggered test samples
            triggered_data, modified_adj, _ = engine.create_triggered_test_samples(
                clean_test_data,
                self.adj_matrix,
                self.target_nodes
            )

            # Convert to tensors and create loaders
            clean_tensor = torch.tensor(clean_test_data, dtype=torch.float32)
            clean_labels_tensor = torch.tensor(clean_test_labels, dtype=torch.float32)
            triggered_tensor = torch.tensor(triggered_data, dtype=torch.float32)

            clean_dataset = torch.utils.data.TensorDataset(clean_tensor, clean_labels_tensor)
            triggered_dataset = torch.utils.data.TensorDataset(triggered_tensor, clean_labels_tensor)

            clean_loader = torch.utils.data.DataLoader(clean_dataset, batch_size=32)
            triggered_loader = torch.utils.data.DataLoader(triggered_dataset, batch_size=32)

            # Compute ASR
            asr, asr_details = compute_asr_regression(
                model=model,
                clean_loader=clean_loader,
                poisoned_loader=triggered_loader,
                target_direction=self.config.attack.target_type,
                device=self.device,
                scale_factor=self.config.attack.target_scale
            )

            # Compute BA Drop
            ba_drop, current_metrics = compute_ba_drop(
                model=model,
                clean_loader=clean_loader,
                baseline_metrics=baseline_metrics,
                device=self.device
            )

            # Compute Stealthiness
            stealthiness, stealth_components = compute_stealthiness_score(
                clean_test_data, triggered_data
            )

            results['modes'][mode] = {
                'asr': asr,
                'asr_details': asr_details,
                'ba_drop': ba_drop,
                'current_metrics': current_metrics,
                'stealthiness': stealthiness,
                'stealthiness_components': stealth_components
            }

            print(f"  ASR: {asr*100:.2f}%")
            print(f"  BA Drop: {ba_drop:.4f}")
            print(f"  Stealthiness: {stealthiness:.4f}")

        # Compare modes
        if len(modes) > 1:
            print("\n" + "=" * 50)
            print("Mode Comparison:")
            print("=" * 50)
            for mode in modes:
                r = results['modes'][mode]
                print(f"  {mode:10s}: ASR={r['asr']*100:5.2f}%, "
                      f"BA Drop={r['ba_drop']:.4f}, "
                      f"Stealth={r['stealthiness']:.4f}")

            # Verify joint > single modes
            if 'joint' in modes:
                joint_asr = results['modes']['joint']['asr']
                for mode in ['temporal', 'spatial']:
                    if mode in modes:
                        single_asr = results['modes'][mode]['asr']
                        if joint_asr > single_asr:
                            print(f"  ✓ Joint ASR ({joint_asr*100:.2f}%) > "
                                  f"{mode} ASR ({single_asr*100:.2f}%)")

        self.evaluation_results = results
        return results

    def save_attack_artifacts(self, save_dir: str):
        """
        Save all attack artifacts.

        Args:
            save_dir: Directory to save artifacts
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save trigger modules
        if self.ttm is not None:
            torch.save(self.ttm.state_dict(), save_path / 'ttm_state.pth')

        if self.stm is not None:
            torch.save(self.stm.state_dict(), save_path / 'stm_state.pth')

        # Save backdoor model
        if self.backdoor_model is not None:
            torch.save(self.backdoor_model.state_dict(), save_path / 'backdoor_model.pth')

        # Save configuration and metadata
        metadata = {
            'trigger_nodes': self.trigger_nodes,
            'target_nodes': self.target_nodes,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'config': {
                'poison_rate': self.config.attack.poison_rate,
                'trigger_mode': self.config.attack.trigger_mode,
                'temporal_type': self.config.temporal.trigger_type,
                'spatial_type': self.config.spatial.trigger_type,
            },
            'evaluation_results': self.evaluation_results
        }

        with open(save_path / 'attack_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        # Save adjacency matrix
        if self.adj_matrix is not None:
            np.save(save_path / 'adj_matrix.npy', self.adj_matrix)

        print(f"Attack artifacts saved to {save_path}")

    def load_attack_artifacts(self, load_dir: str, model_class: Optional[type] = None):
        """
        Load saved attack artifacts.

        Args:
            load_dir: Directory containing saved artifacts
            model_class: Model class for loading backdoor model (optional)
        """
        load_path = Path(load_dir)

        # Load metadata
        with open(load_path / 'attack_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.trigger_nodes = metadata['trigger_nodes']
        self.target_nodes = metadata['target_nodes']
        self.data_mean = metadata['data_mean']
        self.data_std = metadata['data_std']
        self.evaluation_results = metadata.get('evaluation_results', {})

        # Load adjacency
        if (load_path / 'adj_matrix.npy').exists():
            self.adj_matrix = np.load(load_path / 'adj_matrix.npy')

        # Load trigger modules
        if (load_path / 'ttm_state.pth').exists() and self.ttm is not None:
            self.ttm.load_state_dict(torch.load(load_path / 'ttm_state.pth'))

        if (load_path / 'stm_state.pth').exists() and self.stm is not None:
            self.stm.load_state_dict(torch.load(load_path / 'stm_state.pth'))

        print(f"Attack artifacts loaded from {load_path}")

    def run_full_attack(
        self,
        clean_train_data: np.ndarray,
        clean_train_labels: np.ndarray,
        clean_test_data: np.ndarray,
        clean_test_labels: np.ndarray,
        model_class: type,
        model_kwargs: Dict = None,
        surrogate_model: Optional[nn.Module] = None,
        target_nodes: Optional[List[int]] = None,
        row: Optional[int] = None,
        col: Optional[int] = None,
        train_trigger_generators: bool = True,
        epochs: int = 50,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute complete attack pipeline.

        Args:
            clean_train_data: Clean training data
            clean_train_labels: Clean training labels
            clean_test_data: Clean test data
            clean_test_labels: Clean test labels
            model_class: Model class to instantiate
            model_kwargs: Model initialization kwargs
            surrogate_model: Pre-trained surrogate (optional)
            target_nodes: Target attack region
            row, col: Grid dimensions
            train_trigger_generators: Whether to train triggers
            epochs: Training epochs
            save_dir: Save directory

        Returns:
            Complete attack results
        """
        model_kwargs = model_kwargs or {}

        print("=" * 60)
        print("CSTBA Full Attack Pipeline")
        print("=" * 60)

        # Step 1: Setup
        print("\n[Step 1] Setting up from data...")
        self.setup_from_data(clean_train_data, row=row, col=col)

        # Step 2: Graph analysis
        print("\n[Step 2] Analyzing graph structure...")
        graph_analysis = self.analyze_graph(target_nodes=target_nodes)

        # Step 3: Train trigger generators (optional)
        if train_trigger_generators:
            print("\n[Step 3] Training trigger generators...")
            if surrogate_model is None:
                surrogate_model = model_class(**model_kwargs).to(self.device)

            # Create data loader for training
            train_tensor = torch.tensor(clean_train_data, dtype=torch.float32)
            train_labels_tensor = torch.tensor(clean_train_labels, dtype=torch.float32)
            train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config.training.batch_size, shuffle=True
            )

            trigger_history = self.train_trigger_generators(
                train_loader, surrogate_model,
                num_iterations=self.config.training.bilevel_iterations
            )
        else:
            trigger_history = {}

        # Step 4: Create poisoned datasets
        print("\n[Step 4] Creating poisoned datasets...")
        poisoned_datasets = self.create_poisoned_data(
            clean_train_data, clean_train_labels,
            save_dir=save_dir
        )

        # Step 5: Train backdoor model (on joint poisoned data)
        print("\n[Step 5] Training backdoor model...")
        backdoor_model = model_class(**model_kwargs)
        joint_data, joint_labels, _, _ = poisoned_datasets['joint']
        backdoor_model, train_history = self.train_backdoor_model(
            backdoor_model, joint_data, joint_labels,
            epochs=epochs
        )

        # Step 6: Get baseline metrics (train clean model)
        print("\n[Step 6] Computing baseline metrics...")
        clean_model = model_class(**model_kwargs).to(self.device)
        clean_model, _ = self.train_backdoor_model(
            clean_model, clean_train_data, clean_train_labels,
            epochs=epochs, verbose=False
        )

        # Evaluate clean model
        clean_model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(clean_test_data, dtype=torch.float32, device=self.device)
            clean_pred = clean_model(test_tensor)
            if isinstance(clean_pred, tuple):
                clean_pred = clean_pred[0]
            clean_pred = clean_pred.cpu().numpy()

        baseline_rmse = np.sqrt(np.mean((clean_pred - clean_test_labels) ** 2))
        baseline_mae = np.mean(np.abs(clean_pred - clean_test_labels))
        baseline_metrics = {'rmse': baseline_rmse, 'mae': baseline_mae}
        print(f"Baseline RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}")

        # Step 7: Evaluate attack
        print("\n[Step 7] Evaluating attack effectiveness...")
        evaluation_results = self.evaluate(
            backdoor_model, clean_test_data, clean_test_labels,
            baseline_metrics
        )

        # Step 8: Save artifacts
        if save_dir:
            print("\n[Step 8] Saving attack artifacts...")
            self.save_attack_artifacts(save_dir)

        # Summary
        print("\n" + "=" * 60)
        print("Attack Pipeline Completed!")
        print("=" * 60)

        return {
            'graph_analysis': graph_analysis,
            'trigger_history': trigger_history,
            'train_history': train_history,
            'baseline_metrics': baseline_metrics,
            'evaluation': evaluation_results
        }


if __name__ == "__main__":
    print("CSTBA Attack class loaded successfully.")
    print("Use CSTBAAttack() to initialize and run attacks.")
