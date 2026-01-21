"""
Synergy Optimizer

Implements bilevel optimization framework for joint TTM and STM optimization.
Ensures "1+1>2" collaborative effect through mutual information regularization.

Based on CSTBA Algorithm Design Document Section 3: Joint Optimization Strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from copy import deepcopy


class MINEEstimator(nn.Module):
    """
    Mutual Information Neural Estimator (MINE)

    Estimates mutual information I(X; Y) using neural network based lower bound.
    Used for computing synergy loss between temporal and spatial triggers.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute MINE estimate of mutual information.

        Args:
            x: First variable [B, D1]
            y: Second variable [B, D2]

        Returns:
            MI estimate scalar
        """
        batch_size = x.shape[0]

        # Joint samples (x, y)
        joint = torch.cat([x, y], dim=-1)
        t_joint = self.network(joint)

        # Marginal samples (x, y_shuffled)
        y_shuffled = y[torch.randperm(batch_size)]
        marginal = torch.cat([x, y_shuffled], dim=-1)
        t_marginal = self.network(marginal)

        # MINE lower bound: E[T(x,y)] - log(E[exp(T(x,y'))])
        mi_estimate = t_joint.mean() - torch.log(torch.exp(t_marginal).mean() + 1e-8)

        return mi_estimate


class InfoNCEEstimator(nn.Module):
    """
    InfoNCE-based Mutual Information Estimator

    Alternative MI estimator using contrastive learning approach.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

        # Projection heads
        self.proj_x = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_y = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE estimate of mutual information.

        Args:
            x: First variable [B, D1]
            y: Second variable [B, D2]

        Returns:
            MI estimate (as negative InfoNCE loss, to maximize)
        """
        batch_size = x.shape[0]

        # Project to common space
        z_x = F.normalize(self.proj_x(x), dim=-1)
        z_y = F.normalize(self.proj_y(y), dim=-1)

        # Compute similarity matrix
        similarity = torch.mm(z_x, z_y.T) / self.temperature  # [B, B]

        # InfoNCE loss (positive pairs on diagonal)
        labels = torch.arange(batch_size, device=x.device)
        loss = F.cross_entropy(similarity, labels)

        # Return negative loss as MI estimate (higher = more MI)
        return -loss


class SynergyOptimizer:
    """
    Synergy Optimizer

    Implements bilevel optimization framework for jointly optimizing
    Temporal Trigger Module (TTM) and Spatial Trigger Module (STM).

    Key Features:
    1. Bilevel optimization: outer loop for triggers, inner loop for surrogate model
    2. Attack loss: maximize prediction shift toward target
    3. Stealth loss: maintain temporal smoothness, frequency consistency, homophily
    4. Synergy loss: maximize mutual information between temporal and spatial triggers
    """

    def __init__(
        self,
        ttm_module: nn.Module,
        stm_module: nn.Module,
        surrogate_model: nn.Module,
        lambda_attack: float = 1.0,
        lambda_stealth: float = 0.5,
        lambda_synergy: float = 0.3,
        mi_estimation: str = "mine",
        device: torch.device = None
    ):
        """
        Args:
            ttm_module: Temporal Trigger Module instance
            stm_module: Spatial Trigger Module instance
            surrogate_model: Surrogate ST-GNN model for optimization
            lambda_attack: Attack loss weight
            lambda_stealth: Stealthiness loss weight
            lambda_synergy: Synergy loss weight
            mi_estimation: MI estimation method ("mine", "infonce", "nwj")
            device: Computation device
        """
        self.ttm = ttm_module
        self.stm = stm_module
        self.surrogate = surrogate_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Loss weights (from design document Section 3.2)
        self.lambda_attack = lambda_attack
        self.lambda_stealth = lambda_stealth
        self.lambda_synergy = lambda_synergy

        # Move modules to device
        self.ttm.to(self.device)
        self.stm.to(self.device)
        self.surrogate.to(self.device)

        # Initialize MI estimator
        # Estimate dimensions based on typical trigger shapes
        self._init_mi_estimator(mi_estimation)

        # Optimization history
        self.history = {
            'attack_loss': [],
            'stealth_loss': [],
            'synergy_loss': [],
            'total_loss': [],
            'inner_loss': []
        }

    def _init_mi_estimator(self, method: str):
        """Initialize mutual information estimator."""
        # Use feature dimension for MI estimation
        x_dim = 128  # Will be adjusted based on actual trigger size
        y_dim = 128

        if method == "mine":
            self.mi_estimator = MINEEstimator(x_dim, y_dim).to(self.device)
        elif method == "infonce":
            self.mi_estimator = InfoNCEEstimator(x_dim, y_dim).to(self.device)
        else:
            self.mi_estimator = MINEEstimator(x_dim, y_dim).to(self.device)

        self.mi_method = method

    def compute_attack_loss(
        self,
        predictions: torch.Tensor,
        original_predictions: torch.Tensor,
        target_direction: str = "increase",
        target_scale: float = 2.0
    ) -> torch.Tensor:
        """
        Compute attack loss (maximize prediction shift toward target).

        For regression tasks (crime prediction):
            L_attack = -mean(shift) if target_direction == "increase"
                     = mean(shift) if target_direction == "decrease"

        Args:
            predictions: Predictions on poisoned data [B, ...]
            original_predictions: Predictions on clean data [B, ...]
            target_direction: "increase" or "decrease" predictions
            target_scale: Expected scale factor for attack success

        Returns:
            Attack loss scalar (minimize this to maximize attack effect)
        """
        # Compute prediction shift
        shift = predictions - original_predictions

        if target_direction == "increase":
            # Want to maximize positive shift
            # Loss = -mean(shift) so minimizing loss maximizes shift
            target_shift = original_predictions.std() * (target_scale - 1)
            attack_loss = -torch.mean(shift) + F.relu(target_shift - shift.mean())
        else:
            # Want to maximize negative shift
            target_shift = -original_predictions.std() * (target_scale - 1)
            attack_loss = torch.mean(shift) + F.relu(shift.mean() - target_shift)

        return attack_loss

    def compute_stealth_loss(
        self,
        original_data: torch.Tensor,
        poisoned_data: torch.Tensor,
        delta_t: torch.Tensor,
        original_adj: torch.Tensor,
        modified_adj: torch.Tensor,
        modified_features: torch.Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute comprehensive stealthiness loss.

        L_stealth = w1*L_smooth + w2*L_freq + w3*L_homophily + w4*L_topology

        Based on design document Section 3.2.

        Args:
            original_data: Original clean data [B, T, V, C]
            poisoned_data: Poisoned data [B, T, V, C]
            delta_t: Temporal trigger [B, T, V, C]
            original_adj: Original adjacency [V, V]
            modified_adj: Modified adjacency [V, V]
            modified_features: Modified node features
            weights: Component weights

        Returns:
            Total stealth loss and component dict
        """
        weights = weights or {
            'smoothness': 0.3,
            'frequency': 0.3,
            'homophily': 0.2,
            'topology': 0.2
        }

        components = {}

        # 1. Temporal smoothness loss
        smoothness_loss = self.ttm.compute_smoothness_loss(delta_t)
        components['smoothness'] = smoothness_loss

        # 2. Frequency consistency loss
        frequency_loss = self.ttm.compute_frequency_loss(original_data, poisoned_data)
        components['frequency'] = frequency_loss

        # 3. Homophily loss (for spatial trigger)
        if hasattr(self.stm, 'compute_homophily_loss'):
            homophily_loss = self.stm.compute_homophily_loss(modified_features, modified_adj)
        else:
            homophily_loss = torch.tensor(0.0, device=self.device)
        components['homophily'] = homophily_loss

        # 4. Topology stealth loss
        if hasattr(self.stm, 'compute_topology_stealth_loss'):
            topology_loss = self.stm.compute_topology_stealth_loss(original_adj, modified_adj)
        else:
            topology_loss = torch.tensor(0.0, device=self.device)
        components['topology'] = topology_loss

        # Weighted sum
        total_stealth = (
            weights['smoothness'] * smoothness_loss +
            weights['frequency'] * frequency_loss +
            weights['homophily'] * homophily_loss +
            weights['topology'] * topology_loss
        )

        return total_stealth, components

    def compute_synergy_loss(
        self,
        delta_t: torch.Tensor,
        delta_s: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute synergy loss using mutual information.

        L_synergy = -I(delta_t; delta_s | Y_target)

        Maximizes mutual information between temporal and spatial triggers,
        encouraging them to capture complementary information.

        Based on design document Section 3.2.

        Args:
            delta_t: Temporal trigger [B, T, V, C]
            delta_s: Spatial trigger (delta_f) [B, V, C] or [B, T, V, C]
            predictions: Model predictions [B, ...]

        Returns:
            Synergy loss scalar (negative MI, to minimize for maximizing MI)
        """
        batch_size = delta_t.shape[0]

        # Flatten triggers to feature vectors
        delta_t_flat = delta_t.view(batch_size, -1)  # [B, T*V*C]
        if delta_s.dim() == 4:
            delta_s_flat = delta_s.view(batch_size, -1)
        else:
            delta_s_flat = delta_s.view(batch_size, -1)

        # Reduce dimensionality for MI estimation (use PCA-like projection)
        target_dim = 128

        if delta_t_flat.shape[1] > target_dim:
            # Simple random projection for efficiency
            if not hasattr(self, '_proj_t'):
                self._proj_t = nn.Linear(delta_t_flat.shape[1], target_dim, bias=False).to(self.device)
                nn.init.orthogonal_(self._proj_t.weight)
            delta_t_proj = self._proj_t(delta_t_flat)
        else:
            delta_t_proj = F.pad(delta_t_flat, (0, target_dim - delta_t_flat.shape[1]))

        if delta_s_flat.shape[1] > target_dim:
            if not hasattr(self, '_proj_s'):
                self._proj_s = nn.Linear(delta_s_flat.shape[1], target_dim, bias=False).to(self.device)
                nn.init.orthogonal_(self._proj_s.weight)
            delta_s_proj = self._proj_s(delta_s_flat)
        else:
            delta_s_proj = F.pad(delta_s_flat, (0, target_dim - delta_s_flat.shape[1]))

        # Estimate mutual information
        mi_estimate = self.mi_estimator(delta_t_proj, delta_s_proj)

        # Return negative MI (we want to maximize MI, so minimize -MI)
        synergy_loss = -mi_estimate

        return synergy_loss

    def inner_loop_update(
        self,
        poisoned_data: torch.Tensor,
        target_labels: torch.Tensor,
        adj_matrix: torch.Tensor,
        inner_lr: float = 0.001,
        inner_steps: int = 5
    ) -> float:
        """
        Inner loop: Update surrogate model parameters on poisoned data.

        θ* = argmin_θ L_train(f_θ(X_poison), Y_target)

        Args:
            poisoned_data: Poisoned training data [B, T, V, C]
            target_labels: Target labels for poisoned samples [B, ...]
            adj_matrix: Adjacency matrix [V, V]
            inner_lr: Learning rate for inner optimization
            inner_steps: Number of inner optimization steps

        Returns:
            Final inner loss value
        """
        # Create optimizer for surrogate model
        inner_optimizer = torch.optim.Adam(self.surrogate.parameters(), lr=inner_lr)

        self.surrogate.train()
        total_loss = 0.0

        for step in range(inner_steps):
            inner_optimizer.zero_grad()

            # Forward pass through surrogate
            predictions = self.surrogate(poisoned_data)
            if isinstance(predictions, tuple):
                predictions = predictions[0]

            # Compute training loss (MSE for regression)
            loss = F.mse_loss(predictions, target_labels)

            loss.backward()
            inner_optimizer.step()

            total_loss = loss.item()

        return total_loss

    def outer_loop_update(
        self,
        clean_data: torch.Tensor,
        adj_matrix: torch.Tensor,
        target_direction: str = "increase",
        target_scale: float = 2.0,
        outer_lr: float = 0.0001,
        stealth_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Outer loop: Update trigger generator parameters.

        min_{ψ,φ} L_attack + λ1*L_stealth + λ2*L_synergy

        Args:
            clean_data: Original clean data [B, T, V, C]
            adj_matrix: Adjacency matrix [V, V]
            target_direction: Attack target direction
            target_scale: Expected prediction scale factor
            outer_lr: Outer optimization learning rate
            stealth_weights: Weights for stealth loss components

        Returns:
            Total loss and component dict
        """
        # Create optimizer for trigger generators
        outer_params = list(self.ttm.parameters()) + list(self.stm.parameters())
        if hasattr(self, 'mi_estimator'):
            outer_params += list(self.mi_estimator.parameters())

        outer_optimizer = torch.optim.Adam(outer_params, lr=outer_lr)

        self.ttm.train()
        self.stm.train()
        self.surrogate.eval()  # Freeze surrogate for outer loop

        outer_optimizer.zero_grad()

        # Generate triggers
        # 1. Temporal trigger
        poisoned_temporal, delta_t = self.ttm(clean_data, adj_matrix)

        # 2. Spatial trigger (applied to temporally poisoned data)
        # Reshape for STM if needed
        B, T, V, C = poisoned_temporal.shape
        poisoned_full, modified_adj, stm_info = self.stm(
            poisoned_temporal, adj_matrix
        )
        delta_s = stm_info.get('delta_f', torch.zeros_like(poisoned_temporal))

        # Get predictions
        with torch.no_grad():
            original_pred = self.surrogate(clean_data)
            if isinstance(original_pred, tuple):
                original_pred = original_pred[0]

        poisoned_pred = self.surrogate(poisoned_full)
        if isinstance(poisoned_pred, tuple):
            poisoned_pred = poisoned_pred[0]

        # Compute losses
        attack_loss = self.compute_attack_loss(
            poisoned_pred, original_pred, target_direction, target_scale
        )

        stealth_loss, stealth_components = self.compute_stealth_loss(
            clean_data, poisoned_full, delta_t,
            adj_matrix, modified_adj,
            poisoned_full[:, -1, :, :],  # Last timestep features
            stealth_weights
        )

        synergy_loss = self.compute_synergy_loss(delta_t, delta_s, poisoned_pred)

        # Total loss
        total_loss = (
            self.lambda_attack * attack_loss +
            self.lambda_stealth * stealth_loss +
            self.lambda_synergy * synergy_loss
        )

        total_loss.backward()
        outer_optimizer.step()

        # Record history
        losses = {
            'attack': attack_loss.item(),
            'stealth': stealth_loss.item(),
            'synergy': synergy_loss.item(),
            'total': total_loss.item(),
            **{f'stealth_{k}': v.item() for k, v in stealth_components.items()}
        }

        return total_loss.item(), losses

    def bilevel_optimize(
        self,
        train_loader: torch.utils.data.DataLoader,
        adj_matrix: torch.Tensor,
        target_direction: str = "increase",
        target_scale: float = 2.0,
        num_iterations: int = 100,
        inner_lr: float = 0.001,
        outer_lr: float = 0.0001,
        inner_steps: int = 5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Execute bilevel optimization.

        Alternates between:
        1. Inner loop: Train surrogate on poisoned data
        2. Outer loop: Optimize trigger generators

        Args:
            train_loader: DataLoader for training data
            adj_matrix: Graph adjacency matrix [V, V]
            target_direction: Attack direction ("increase" or "decrease")
            target_scale: Target prediction scale factor
            num_iterations: Number of bilevel iterations
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop learning rate
            inner_steps: Inner loop steps per outer step
            verbose: Print progress

        Returns:
            Training history dict
        """
        adj_matrix = adj_matrix.to(self.device)

        for iteration in range(num_iterations):
            epoch_losses = {
                'attack': 0.0, 'stealth': 0.0, 'synergy': 0.0,
                'total': 0.0, 'inner': 0.0
            }
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    clean_data = batch[0].to(self.device)
                    labels = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    clean_data = batch.to(self.device)
                    labels = None

                # Generate poisoned data for inner loop
                with torch.no_grad():
                    poisoned_data, delta_t = self.ttm(clean_data, adj_matrix)
                    poisoned_data, _, _ = self.stm(poisoned_data, adj_matrix)

                # Create target labels (shifted predictions)
                if labels is not None:
                    if target_direction == "increase":
                        target_labels = labels * target_scale
                    else:
                        target_labels = labels / target_scale
                else:
                    with torch.no_grad():
                        base_pred = self.surrogate(clean_data)
                        if isinstance(base_pred, tuple):
                            base_pred = base_pred[0]
                        if target_direction == "increase":
                            target_labels = base_pred * target_scale
                        else:
                            target_labels = base_pred / target_scale

                # Inner loop
                inner_loss = self.inner_loop_update(
                    poisoned_data.detach(), target_labels,
                    adj_matrix, inner_lr, inner_steps
                )
                epoch_losses['inner'] += inner_loss

                # Outer loop
                total_loss, outer_losses = self.outer_loop_update(
                    clean_data, adj_matrix,
                    target_direction, target_scale, outer_lr
                )

                for key in ['attack', 'stealth', 'synergy', 'total']:
                    epoch_losses[key] += outer_losses.get(key, 0.0)

                num_batches += 1

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= max(num_batches, 1)
                self.history[f'{key}_loss' if key != 'total' else 'total_loss'].append(epoch_losses[key])

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}: "
                      f"Attack={epoch_losses['attack']:.4f}, "
                      f"Stealth={epoch_losses['stealth']:.4f}, "
                      f"Synergy={epoch_losses['synergy']:.4f}, "
                      f"Total={epoch_losses['total']:.4f}")

        return self.history

    def get_optimized_triggers(self) -> Tuple[nn.Module, nn.Module]:
        """Return optimized trigger modules."""
        return self.ttm, self.stm


if __name__ == "__main__":
    print("Testing Synergy Optimizer...")

    # Create dummy modules
    from cstba.modules.temporal_trigger import TemporalTriggerModule
    from cstba.modules.spatial_trigger import SpatialTriggerModule

    B, T, V, C = 4, 30, 100, 4

    # Simple surrogate model for testing
    class DummySurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(C, C, 3, padding=1)
            self.fc = nn.Linear(T * V * C, V * C)

        def forward(self, x):
            B = x.shape[0]
            x = x.permute(0, 3, 1, 2)  # [B, C, T, V]
            x = self.conv(x)
            x = x.view(B, -1)
            x = self.fc(x)
            return x.view(B, V, C)

    ttm = TemporalTriggerModule(C, 32, V)
    stm = SpatialTriggerModule(C, V)
    stm.set_trigger_nodes([10, 20, 30, 40, 50])
    surrogate = DummySurrogate()

    # Create optimizer
    optimizer = SynergyOptimizer(ttm, stm, surrogate)

    # Test with dummy data
    clean_data = torch.randn(B, T, V, C)
    adj = torch.eye(V) + torch.randn(V, V) * 0.1
    adj = (adj > 0.5).float()

    # Test outer loop
    loss, losses = optimizer.outer_loop_update(clean_data, adj)
    print(f"Outer loop loss: {loss:.4f}")
    print(f"Components: {losses}")

    print("\nSynergy Optimizer test completed!")
