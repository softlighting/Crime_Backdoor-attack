"""
Spatial Trigger Module (STM)

Generates spatial triggers through node feature and topology modifications.

Core Features:
1. Feature trigger: Modify key node features
2. Topology trigger: Inject fake edges or modify edge weights
3. Homophily camouflage: Ensure modifications blend with neighbor distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal, Union


class SpatialTriggerModule(nn.Module):
    """
    Spatial Trigger Module (STM)

    Generates spatial triggers for backdoor attacks on ST-GNNs through:
    1. Feature modifications on key nodes
    2. Topology modifications (edge injection)
    3. Homophily-based camouflage for stealthiness
    """

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        trigger_type: Literal["feature", "topology", "both"] = "feature",
        homophily_constraint: bool = True,
        perturbation_ratio: float = 0.1
    ):
        """
        Args:
            feature_dim: Node feature dimension
            num_nodes: Total number of nodes in graph
            trigger_type: Type of spatial trigger
            homophily_constraint: Whether to use homophily-based camouflage
            perturbation_ratio: Perturbation strength as ratio of feature std
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.trigger_type = trigger_type
        self.homophily_constraint = homophily_constraint
        self.perturbation_ratio = perturbation_ratio

        # Trigger node indices (to be set later)
        self.register_buffer('trigger_node_mask', torch.zeros(num_nodes, dtype=torch.bool))
        self.trigger_node_indices: List[int] = []

        # Feature trigger parameters
        if trigger_type in ["feature", "both"]:
            # Learnable perturbation for each feature dimension
            self.feature_perturbation = nn.Parameter(torch.zeros(feature_dim))

            # Node-specific scaling
            self.node_scaling = nn.Parameter(torch.ones(num_nodes))

            # Feature generator network
            self.feature_generator = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Tanh()
            )

        # Topology trigger parameters
        if trigger_type in ["topology", "both"]:
            # Learnable edge weights for injected edges
            self.edge_weights = nn.Parameter(torch.ones(num_nodes, num_nodes) * 0.01)

            # Edge injection mask (which edges to inject)
            self.register_buffer('edge_mask', torch.zeros(num_nodes, num_nodes, dtype=torch.bool))

        # Data statistics
        self.register_buffer('feature_mean', torch.zeros(feature_dim))
        self.register_buffer('feature_std', torch.ones(feature_dim))

    def set_data_statistics(self, mean: torch.Tensor, std: torch.Tensor):
        """Set feature statistics for proper scaling."""
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)

    def set_trigger_nodes(self, trigger_node_indices: List[int]):
        """
        Set trigger node indices.

        Args:
            trigger_node_indices: List of node indices to use as triggers
        """
        self.trigger_node_indices = trigger_node_indices

        # Update mask
        self.trigger_node_mask.zero_()
        for idx in trigger_node_indices:
            if 0 <= idx < self.num_nodes:
                self.trigger_node_mask[idx] = True

    def set_edge_injection(
        self,
        source_nodes: List[int],
        target_nodes: List[int]
    ):
        """
        Set edges to inject for topology trigger.

        Args:
            source_nodes: Source node indices (typically trigger nodes)
            target_nodes: Target node indices (attack targets)
        """
        self.edge_mask.zero_()
        for src in source_nodes:
            for tgt in target_nodes:
                if src != tgt and 0 <= src < self.num_nodes and 0 <= tgt < self.num_nodes:
                    self.edge_mask[src, tgt] = True
                    self.edge_mask[tgt, src] = True  # Undirected

    def generate_feature_trigger(
        self,
        node_features: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate feature trigger perturbation.

        Args:
            node_features: Node features [B, V, C] or [V, C]
            adj_matrix: Adjacency matrix [V, V]

        Returns:
            delta_f: Feature perturbation [B, V, C] or [V, C]
        """
        # Handle dimensions
        squeeze_output = False
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            squeeze_output = True

        B, V, C = node_features.shape
        device = node_features.device

        # Generate base perturbation using feature generator
        # Use mean features of trigger nodes as input
        trigger_features = node_features[:, self.trigger_node_mask, :]  # [B, K, C]
        if trigger_features.shape[1] > 0:
            mean_trigger_features = trigger_features.mean(dim=1)  # [B, C]
        else:
            mean_trigger_features = node_features.mean(dim=1)

        base_perturbation = self.feature_generator(mean_trigger_features)  # [B, C]

        # Initialize delta_f with zeros
        delta_f = torch.zeros_like(node_features)

        # Apply perturbation only to trigger nodes
        for idx in self.trigger_node_indices:
            if 0 <= idx < V:
                # Scale perturbation by learned node-specific factor
                scaling = torch.sigmoid(self.node_scaling[idx])
                # Combine learned perturbation with generated
                combined = self.feature_perturbation + base_perturbation
                delta_f[:, idx, :] = combined * scaling

        # Apply homophily constraint if enabled
        if self.homophily_constraint and adj_matrix is not None:
            delta_f = self._apply_homophily_constraint(delta_f, node_features, adj_matrix)

        # Scale by data statistics
        max_perturbation = self.perturbation_ratio * self.feature_std
        delta_f = torch.tanh(delta_f) * max_perturbation.view(1, 1, -1)

        if squeeze_output:
            delta_f = delta_f.squeeze(0)

        return delta_f

    def _apply_homophily_constraint(
        self,
        delta_f: torch.Tensor,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply homophily constraint to feature perturbation.

        Adjusts perturbation so modified features match neighbor distribution.

        Args:
            delta_f: Raw perturbation [B, V, C]
            node_features: Original features [B, V, C]
            adj_matrix: Adjacency matrix [V, V]

        Returns:
            Adjusted perturbation [B, V, C]
        """
        B, V, C = node_features.shape

        # Compute neighbor mean for each node
        adj_normalized = adj_matrix / (adj_matrix.sum(dim=1, keepdim=True) + 1e-8)
        neighbor_mean = torch.bmm(
            adj_normalized.unsqueeze(0).expand(B, -1, -1),
            node_features
        )  # [B, V, C]

        # For trigger nodes, adjust perturbation toward neighbor mean
        adjusted_delta = delta_f.clone()

        for idx in self.trigger_node_indices:
            if 0 <= idx < V:
                current_feature = node_features[:, idx, :]  # [B, C]
                target_mean = neighbor_mean[:, idx, :]  # [B, C]
                raw_delta = delta_f[:, idx, :]  # [B, C]

                # Direction toward neighbor mean
                direction = target_mean - current_feature
                direction_norm = F.normalize(direction, dim=-1)

                # Project perturbation onto direction (keep component aligned with homophily)
                raw_norm = F.normalize(raw_delta, dim=-1)
                alignment = (raw_norm * direction_norm).sum(dim=-1, keepdim=True)

                # Adjust perturbation to be more aligned with neighbor distribution
                adjusted = raw_delta + 0.3 * direction * torch.abs(alignment)
                adjusted_delta[:, idx, :] = adjusted

        return adjusted_delta

    def generate_topology_trigger(
        self,
        adj_matrix: torch.Tensor,
        target_nodes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Generate topology trigger (edge modifications).

        Args:
            adj_matrix: Original adjacency matrix [V, V]
            target_nodes: Target attack nodes (for directed edge injection)

        Returns:
            delta_A: Adjacency matrix modification [V, V]
        """
        device = adj_matrix.device
        V = adj_matrix.shape[0]

        # Initialize modification
        delta_A = torch.zeros_like(adj_matrix)

        if target_nodes is None:
            target_nodes = list(range(V))

        # Get edge weights (constrained to positive)
        weights = torch.sigmoid(self.edge_weights) * 0.5  # Scale to [0, 0.5]

        # Apply only to specified edges
        delta_A = weights * self.edge_mask.float()

        # Ensure symmetry (undirected graph)
        delta_A = (delta_A + delta_A.T) / 2

        return delta_A

    def compute_homophily_loss(
        self,
        modified_features: torch.Tensor,
        adj_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute homophily loss for feature modifications.

        L_homophily = sum_v ||f_v - mean(f_neighbors)||^2 for v in trigger_nodes

        Args:
            modified_features: Features after trigger injection [B, V, C] or [V, C]
            adj_matrix: Adjacency matrix [V, V]

        Returns:
            Homophily loss scalar
        """
        if modified_features.dim() == 2:
            modified_features = modified_features.unsqueeze(0)

        B, V, C = modified_features.shape

        # Compute neighbor means
        adj_normalized = adj_matrix / (adj_matrix.sum(dim=1, keepdim=True) + 1e-8)
        neighbor_mean = torch.bmm(
            adj_normalized.unsqueeze(0).expand(B, -1, -1),
            modified_features
        )

        # Compute loss only for trigger nodes
        loss = 0.0
        count = 0
        for idx in self.trigger_node_indices:
            if 0 <= idx < V:
                node_features = modified_features[:, idx, :]
                node_neighbor_mean = neighbor_mean[:, idx, :]
                loss = loss + torch.mean((node_features - node_neighbor_mean) ** 2)
                count += 1

        return loss / max(count, 1)

    def compute_topology_stealth_loss(
        self,
        original_adj: torch.Tensor,
        modified_adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute topology stealthiness loss.

        Ensures injected edge weights follow original distribution.

        Args:
            original_adj: Original adjacency matrix [V, V]
            modified_adj: Modified adjacency matrix [V, V]

        Returns:
            Topology stealth loss scalar
        """
        # Get statistics of original edge weights
        orig_edges = original_adj[original_adj > 0]
        if len(orig_edges) == 0:
            return torch.tensor(0.0, device=original_adj.device)

        orig_mean = orig_edges.mean()
        orig_std = orig_edges.std() + 1e-8

        # Get injected edge weights
        delta_A = modified_adj - original_adj
        new_edges = delta_A[delta_A > 0]

        if len(new_edges) == 0:
            return torch.tensor(0.0, device=original_adj.device)

        # Penalize deviation from original distribution
        new_mean = new_edges.mean()
        new_std = new_edges.std() + 1e-8

        mean_loss = (new_mean - orig_mean) ** 2
        std_loss = (new_std - orig_std) ** 2

        return mean_loss + 0.5 * std_loss

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        target_nodes: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass: generate and apply spatial triggers.

        Args:
            node_features: Node features [B, V, C] or [B, T, V, C]
            adj_matrix: Adjacency matrix [V, V]
            target_nodes: Target attack nodes (optional)

        Returns:
            Tuple of (modified_features, modified_adj, trigger_info dict)
        """
        # Handle 4D input (spatio-temporal)
        if node_features.dim() == 4:
            B, T, V, C = node_features.shape
            # Apply trigger to each time step
            modified_features = node_features.clone()
            delta_f_total = torch.zeros_like(node_features)

            for t in range(T):
                features_t = node_features[:, t, :, :]  # [B, V, C]
                if self.trigger_type in ["feature", "both"]:
                    delta_f = self.generate_feature_trigger(features_t, adj_matrix)
                    modified_features[:, t, :, :] = features_t + delta_f
                    delta_f_total[:, t, :, :] = delta_f
        else:
            delta_f_total = None
            if self.trigger_type in ["feature", "both"]:
                delta_f = self.generate_feature_trigger(node_features, adj_matrix)
                modified_features = node_features + delta_f
                delta_f_total = delta_f
            else:
                modified_features = node_features

        # Topology trigger
        if self.trigger_type in ["topology", "both"]:
            delta_A = self.generate_topology_trigger(adj_matrix, target_nodes)
            modified_adj = adj_matrix + delta_A
        else:
            modified_adj = adj_matrix
            delta_A = torch.zeros_like(adj_matrix)

        trigger_info = {
            'delta_f': delta_f_total,
            'delta_A': delta_A,
            'trigger_nodes': self.trigger_node_indices
        }

        return modified_features, modified_adj, trigger_info

    def compute_total_loss(
        self,
        modified_features: torch.Tensor,
        original_adj: torch.Tensor,
        modified_adj: torch.Tensor,
        homophily_weight: float = 0.3,
        topology_weight: float = 0.3
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total STM loss.

        Args:
            modified_features: Features after trigger [B, V, C]
            original_adj: Original adjacency [V, V]
            modified_adj: Modified adjacency [V, V]
            homophily_weight: Weight for homophily loss
            topology_weight: Weight for topology stealth loss

        Returns:
            Total loss and component dict
        """
        losses = {}

        if self.trigger_type in ["feature", "both"]:
            homophily_loss = self.compute_homophily_loss(modified_features, modified_adj)
            losses['homophily'] = homophily_loss
        else:
            homophily_loss = torch.tensor(0.0, device=modified_features.device)

        if self.trigger_type in ["topology", "both"]:
            topology_loss = self.compute_topology_stealth_loss(original_adj, modified_adj)
            losses['topology'] = topology_loss
        else:
            topology_loss = torch.tensor(0.0, device=modified_features.device)

        total_loss = homophily_weight * homophily_loss + topology_weight * topology_loss
        losses['total'] = total_loss

        return total_loss, losses


if __name__ == "__main__":
    print("Testing Spatial Trigger Module...")

    # Create dummy data
    B, V, C = 4, 100, 4
    node_features = torch.randn(B, V, C)

    # Create grid adjacency
    from cstba.utils.node_selection import build_grid_adjacency
    adj = torch.tensor(build_grid_adjacency(10, 10, connectivity=8), dtype=torch.float32)

    # Test each trigger type
    for trigger_type in ["feature", "topology", "both"]:
        print(f"\nTesting {trigger_type} trigger:")

        stm = SpatialTriggerModule(
            feature_dim=C,
            num_nodes=V,
            trigger_type=trigger_type
        )

        # Set trigger nodes
        stm.set_trigger_nodes([10, 15, 20, 25, 30])

        if trigger_type in ["topology", "both"]:
            stm.set_edge_injection([10, 15, 20], [50, 51, 52])

        # Forward pass
        mod_features, mod_adj, info = stm(node_features, adj)

        print(f"  Input features shape: {node_features.shape}")
        print(f"  Modified features shape: {mod_features.shape}")
        print(f"  Adjacency modification: {(mod_adj - adj).abs().sum():.4f}")

        # Test losses
        total_loss, losses = stm.compute_total_loss(mod_features, adj, mod_adj)
        print(f"  Total loss: {total_loss:.6f}")
        for k, v in losses.items():
            if k != 'total':
                print(f"    {k}: {v:.6f}")

    print("\nSTM test completed!")
