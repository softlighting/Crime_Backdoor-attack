"""
Node Selection Strategies for Spatial Trigger Module

Implements multi-metric fusion strategy for identifying critical nodes in graphs.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class NodeSelector:
    """
    Critical Node Selector

    Uses multi-metric fusion strategy to identify key nodes in graphs
    for effective backdoor trigger injection.
    """

    def __init__(
        self,
        adj_matrix: Union[np.ndarray, torch.Tensor],
        centrality_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            adj_matrix: Adjacency matrix [V, V]
            centrality_weights: Weights for centrality metrics
                Default: {'pagerank': 0.4, 'betweenness': 0.3, 'eigenvector': 0.3}
        """
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        self.adj_matrix = adj_matrix.astype(np.float32)
        self.num_nodes = adj_matrix.shape[0]

        # Default weights
        self.centrality_weights = centrality_weights or {
            'pagerank': 0.4,
            'betweenness': 0.3,
            'eigenvector': 0.3
        }

        # Normalize weights
        total_weight = sum(self.centrality_weights.values())
        self.centrality_weights = {k: v / total_weight for k, v in self.centrality_weights.items()}

        # Cache for computed centralities
        self._pagerank = None
        self._betweenness = None
        self._eigenvector = None
        self._degree = None

    def compute_degree_centrality(self) -> np.ndarray:
        """Compute degree centrality for all nodes."""
        if self._degree is None:
            degrees = np.sum(self.adj_matrix > 0, axis=1)
            max_degree = self.num_nodes - 1
            self._degree = degrees / max_degree
        return self._degree

    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Compute PageRank centrality using power iteration.

        Args:
            damping: Damping factor (probability of following links)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            PageRank scores [V]
        """
        if self._pagerank is not None:
            return self._pagerank

        # Normalize adjacency to transition matrix
        out_degree = np.sum(self.adj_matrix, axis=1, keepdims=True)
        out_degree = np.where(out_degree == 0, 1, out_degree)  # Avoid division by zero
        transition = self.adj_matrix / out_degree

        # Initialize PageRank
        pr = np.ones(self.num_nodes) / self.num_nodes
        teleport = np.ones(self.num_nodes) / self.num_nodes

        # Power iteration
        for _ in range(max_iter):
            pr_new = damping * transition.T @ pr + (1 - damping) * teleport
            if np.linalg.norm(pr_new - pr, 1) < tol:
                break
            pr = pr_new

        self._pagerank = pr / pr.sum()  # Normalize
        return self._pagerank

    def compute_betweenness(self) -> np.ndarray:
        """
        Compute betweenness centrality using shortest paths.

        Returns:
            Betweenness centrality scores [V]
        """
        if self._betweenness is not None:
            return self._betweenness

        # Create sparse matrix for efficiency
        sparse_adj = csr_matrix(self.adj_matrix)

        # Compute all-pairs shortest paths
        dist_matrix, predecessors = shortest_path(
            sparse_adj,
            directed=False,
            return_predecessors=True
        )

        # Count paths through each node
        betweenness = np.zeros(self.num_nodes)

        for s in range(self.num_nodes):
            for t in range(s + 1, self.num_nodes):
                if np.isinf(dist_matrix[s, t]):
                    continue

                # Reconstruct path
                path = []
                current = t
                while current != s and current >= 0:
                    path.append(current)
                    current = predecessors[s, current]
                path.append(s)

                # Count intermediate nodes
                for node in path[1:-1]:
                    betweenness[node] += 1

        # Normalize
        if self.num_nodes > 2:
            norm = (self.num_nodes - 1) * (self.num_nodes - 2) / 2
            betweenness = betweenness / norm if norm > 0 else betweenness

        self._betweenness = betweenness
        return self._betweenness

    def compute_eigenvector_centrality(
        self,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Compute eigenvector centrality using power iteration.

        Returns:
            Eigenvector centrality scores [V]
        """
        if self._eigenvector is not None:
            return self._eigenvector

        # Power iteration for dominant eigenvector
        x = np.ones(self.num_nodes) / np.sqrt(self.num_nodes)

        for _ in range(max_iter):
            x_new = self.adj_matrix @ x
            norm = np.linalg.norm(x_new)
            if norm == 0:
                break
            x_new = x_new / norm

            if np.linalg.norm(x_new - x, 1) < tol:
                break
            x = x_new

        # Ensure positive values
        self._eigenvector = np.abs(x)
        self._eigenvector = self._eigenvector / (self._eigenvector.sum() + 1e-8)
        return self._eigenvector

    def compute_composite_score(self) -> np.ndarray:
        """
        Compute composite centrality score.

        S(v) = α·PageRank(v) + β·Betweenness(v) + γ·Eigenvector(v)

        Returns:
            Composite scores [V]
        """
        pr = self.compute_pagerank()
        betw = self.compute_betweenness()
        eigen = self.compute_eigenvector_centrality()

        # Normalize each to [0, 1]
        pr_norm = (pr - pr.min()) / (pr.max() - pr.min() + 1e-8)
        betw_norm = (betw - betw.min()) / (betw.max() - betw.min() + 1e-8)
        eigen_norm = (eigen - eigen.min()) / (eigen.max() - eigen.min() + 1e-8)

        # Weighted combination
        composite = (
            self.centrality_weights['pagerank'] * pr_norm +
            self.centrality_weights['betweenness'] * betw_norm +
            self.centrality_weights['eigenvector'] * eigen_norm
        )

        return composite

    def compute_message_passing_influence(
        self,
        k_hops: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k-hop message passing influence for each node.

        Args:
            k_hops: Number of hops to consider

        Returns:
            Tuple of (influence_range, decay_rate) arrays
        """
        # Compute k-hop adjacency
        adj_power = np.eye(self.num_nodes)
        influence_range = np.zeros(self.num_nodes)

        for k in range(1, k_hops + 1):
            adj_power = adj_power @ self.adj_matrix
            # Count reachable nodes at hop k
            reachable = (adj_power > 0).sum(axis=1)
            influence_range += reachable / (k ** 2)  # Weight by inverse square of distance

        # Compute decay rate (how quickly influence diminishes)
        adj_1 = self.adj_matrix > 0
        adj_k = np.linalg.matrix_power(adj_1.astype(float), k_hops) > 0

        direct_neighbors = adj_1.sum(axis=1)
        k_hop_neighbors = adj_k.sum(axis=1)

        decay_rate = k_hop_neighbors / (direct_neighbors * k_hops + 1e-8)
        decay_rate = np.clip(decay_rate, 0, 1)

        return influence_range, decay_rate

    def select_trigger_nodes(
        self,
        num_nodes: int,
        target_nodes: Optional[List[int]] = None,
        strategy: str = "centrality",
        exclude_nodes: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select trigger nodes for backdoor injection.

        Args:
            num_nodes: Number of nodes to select
            target_nodes: Target attack region nodes (optional)
            strategy: Selection strategy ("centrality", "random", "target_proximity")
            exclude_nodes: Nodes to exclude from selection

        Returns:
            List of selected node indices
        """
        exclude_set = set(exclude_nodes) if exclude_nodes else set()
        num_nodes = min(num_nodes, self.num_nodes - len(exclude_set))

        if strategy == "random":
            candidates = [i for i in range(self.num_nodes) if i not in exclude_set]
            selected = np.random.choice(candidates, size=num_nodes, replace=False)
            return selected.tolist()

        elif strategy == "centrality":
            scores = self.compute_composite_score()

            # Zero out excluded nodes
            for node in exclude_set:
                scores[node] = -np.inf

            # Select top nodes
            selected = np.argsort(scores)[-num_nodes:][::-1]
            return selected.tolist()

        elif strategy == "target_proximity":
            if target_nodes is None:
                return self.select_trigger_nodes(num_nodes, strategy="centrality")

            # Find bridge nodes that can reach targets
            scores = self.compute_composite_score()

            # Compute distance to target nodes
            sparse_adj = csr_matrix(self.adj_matrix)
            dist_matrix = shortest_path(sparse_adj, directed=False)

            # Score nodes by: centrality * (1 / avg_distance_to_targets)
            target_distances = dist_matrix[:, target_nodes].mean(axis=1)
            target_distances = np.where(np.isinf(target_distances), self.num_nodes, target_distances)

            proximity_scores = scores * (1 / (target_distances + 1))

            for node in exclude_set:
                proximity_scores[node] = -np.inf
            for node in target_nodes:
                proximity_scores[node] = -np.inf  # Don't select target nodes as triggers

            selected = np.argsort(proximity_scores)[-num_nodes:][::-1]
            return selected.tolist()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_node_neighbors(self, node_idx: int, k_hops: int = 1) -> List[int]:
        """Get k-hop neighbors of a node."""
        adj_power = self.adj_matrix.copy()
        for _ in range(k_hops - 1):
            adj_power = adj_power @ self.adj_matrix

        neighbors = np.where(adj_power[node_idx] > 0)[0]
        return neighbors.tolist()

    def visualize_centralities(self) -> Dict[str, np.ndarray]:
        """Return all computed centralities for visualization."""
        return {
            'pagerank': self.compute_pagerank(),
            'betweenness': self.compute_betweenness(),
            'eigenvector': self.compute_eigenvector_centrality(),
            'degree': self.compute_degree_centrality(),
            'composite': self.compute_composite_score()
        }


def build_grid_adjacency(
    row: int,
    col: int,
    connectivity: int = 8
) -> np.ndarray:
    """
    Build adjacency matrix for a grid graph.

    Args:
        row: Number of rows
        col: Number of columns
        connectivity: 4 (cardinal) or 8 (cardinal + diagonal)

    Returns:
        Adjacency matrix [row*col, row*col]
    """
    area_num = row * col
    adj = np.zeros((area_num, area_num))

    for i in range(row):
        for j in range(col):
            idx = i * col + j

            # Cardinal neighbors (4-connectivity)
            if i > 0:
                adj[idx, (i - 1) * col + j] = 1
            if i < row - 1:
                adj[idx, (i + 1) * col + j] = 1
            if j > 0:
                adj[idx, i * col + (j - 1)] = 1
            if j < col - 1:
                adj[idx, i * col + (j + 1)] = 1

            # Diagonal neighbors (8-connectivity)
            if connectivity == 8:
                if i > 0 and j > 0:
                    adj[idx, (i - 1) * col + (j - 1)] = 1
                if i > 0 and j < col - 1:
                    adj[idx, (i - 1) * col + (j + 1)] = 1
                if i < row - 1 and j > 0:
                    adj[idx, (i + 1) * col + (j - 1)] = 1
                if i < row - 1 and j < col - 1:
                    adj[idx, (i + 1) * col + (j + 1)] = 1

    return adj


if __name__ == "__main__":
    print("Testing Node Selection utilities...")

    # Create grid adjacency
    row, col = 16, 16
    adj = build_grid_adjacency(row, col, connectivity=8)
    print(f"Grid adjacency shape: {adj.shape}")
    print(f"Number of edges: {adj.sum()}")

    # Test NodeSelector
    selector = NodeSelector(adj)

    # Compute centralities
    print("\nComputing centralities...")
    pr = selector.compute_pagerank()
    print(f"PageRank: min={pr.min():.4f}, max={pr.max():.4f}")

    betw = selector.compute_betweenness()
    print(f"Betweenness: min={betw.min():.4f}, max={betw.max():.4f}")

    eigen = selector.compute_eigenvector_centrality()
    print(f"Eigenvector: min={eigen.min():.4f}, max={eigen.max():.4f}")

    # Select trigger nodes
    print("\nSelecting trigger nodes...")
    trigger_nodes = selector.select_trigger_nodes(5, strategy="centrality")
    print(f"Selected trigger nodes (centrality): {trigger_nodes}")

    # With target nodes
    target_nodes = [100, 101, 102]  # Some target region
    trigger_nodes_prox = selector.select_trigger_nodes(5, target_nodes=target_nodes, strategy="target_proximity")
    print(f"Selected trigger nodes (proximity to targets): {trigger_nodes_prox}")

    print("\nNode selection test completed!")
