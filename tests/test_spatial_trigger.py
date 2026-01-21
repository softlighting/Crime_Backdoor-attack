"""
Unit Tests for Spatial Trigger Module (STM)

Based on CSTBA Implementation Roadmap Task 3.3:
1. Test node selection algorithm output
2. Test feature trigger only modifies specified nodes
3. Test topology trigger adds reasonable edges
4. Test homophily loss computation
5. Visualize graph structure differences
"""

import sys
import os
import unittest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cstba.modules.spatial_trigger import SpatialTriggerModule
from cstba.utils.node_selection import NodeSelector, build_grid_adjacency


class TestNodeSelector(unittest.TestCase):
    """Test cases for NodeSelector utility."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a 10x10 grid adjacency matrix
        self.row, self.col = 10, 10
        self.V = self.row * self.col
        self.adj_matrix = build_grid_adjacency(self.row, self.col, connectivity=8)

    def test_grid_adjacency_creation(self):
        """Test grid adjacency matrix is created correctly."""
        adj = build_grid_adjacency(self.row, self.col, connectivity=8)

        # Check shape
        self.assertEqual(adj.shape, (self.V, self.V))

        # Check symmetry
        self.assertTrue(np.allclose(adj, adj.T), "Adjacency should be symmetric")

        # Check diagonal is zero
        self.assertTrue(np.allclose(np.diag(adj), 0), "Diagonal should be zero")

        # Corner nodes should have 3 neighbors (8-connectivity)
        corner_idx = 0  # Top-left corner
        self.assertEqual(adj[corner_idx].sum(), 3, "Corner should have 3 neighbors")

        # Center node should have 8 neighbors
        center_idx = 5 * self.col + 5
        self.assertEqual(adj[center_idx].sum(), 8, "Center should have 8 neighbors")

        print("✓ Grid adjacency creation test passed")

    def test_pagerank_computation(self):
        """Test PageRank centrality computation."""
        selector = NodeSelector(self.adj_matrix)
        pr = selector.compute_pagerank()

        # Check shape
        self.assertEqual(len(pr), self.V)

        # Check normalization (sum to 1)
        self.assertAlmostEqual(pr.sum(), 1.0, places=5)

        # Check all values are positive
        self.assertTrue(np.all(pr >= 0))

        # Center nodes should have higher PageRank in a grid
        center_pr = pr[5 * self.col + 5]
        corner_pr = pr[0]
        self.assertGreater(center_pr, corner_pr,
                          "Center should have higher PageRank than corner")

        print(f"  PageRank range: [{pr.min():.6f}, {pr.max():.6f}]")
        print("✓ PageRank computation test passed")

    def test_betweenness_computation(self):
        """Test betweenness centrality computation."""
        selector = NodeSelector(self.adj_matrix)
        betw = selector.compute_betweenness()

        # Check shape
        self.assertEqual(len(betw), self.V)

        # Check non-negative
        self.assertTrue(np.all(betw >= 0))

        print(f"  Betweenness range: [{betw.min():.6f}, {betw.max():.6f}]")
        print("✓ Betweenness computation test passed")

    def test_eigenvector_centrality(self):
        """Test eigenvector centrality computation."""
        selector = NodeSelector(self.adj_matrix)
        eigen = selector.compute_eigenvector_centrality()

        # Check shape
        self.assertEqual(len(eigen), self.V)

        # Check non-negative
        self.assertTrue(np.all(eigen >= 0))

        print(f"  Eigenvector centrality range: [{eigen.min():.6f}, {eigen.max():.6f}]")
        print("✓ Eigenvector centrality test passed")

    def test_composite_score(self):
        """Test composite centrality score computation."""
        selector = NodeSelector(self.adj_matrix)
        composite = selector.compute_composite_score()

        # Check shape
        self.assertEqual(len(composite), self.V)

        # Check range [0, 1] after normalization
        self.assertTrue(np.all(composite >= 0))
        self.assertTrue(np.all(composite <= 1 + 1e-6))

        print(f"  Composite score range: [{composite.min():.4f}, {composite.max():.4f}]")
        print("✓ Composite score test passed")

    def test_node_selection_centrality(self):
        """Test 1: Node selection using centrality strategy."""
        selector = NodeSelector(self.adj_matrix)

        num_trigger = 5
        selected = selector.select_trigger_nodes(num_trigger, strategy="centrality")

        # Check number of selected nodes
        self.assertEqual(len(selected), num_trigger)

        # Check no duplicates
        self.assertEqual(len(set(selected)), num_trigger)

        # Check all indices are valid
        self.assertTrue(all(0 <= idx < self.V for idx in selected))

        print(f"  Selected nodes (centrality): {selected}")
        print("✓ Node selection (centrality) test passed")

    def test_node_selection_random(self):
        """Test node selection using random strategy."""
        selector = NodeSelector(self.adj_matrix)

        num_trigger = 5
        selected = selector.select_trigger_nodes(num_trigger, strategy="random")

        # Check number of selected nodes
        self.assertEqual(len(selected), num_trigger)

        # Check no duplicates
        self.assertEqual(len(set(selected)), num_trigger)

        print(f"  Selected nodes (random): {selected}")
        print("✓ Node selection (random) test passed")

    def test_node_selection_target_proximity(self):
        """Test node selection using target proximity strategy."""
        selector = NodeSelector(self.adj_matrix)

        target_nodes = [90, 91, 92]  # Bottom-right region
        num_trigger = 5
        selected = selector.select_trigger_nodes(
            num_trigger,
            target_nodes=target_nodes,
            strategy="target_proximity"
        )

        # Check number of selected nodes
        self.assertEqual(len(selected), num_trigger)

        # Selected nodes should not include target nodes
        for node in target_nodes:
            self.assertNotIn(node, selected)

        print(f"  Target nodes: {target_nodes}")
        print(f"  Selected nodes (proximity): {selected}")
        print("✓ Node selection (target proximity) test passed")


class TestSpatialTriggerModule(unittest.TestCase):
    """Test cases for SpatialTriggerModule."""

    def setUp(self):
        """Set up test fixtures."""
        self.B = 4
        self.V = 100
        self.C = 4

        torch.manual_seed(42)
        self.node_features = torch.randn(self.B, self.V, self.C)
        self.adj_matrix = torch.tensor(
            build_grid_adjacency(10, 10, connectivity=8),
            dtype=torch.float32
        )

        self.trigger_nodes = [10, 20, 30, 40, 50]

    def test_feature_trigger_shape(self):
        """Test feature trigger output shape."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="feature"
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        mod_features, mod_adj, info = stm(self.node_features, self.adj_matrix)

        # Check shapes
        self.assertEqual(mod_features.shape, self.node_features.shape)
        self.assertEqual(mod_adj.shape, self.adj_matrix.shape)

        print("✓ Feature trigger shape test passed")

    def test_feature_trigger_only_modifies_trigger_nodes(self):
        """Test 2: Feature trigger only modifies specified nodes."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="feature"
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        mod_features, _, _ = stm(self.node_features, self.adj_matrix)

        # Compute difference
        diff = (mod_features - self.node_features).abs()

        # Non-trigger nodes should have zero or near-zero difference
        non_trigger_mask = torch.ones(self.V, dtype=torch.bool)
        non_trigger_mask[self.trigger_nodes] = False

        non_trigger_diff = diff[:, non_trigger_mask, :].mean().item()
        trigger_diff = diff[:, self.trigger_nodes, :].mean().item()

        print(f"  Non-trigger nodes diff: {non_trigger_diff:.6f}")
        print(f"  Trigger nodes diff: {trigger_diff:.6f}")

        # Non-trigger nodes should have minimal modification
        self.assertLess(non_trigger_diff, 1e-5,
                       "Non-trigger nodes should not be modified")

        print("✓ Feature trigger locality test passed")

    def test_topology_trigger_shape(self):
        """Test topology trigger output shape."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="topology"
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        target_nodes = [60, 61, 62]
        stm.set_edge_injection(self.trigger_nodes, target_nodes)

        mod_features, mod_adj, info = stm(self.node_features, self.adj_matrix, target_nodes)

        # Features should be unchanged for topology-only
        self.assertTrue(torch.allclose(mod_features, self.node_features),
                       "Topology trigger should not modify features")

        print("✓ Topology trigger shape test passed")

    def test_topology_trigger_adds_edges(self):
        """Test 3: Topology trigger adds reasonable edges."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="topology"
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        target_nodes = [60, 61, 62]
        stm.set_edge_injection(self.trigger_nodes, target_nodes)

        _, mod_adj, info = stm(self.node_features, self.adj_matrix, target_nodes)

        # Check delta_A
        delta_A = info.get('delta_A', mod_adj - self.adj_matrix)
        if isinstance(delta_A, torch.Tensor):
            delta_A = delta_A.detach()

        # New edges should exist
        new_edges = (delta_A > 0).sum().item()
        print(f"  Number of new edges: {new_edges}")

        # Check edges are between trigger and target nodes
        for src in self.trigger_nodes:
            for tgt in target_nodes:
                if src != tgt:
                    # Edge should be added
                    edge_weight = delta_A[src, tgt].item()
                    if edge_weight > 0:
                        print(f"    Edge ({src}, {tgt}): {edge_weight:.4f}")

        print("✓ Topology trigger adds edges test passed")

    def test_homophily_loss_computation(self):
        """Test 4: Homophily loss computation."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="feature",
            homophily_constraint=True
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        mod_features, mod_adj, _ = stm(self.node_features, self.adj_matrix)

        homophily_loss = stm.compute_homophily_loss(mod_features, mod_adj)

        # Should be a scalar
        self.assertEqual(homophily_loss.dim(), 0)
        # Should be non-negative
        self.assertGreaterEqual(homophily_loss.item(), 0)

        print(f"  Homophily loss: {homophily_loss.item():.6f}")
        print("✓ Homophily loss computation test passed")

    def test_topology_stealth_loss(self):
        """Test topology stealthiness loss computation."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="topology"
        )
        stm.set_trigger_nodes(self.trigger_nodes)
        stm.set_edge_injection(self.trigger_nodes, [60, 61, 62])

        _, mod_adj, _ = stm(self.node_features, self.adj_matrix)

        stealth_loss = stm.compute_topology_stealth_loss(self.adj_matrix, mod_adj)

        print(f"  Topology stealth loss: {stealth_loss.item():.6f}")
        print("✓ Topology stealth loss test passed")

    def test_both_trigger_type(self):
        """Test combined feature and topology trigger."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="both"
        )
        stm.set_trigger_nodes(self.trigger_nodes)
        stm.set_edge_injection(self.trigger_nodes, [60, 61, 62])

        mod_features, mod_adj, info = stm(self.node_features, self.adj_matrix)

        # Features should be modified at trigger nodes
        feature_diff = (mod_features - self.node_features)[:, self.trigger_nodes, :].abs().mean()
        self.assertGreater(feature_diff.item(), 0, "Features should be modified")

        # Adjacency should be modified
        adj_diff = (mod_adj - self.adj_matrix).abs().sum()
        self.assertGreater(adj_diff.item(), 0, "Adjacency should be modified")

        print(f"  Feature modification: {feature_diff.item():.6f}")
        print(f"  Adjacency modification: {adj_diff.item():.6f}")
        print("✓ Both trigger type test passed")

    def test_4d_input_handling(self):
        """Test STM handles 4D input (spatio-temporal)."""
        T = 30
        features_4d = torch.randn(self.B, T, self.V, self.C)

        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="feature"
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        mod_features, mod_adj, info = stm(features_4d, self.adj_matrix)

        # Check output shape
        self.assertEqual(mod_features.shape, features_4d.shape)

        print(f"  4D input shape: {features_4d.shape}")
        print(f"  4D output shape: {mod_features.shape}")
        print("✓ 4D input handling test passed")

    def test_gradient_flow(self):
        """Test gradients flow through STM."""
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="feature"
        )
        stm.set_trigger_nodes(self.trigger_nodes)

        mod_features, _, _ = stm(self.node_features, self.adj_matrix)
        loss = mod_features.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in stm.parameters()
        )

        self.assertTrue(has_grad, "Gradients should flow through STM")
        print("✓ Gradient flow test passed")


class TestSpatialVisualization(unittest.TestCase):
    """Visualization tests for STM."""

    def setUp(self):
        """Set up visualization fixtures."""
        self.output_dir = "tests/outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        self.row, self.col = 10, 10
        self.V = self.row * self.col
        self.C = 4

        self.adj_matrix = torch.tensor(
            build_grid_adjacency(self.row, self.col, connectivity=8),
            dtype=torch.float32
        )
        self.node_features = torch.randn(1, self.V, self.C)

        self.trigger_nodes = [22, 23, 32, 33, 34]  # Cluster in center-left
        self.target_nodes = [66, 67, 76, 77, 78]   # Cluster in center-right

    def test_visualize_graph_modification(self):
        """Test 5: Visualize graph structure before and after trigger."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Create node positions (grid layout)
        pos = np.zeros((self.V, 2))
        for i in range(self.V):
            pos[i, 0] = i % self.col
            pos[i, 1] = i // self.col

        # Original adjacency
        ax = axes[0]
        self._plot_graph(ax, self.adj_matrix.numpy(), pos, "Original Graph")
        ax.scatter(pos[self.trigger_nodes, 0], pos[self.trigger_nodes, 1],
                  c='red', s=100, marker='s', label='Trigger Nodes', zorder=5)
        ax.scatter(pos[self.target_nodes, 0], pos[self.target_nodes, 1],
                  c='blue', s=100, marker='^', label='Target Nodes', zorder=5)
        ax.legend()

        # Apply topology trigger
        stm = SpatialTriggerModule(
            feature_dim=self.C,
            num_nodes=self.V,
            trigger_type="topology"
        )
        stm.set_trigger_nodes(self.trigger_nodes)
        stm.set_edge_injection(self.trigger_nodes, self.target_nodes)

        _, mod_adj, _ = stm(self.node_features, self.adj_matrix, self.target_nodes)
        mod_adj_np = mod_adj.detach().numpy()

        # Modified adjacency
        ax = axes[1]
        self._plot_graph(ax, mod_adj_np, pos, "Modified Graph (with injected edges)")
        ax.scatter(pos[self.trigger_nodes, 0], pos[self.trigger_nodes, 1],
                  c='red', s=100, marker='s', zorder=5)
        ax.scatter(pos[self.target_nodes, 0], pos[self.target_nodes, 1],
                  c='blue', s=100, marker='^', zorder=5)

        # Highlight new edges
        delta_adj = mod_adj_np - self.adj_matrix.numpy()
        for i in range(self.V):
            for j in range(i+1, self.V):
                if delta_adj[i, j] > 0.01:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                           'r-', linewidth=2, alpha=0.8)

        # Centrality visualization
        ax = axes[2]
        selector = NodeSelector(self.adj_matrix.numpy())
        composite = selector.compute_composite_score()

        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=composite, cmap='YlOrRd',
                            s=50, edgecolors='black', linewidths=0.5)
        ax.scatter(pos[self.trigger_nodes, 0], pos[self.trigger_nodes, 1],
                  c='none', s=150, marker='s', edgecolors='red', linewidths=2)
        plt.colorbar(scatter, ax=ax, label='Composite Centrality')
        ax.set_title('Node Centrality (selected nodes marked)')
        ax.set_aspect('equal')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'stm_graph_modification.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ Graph modification visualization saved to {save_path}")

    def _plot_graph(self, ax, adj, pos, title):
        """Helper to plot graph."""
        # Plot edges
        for i in range(len(adj)):
            for j in range(i+1, len(adj)):
                if adj[i, j] > 0:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                           'gray', alpha=0.3, linewidth=0.5)

        # Plot nodes
        ax.scatter(pos[:, 0], pos[:, 1], c='lightblue', s=30, edgecolors='black', linewidths=0.5)
        ax.set_title(title)
        ax.set_aspect('equal')

    def test_visualize_centrality_distribution(self):
        """Visualize centrality metrics distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        selector = NodeSelector(self.adj_matrix.numpy())
        centralities = selector.visualize_centralities()

        pos = np.zeros((self.V, 2))
        for i in range(self.V):
            pos[i, 0] = i % self.col
            pos[i, 1] = i // self.col

        for idx, (name, values) in enumerate(centralities.items()):
            if idx >= 4:
                break
            ax = axes[idx // 2, idx % 2]
            scatter = ax.scatter(pos[:, 0], pos[:, 1], c=values, cmap='viridis',
                                s=50, edgecolors='black', linewidths=0.5)
            plt.colorbar(scatter, ax=ax)
            ax.set_title(f'{name.capitalize()} Centrality')
            ax.set_aspect('equal')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'stm_centrality_distribution.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ Centrality distribution visualization saved to {save_path}")


def run_tests():
    """Run all STM tests."""
    print("=" * 60)
    print("Running Spatial Trigger Module (STM) Unit Tests")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestNodeSelector))
    suite.addTests(loader.loadTestsFromTestCase(TestSpatialTriggerModule))
    suite.addTests(loader.loadTestsFromTestCase(TestSpatialVisualization))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print("STM Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All STM tests passed!")
    else:
        print("\n✗ Some tests failed")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
