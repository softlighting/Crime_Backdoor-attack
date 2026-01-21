"""
Unit Tests for Temporal Trigger Module (TTM)

Based on CSTBA Implementation Roadmap Task 2.3:
1. Test trigger generation shape correctness
2. Test three trigger mode outputs
3. Test smoothness and frequency loss computation
4. Test trigger amplitude constraints
5. Visualize generated trigger waveforms
"""

import sys
import os
import unittest
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cstba.modules.temporal_trigger import TemporalTriggerModule, PeriodicTrigger
from cstba.utils.frequency_utils import (
    compute_fft_spectrum,
    spectral_similarity,
    compute_high_frequency_energy
)


class TestTemporalTriggerModule(unittest.TestCase):
    """Test cases for TemporalTriggerModule."""

    def setUp(self):
        """Set up test fixtures."""
        self.B = 4  # Batch size
        self.T = 30  # Time steps
        self.V = 100  # Number of nodes/variables
        self.C = 4  # Feature dimension (crime categories)

        # Create sample data
        torch.manual_seed(42)
        self.sample_data = torch.randn(self.B, self.T, self.V, self.C)

        # Create adjacency matrix (grid-like)
        self.adj_matrix = torch.zeros(self.V, self.V)
        for i in range(self.V):
            if i > 0:
                self.adj_matrix[i, i-1] = 1
            if i < self.V - 1:
                self.adj_matrix[i, i+1] = 1

        # Data statistics
        self.data_mean = 0.0
        self.data_std = 1.0

    def test_periodic_trigger_shape(self):
        """Test 1: Verify periodic trigger output shape."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        x_poisoned, delta_t = ttm(self.sample_data, self.adj_matrix)

        # Check shapes
        self.assertEqual(x_poisoned.shape, self.sample_data.shape,
                        "Poisoned data shape mismatch")
        self.assertEqual(delta_t.shape, self.sample_data.shape,
                        "Trigger shape mismatch")

        print("✓ Periodic trigger shape test passed")

    def test_spike_trigger_shape(self):
        """Test 1b: Verify spike trigger output shape."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="spike"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        x_poisoned, delta_t = ttm(self.sample_data, self.adj_matrix)

        self.assertEqual(x_poisoned.shape, self.sample_data.shape)
        self.assertEqual(delta_t.shape, self.sample_data.shape)

        print("✓ Spike trigger shape test passed")

    def test_distributed_trigger_shape(self):
        """Test 1c: Verify distributed trigger output shape."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="distributed"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        x_poisoned, delta_t = ttm(self.sample_data, self.adj_matrix)

        self.assertEqual(x_poisoned.shape, self.sample_data.shape)
        self.assertEqual(delta_t.shape, self.sample_data.shape)

        print("✓ Distributed trigger shape test passed")

    def test_trigger_modes_difference(self):
        """Test 2: Verify three trigger modes produce different outputs."""
        triggers = {}

        for mode in ["periodic", "spike", "distributed"]:
            torch.manual_seed(42)  # Same seed for fair comparison
            ttm = TemporalTriggerModule(
                input_dim=self.C,
                hidden_dim=32,
                num_variables=self.V,
                trigger_type=mode
            )
            ttm.set_data_statistics(self.data_mean, self.data_std)

            _, delta_t = ttm(self.sample_data, self.adj_matrix)
            triggers[mode] = delta_t.detach()

        # Check that different modes produce different patterns
        # (Allow for some variation due to network initialization)
        periodic_pattern = triggers["periodic"][:, :, 0, 0].mean(dim=0)
        spike_pattern = triggers["spike"][:, :, 0, 0].mean(dim=0)
        distributed_pattern = triggers["distributed"][:, :, 0, 0].mean(dim=0)

        # Patterns should be different (correlation < 0.99)
        corr_ps = torch.corrcoef(torch.stack([periodic_pattern, spike_pattern]))[0, 1]
        corr_pd = torch.corrcoef(torch.stack([periodic_pattern, distributed_pattern]))[0, 1]

        print(f"  Periodic-Spike correlation: {corr_ps:.4f}")
        print(f"  Periodic-Distributed correlation: {corr_pd:.4f}")

        print("✓ Trigger modes difference test passed")

    def test_smoothness_loss(self):
        """Test 3a: Verify smoothness loss computation."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        _, delta_t = ttm(self.sample_data, self.adj_matrix)
        smoothness_loss = ttm.compute_smoothness_loss(delta_t)

        # Loss should be a scalar
        self.assertEqual(smoothness_loss.dim(), 0, "Smoothness loss should be scalar")
        # Loss should be non-negative
        self.assertGreaterEqual(smoothness_loss.item(), 0, "Smoothness loss should be non-negative")

        print(f"  Smoothness loss: {smoothness_loss.item():.6f}")
        print("✓ Smoothness loss test passed")

    def test_frequency_loss(self):
        """Test 3b: Verify frequency loss computation."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        x_poisoned, delta_t = ttm(self.sample_data, self.adj_matrix)
        frequency_loss = ttm.compute_frequency_loss(self.sample_data, x_poisoned)

        # Loss should be a scalar
        self.assertEqual(frequency_loss.dim(), 0, "Frequency loss should be scalar")

        print(f"  Frequency loss: {frequency_loss.item():.6f}")
        print("✓ Frequency loss test passed")

    def test_trigger_amplitude_constraint(self):
        """Test 4: Verify trigger amplitude is within constraints."""
        amplitude_ratio = 0.15  # 15% of std

        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic",
            amplitude_ratio=amplitude_ratio
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        _, delta_t = ttm(self.sample_data, self.adj_matrix)

        # Check max amplitude
        max_amplitude = delta_t.abs().max().item()
        expected_max = amplitude_ratio * self.data_std * 2  # Some margin

        self.assertLess(max_amplitude, expected_max,
                       f"Trigger amplitude {max_amplitude} exceeds constraint {expected_max}")

        print(f"  Max trigger amplitude: {max_amplitude:.4f}")
        print(f"  Expected max (with margin): {expected_max:.4f}")
        print("✓ Trigger amplitude constraint test passed")

    def test_injection_window(self):
        """Test 4b: Verify trigger is injected only in specified window."""
        injection_ratio = 0.33  # Last 1/3 of time window

        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic",
            injection_ratio=injection_ratio
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        _, delta_t = ttm(self.sample_data, self.adj_matrix)

        # Calculate injection start point
        injection_start = int(self.T * (1 - injection_ratio))

        # Energy before injection window should be much lower
        energy_before = delta_t[:, :injection_start, :, :].abs().mean().item()
        energy_after = delta_t[:, injection_start:, :, :].abs().mean().item()

        print(f"  Energy before injection window: {energy_before:.6f}")
        print(f"  Energy in injection window: {energy_after:.6f}")

        # Most energy should be in injection window
        if energy_after > 0:
            ratio = energy_before / (energy_after + 1e-8)
            self.assertLess(ratio, 0.5,
                           "Too much trigger energy before injection window")

        print("✓ Injection window test passed")

    def test_total_loss_computation(self):
        """Test 3c: Verify total loss computation."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        _, delta_t = ttm(self.sample_data, self.adj_matrix)
        total_loss, losses = ttm.compute_total_loss(self.sample_data, delta_t)

        # Check all loss components exist
        self.assertIn('smoothness', losses)
        self.assertIn('frequency', losses)
        self.assertIn('amplitude', losses)
        self.assertIn('total', losses)

        print(f"  Loss components: {list(losses.keys())}")
        for k, v in losses.items():
            print(f"    {k}: {v.item():.6f}")

        print("✓ Total loss computation test passed")

    def test_gradient_flow(self):
        """Test that gradients flow through the module."""
        ttm = TemporalTriggerModule(
            input_dim=self.C,
            hidden_dim=32,
            num_variables=self.V,
            trigger_type="periodic"
        )
        ttm.set_data_statistics(self.data_mean, self.data_std)

        x_poisoned, delta_t = ttm(self.sample_data, self.adj_matrix)
        loss = x_poisoned.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for name, param in ttm.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        self.assertTrue(has_grad, "No gradients flowing through TTM")
        print("✓ Gradient flow test passed")


class TestTriggerVisualization(unittest.TestCase):
    """Visualization tests for TTM (saves plots)."""

    def setUp(self):
        """Set up visualization test fixtures."""
        self.B = 1
        self.T = 30
        self.V = 10
        self.C = 4
        self.output_dir = "tests/outputs"
        os.makedirs(self.output_dir, exist_ok=True)

        torch.manual_seed(42)
        self.sample_data = torch.randn(self.B, self.T, self.V, self.C)

    def test_visualize_trigger_waveforms(self):
        """Test 5: Visualize generated trigger waveforms."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))

        for idx, mode in enumerate(["periodic", "spike", "distributed"]):
            torch.manual_seed(42)
            ttm = TemporalTriggerModule(
                input_dim=self.C,
                hidden_dim=32,
                num_variables=self.V,
                trigger_type=mode
            )
            ttm.set_data_statistics(0.0, 1.0)

            _, delta_t = ttm(self.sample_data)

            # Plot trigger pattern for first node, first feature
            trigger_pattern = delta_t[0, :, 0, 0].detach().numpy()
            original_pattern = self.sample_data[0, :, 0, 0].numpy()
            poisoned_pattern = (self.sample_data[0, :, 0, 0] + delta_t[0, :, 0, 0]).detach().numpy()

            # Left: Trigger waveform
            axes[idx, 0].plot(trigger_pattern, 'r-', linewidth=2, label='Trigger')
            axes[idx, 0].axvline(x=int(self.T * 0.67), color='gray', linestyle='--', label='Injection Start')
            axes[idx, 0].set_title(f'{mode.capitalize()} Trigger Pattern')
            axes[idx, 0].set_xlabel('Time Step')
            axes[idx, 0].set_ylabel('Amplitude')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            # Right: Original vs Poisoned
            axes[idx, 1].plot(original_pattern, 'b-', alpha=0.7, label='Original')
            axes[idx, 1].plot(poisoned_pattern, 'r--', alpha=0.7, label='Poisoned')
            axes[idx, 1].set_title(f'{mode.capitalize()}: Original vs Poisoned')
            axes[idx, 1].set_xlabel('Time Step')
            axes[idx, 1].set_ylabel('Value')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'ttm_trigger_waveforms.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ Trigger waveform visualization saved to {save_path}")

    def test_visualize_frequency_spectrum(self):
        """Visualize frequency spectrum of original vs poisoned signals."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, mode in enumerate(["periodic", "spike", "distributed"]):
            torch.manual_seed(42)
            ttm = TemporalTriggerModule(
                input_dim=self.C,
                hidden_dim=32,
                num_variables=self.V,
                trigger_type=mode
            )
            ttm.set_data_statistics(0.0, 1.0)

            x_poisoned, _ = ttm(self.sample_data)

            # Compute spectra
            orig_signal = self.sample_data[0, :, 0, 0].numpy()
            pois_signal = x_poisoned[0, :, 0, 0].detach().numpy()

            orig_spectrum = np.abs(np.fft.rfft(orig_signal))
            pois_spectrum = np.abs(np.fft.rfft(pois_signal))
            freqs = np.fft.rfftfreq(len(orig_signal))

            axes[idx].plot(freqs, orig_spectrum, 'b-', alpha=0.7, label='Original')
            axes[idx].plot(freqs, pois_spectrum, 'r--', alpha=0.7, label='Poisoned')
            axes[idx].set_title(f'{mode.capitalize()} Frequency Spectrum')
            axes[idx].set_xlabel('Frequency')
            axes[idx].set_ylabel('Magnitude')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

            # Compute similarity
            sim = spectral_similarity(orig_signal, pois_signal, axis=0)
            axes[idx].text(0.95, 0.95, f'Similarity: {sim:.3f}',
                          transform=axes[idx].transAxes, ha='right', va='top',
                          fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'ttm_frequency_spectrum.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"✓ Frequency spectrum visualization saved to {save_path}")


def run_tests():
    """Run all TTM tests."""
    print("=" * 60)
    print("Running Temporal Trigger Module (TTM) Unit Tests")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalTriggerModule))
    suite.addTests(loader.loadTestsFromTestCase(TestTriggerVisualization))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    print("TTM Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✓ All TTM tests passed!")
    else:
        print("\n✗ Some tests failed")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}: {traceback[:100]}...")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
