"""
Temporal Trigger Module (TTM)

Generates stealthy temporal trigger patterns for ST-GNN backdoor attacks.

Core Features:
1. GAT-based trigger generator modeling multi-variable coupling
2. Three trigger modes: periodic, spike, distributed
3. Shape-aware normalization for frequency domain stealth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Literal


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for modeling inter-variable dependencies.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Learnable parameters
        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(
        self,
        h: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h: Node features [B, V, F] or [V, F]
            adj: Adjacency matrix [V, V] (optional, uses full attention if None)

        Returns:
            Updated node features [B, V, F'] or [V, F']
        """
        # Handle batched input
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, V, _ = h.shape

        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [B, V, F']

        # Compute attention coefficients
        a_input = self._prepare_attention_input(Wh)  # [B, V, V, 2F']
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))  # [B, V, V]

        # Mask with adjacency if provided
        if adj is not None:
            # Expand adj for batch dimension if needed
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)
            # Mask non-adjacent nodes with large negative value
            e = e.masked_fill(adj == 0, float('-inf'))

        # Softmax attention
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Handle NaN from softmax on all -inf rows
        attention = torch.nan_to_num(attention, nan=0.0)

        # Apply attention
        h_prime = torch.bmm(attention, Wh)  # [B, V, F']

        if squeeze_output:
            h_prime = h_prime.squeeze(0)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attention_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """Prepare input for attention computation."""
        B, V, F = Wh.shape

        # Create all pairs of node features
        Wh_repeated_in_chunks = Wh.repeat_interleave(V, dim=1)  # [B, V*V, F]
        Wh_repeated_alternating = Wh.repeat(1, V, 1)  # [B, V*V, F]

        # Concatenate
        all_combinations = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        return all_combinations.view(B, V, V, 2 * F)


class TemporalTriggerModule(nn.Module):
    """
    Temporal Trigger Module (TTM)

    Generates stealthy temporal trigger patterns for backdoor attacks on ST-GNNs.

    Core Capabilities:
    1. GAT-based generator modeling multi-variable coupling relationships
    2. Support for three trigger modes: periodic, spike, distributed
    3. Shape-aware normalization loss ensuring frequency domain stealth
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_variables: int,
        trigger_type: Literal["periodic", "spike", "distributed"] = "periodic",
        injection_ratio: float = 0.33,
        amplitude_ratio: float = 0.1,
        num_gat_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension C
            hidden_dim: Hidden layer dimension
            num_variables: Number of variables V (nodes)
            trigger_type: Type of temporal trigger pattern
            injection_ratio: Portion of time window to inject trigger (last X%)
            amplitude_ratio: Trigger amplitude as ratio of data std
            num_gat_heads: Number of GAT attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_variables = num_variables
        self.trigger_type = trigger_type
        self.injection_ratio = injection_ratio
        self.amplitude_ratio = amplitude_ratio

        # Multi-head GAT for modeling variable dependencies
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout)
            for _ in range(num_gat_heads)
        ])

        # Temporal encoder (capture temporal patterns)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim * num_gat_heads,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Trigger pattern generator
        self.trigger_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Trigger type-specific parameters
        if trigger_type == "periodic":
            # Learnable frequencies and amplitudes for periodic trigger
            self.num_frequencies = 3
            self.frequencies = nn.Parameter(torch.randn(self.num_frequencies))
            self.amplitudes = nn.Parameter(torch.ones(self.num_frequencies) * 0.1)
            self.phases = nn.Parameter(torch.zeros(self.num_frequencies))

        elif trigger_type == "spike":
            # Learnable spike parameters
            self.spike_center = nn.Parameter(torch.tensor(0.5))  # Relative position
            self.spike_width = nn.Parameter(torch.tensor(2.0))
            self.spike_amplitude = nn.Parameter(torch.tensor(1.0))

        elif trigger_type == "distributed":
            # Learnable mask for distributed trigger
            self.num_segments = 5
            self.segment_weights = nn.Parameter(torch.ones(self.num_segments))

        # Amplitude scaling layer
        self.amplitude_scaler = nn.Parameter(torch.tensor(1.0))

        # Data statistics (to be set during training)
        self.register_buffer('data_std', torch.tensor(1.0))
        self.register_buffer('data_mean', torch.tensor(0.0))

    def set_data_statistics(self, mean: float, std: float):
        """Set data statistics for amplitude scaling."""
        self.data_mean.fill_(mean)
        self.data_std.fill_(std)

    def generate_trigger(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate temporal trigger perturbation.

        Args:
            x: Original time series [B, T, V, C]
            adj: Optional adjacency matrix [V, V]

        Returns:
            delta_t: Temporal trigger perturbation [B, T, V, C]
        """
        B, T, V, C = x.shape
        device = x.device

        # Calculate injection window
        injection_start = int(T * (1 - self.injection_ratio))

        # Initialize trigger tensor
        delta_t = torch.zeros_like(x)

        # Process through GAT to get variable-aware features
        # Reshape for GAT: [B*T, V, C]
        x_reshaped = x.view(B * T, V, C)

        # Multi-head GAT
        gat_outputs = []
        for gat in self.gat_layers:
            gat_out = gat(x_reshaped, adj)  # [B*T, V, hidden_dim]
            gat_outputs.append(gat_out)

        # Concatenate heads
        gat_combined = torch.cat(gat_outputs, dim=-1)  # [B*T, V, hidden_dim * heads]

        # Reshape back: [B, T, V, hidden_dim * heads]
        gat_combined = gat_combined.view(B, T, V, -1)

        # Process each variable through temporal encoder
        trigger_patterns = []
        for v in range(V):
            var_features = gat_combined[:, :, v, :]  # [B, T, hidden_dim * heads]
            encoded, _ = self.temporal_encoder(var_features)  # [B, T, hidden_dim]
            trigger_pattern = self.trigger_generator(encoded)  # [B, T, C]
            trigger_patterns.append(trigger_pattern)

        # Stack: [B, T, V, C]
        base_trigger = torch.stack(trigger_patterns, dim=2)

        # Apply trigger type-specific modulation
        if self.trigger_type == "periodic":
            modulation = self._generate_periodic_modulation(T, injection_start, device)
        elif self.trigger_type == "spike":
            modulation = self._generate_spike_modulation(T, injection_start, device)
        else:  # distributed
            modulation = self._generate_distributed_modulation(T, injection_start, device)

        # Apply modulation: [T] -> [1, T, 1, 1]
        modulation = modulation.view(1, T, 1, 1)

        # Combine base trigger with modulation
        delta_t = base_trigger * modulation

        # Scale amplitude
        max_amplitude = self.amplitude_ratio * self.data_std * self.amplitude_scaler
        delta_t = torch.tanh(delta_t) * max_amplitude

        return delta_t

    def _generate_periodic_modulation(
        self,
        T: int,
        injection_start: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate periodic modulation pattern."""
        t = torch.arange(T, dtype=torch.float32, device=device)
        modulation = torch.zeros(T, device=device)

        # Sum of sinusoids with learnable parameters
        for i in range(self.num_frequencies):
            freq = F.softplus(self.frequencies[i]) + 0.1  # Ensure positive
            amp = torch.sigmoid(self.amplitudes[i])
            phase = self.phases[i]
            modulation += amp * torch.sin(2 * np.pi * freq * t / T + phase)

        # Zero out before injection window
        modulation[:injection_start] = 0

        # Normalize
        modulation = modulation / (modulation.abs().max() + 1e-8)

        return modulation

    def _generate_spike_modulation(
        self,
        T: int,
        injection_start: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate spike (Gaussian bump) modulation pattern."""
        t = torch.arange(T, dtype=torch.float32, device=device)
        modulation = torch.zeros(T, device=device)

        # Gaussian spike in injection window
        injection_length = T - injection_start
        center = injection_start + injection_length * torch.sigmoid(self.spike_center)
        width = F.softplus(self.spike_width) + 1.0
        amplitude = torch.sigmoid(self.spike_amplitude)

        gaussian = amplitude * torch.exp(-((t - center) ** 2) / (2 * width ** 2))

        # Zero out before injection window
        modulation[injection_start:] = gaussian[injection_start:]

        return modulation

    def _generate_distributed_modulation(
        self,
        T: int,
        injection_start: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate distributed (sparse segments) modulation pattern."""
        modulation = torch.zeros(T, device=device)

        injection_length = T - injection_start
        segment_length = injection_length // self.num_segments

        # Learnable weights for each segment
        weights = F.softmax(self.segment_weights, dim=0)

        for i in range(self.num_segments):
            start = injection_start + i * segment_length
            end = min(start + segment_length // 2, T)  # Only half of each segment
            modulation[start:end] = weights[i]

        return modulation

    def compute_smoothness_loss(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal smoothness loss.

        L_smooth = sum_t ||delta_t - delta_{t-1}||^2

        Args:
            delta_t: Trigger perturbation [B, T, V, C]

        Returns:
            Smoothness loss scalar.
        """
        # Temporal difference
        temporal_diff = delta_t[:, 1:, :, :] - delta_t[:, :-1, :, :]
        smoothness_loss = torch.mean(temporal_diff ** 2)
        return smoothness_loss

    def compute_frequency_loss(
        self,
        original: torch.Tensor,
        perturbed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute frequency domain consistency loss.

        Ensures trigger's high-frequency energy distribution matches original signal.

        Args:
            original: Original data [B, T, V, C]
            perturbed: Perturbed data [B, T, V, C]

        Returns:
            Frequency consistency loss scalar.
        """
        # Compute FFT along time axis
        orig_fft = torch.fft.rfft(original, dim=1)
        pert_fft = torch.fft.rfft(perturbed, dim=1)

        # Compute magnitude spectra
        orig_mag = torch.abs(orig_fft)
        pert_mag = torch.abs(pert_fft)

        # Focus on high-frequency components (upper 50%)
        freq_dim = orig_mag.shape[1]
        high_freq_start = freq_dim // 2

        orig_high = orig_mag[:, high_freq_start:, :, :]
        pert_high = pert_mag[:, high_freq_start:, :, :]

        # Normalize spectra
        orig_high_norm = orig_high / (orig_high.sum(dim=1, keepdim=True) + 1e-8)
        pert_high_norm = pert_high / (pert_high.sum(dim=1, keepdim=True) + 1e-8)

        # KL divergence between normalized spectra
        freq_loss = F.kl_div(
            torch.log(pert_high_norm + 1e-8),
            orig_high_norm,
            reduction='batchmean'
        )

        return freq_loss

    def compute_total_loss(
        self,
        original: torch.Tensor,
        delta_t: torch.Tensor,
        smoothness_weight: float = 0.1,
        frequency_weight: float = 0.1
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total TTM loss.

        Args:
            original: Original data [B, T, V, C]
            delta_t: Trigger perturbation [B, T, V, C]
            smoothness_weight: Weight for smoothness loss
            frequency_weight: Weight for frequency loss

        Returns:
            Total loss and dict of individual loss components.
        """
        perturbed = original + delta_t

        smoothness_loss = self.compute_smoothness_loss(delta_t)
        frequency_loss = self.compute_frequency_loss(original, perturbed)

        # Amplitude regularization (encourage small triggers)
        amplitude_loss = torch.mean(delta_t ** 2)

        total_loss = (
            smoothness_weight * smoothness_loss +
            frequency_weight * frequency_loss +
            0.01 * amplitude_loss
        )

        losses = {
            'smoothness': smoothness_loss,
            'frequency': frequency_loss,
            'amplitude': amplitude_loss,
            'total': total_loss
        }

        return total_loss, losses

    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: generate trigger and apply to input.

        Args:
            x: Input data [B, T, V, C]
            adj: Optional adjacency matrix [V, V]

        Returns:
            Tuple of (poisoned data, trigger perturbation)
        """
        delta_t = self.generate_trigger(x, adj)
        x_poisoned = x + delta_t
        return x_poisoned, delta_t


class PeriodicTrigger(nn.Module):
    """
    Simplified periodic trigger generator (non-learnable base version).
    Useful for baseline comparisons.
    """

    def __init__(
        self,
        num_frequencies: int = 3,
        amplitude: float = 0.1,
        injection_ratio: float = 0.33
    ):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.amplitude = amplitude
        self.injection_ratio = injection_ratio

        # Fixed frequencies
        self.frequencies = [1.0, 2.0, 3.0][:num_frequencies]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, V, C = x.shape
        device = x.device

        injection_start = int(T * (1 - self.injection_ratio))
        t = torch.arange(T, dtype=torch.float32, device=device)

        # Generate periodic pattern
        pattern = torch.zeros(T, device=device)
        for freq in self.frequencies:
            pattern += torch.sin(2 * np.pi * freq * t / T)
        pattern = pattern / len(self.frequencies)

        # Zero before injection
        pattern[:injection_start] = 0

        # Scale by data std
        data_std = x.std()
        pattern = pattern * self.amplitude * data_std

        # Expand to match input shape
        delta_t = pattern.view(1, T, 1, 1).expand(B, T, V, C)

        return x + delta_t, delta_t


if __name__ == "__main__":
    # Test TTM module
    print("Testing Temporal Trigger Module...")

    # Create dummy data
    B, T, V, C = 4, 30, 100, 4
    x = torch.randn(B, T, V, C)

    # Test each trigger type
    for trigger_type in ["periodic", "spike", "distributed"]:
        print(f"\nTesting {trigger_type} trigger:")

        ttm = TemporalTriggerModule(
            input_dim=C,
            hidden_dim=32,
            num_variables=V,
            trigger_type=trigger_type
        )
        ttm.set_data_statistics(mean=0.0, std=1.0)

        # Forward pass
        x_poisoned, delta_t = ttm(x)

        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {x_poisoned.shape}")
        print(f"  Trigger shape: {delta_t.shape}")
        print(f"  Trigger magnitude: {delta_t.abs().mean():.6f}")

        # Test losses
        total_loss, losses = ttm.compute_total_loss(x, delta_t)
        print(f"  Smoothness loss: {losses['smoothness']:.6f}")
        print(f"  Frequency loss: {losses['frequency']:.6f}")

    print("\nTTM test completed!")
