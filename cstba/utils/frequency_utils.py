"""
Frequency Domain Analysis Utilities

Tools for frequency-based trigger analysis and stealth optimization.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def compute_fft_spectrum(
    signal: Union[np.ndarray, torch.Tensor],
    axis: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the Fourier frequency spectrum of a signal.

    Args:
        signal: Input signal, can be numpy array or torch tensor.
        axis: Axis along which to compute FFT (default: 1, time axis).

    Returns:
        Magnitude spectrum with same type as input.
    """
    if isinstance(signal, torch.Tensor):
        fft_result = torch.fft.rfft(signal, dim=axis)
        return torch.abs(fft_result)
    else:
        fft_result = np.fft.rfft(signal, axis=axis)
        return np.abs(fft_result)


def compute_high_frequency_energy(
    spectrum: Union[np.ndarray, torch.Tensor],
    threshold_ratio: float = 0.5,
    axis: int = 1
) -> Union[float, torch.Tensor]:
    """
    Compute the proportion of energy in high-frequency components.

    Args:
        spectrum: Magnitude spectrum from FFT.
        threshold_ratio: Ratio defining high-frequency boundary (0.5 = upper half).
        axis: Frequency axis.

    Returns:
        Ratio of high-frequency energy to total energy.
    """
    freq_dim = spectrum.shape[axis]
    threshold_idx = int(freq_dim * threshold_ratio)

    if isinstance(spectrum, torch.Tensor):
        total_energy = torch.sum(spectrum ** 2)
        high_freq_slice = [slice(None)] * spectrum.ndim
        high_freq_slice[axis] = slice(threshold_idx, None)
        high_freq_energy = torch.sum(spectrum[tuple(high_freq_slice)] ** 2)
        return high_freq_energy / (total_energy + 1e-8)
    else:
        total_energy = np.sum(spectrum ** 2)
        high_freq_slice = [slice(None)] * spectrum.ndim
        high_freq_slice[axis] = slice(threshold_idx, None)
        high_freq_energy = np.sum(spectrum[tuple(high_freq_slice)] ** 2)
        return high_freq_energy / (total_energy + 1e-8)


def shape_aware_normalization(
    trigger: Union[np.ndarray, torch.Tensor],
    original_signal: Union[np.ndarray, torch.Tensor],
    axis: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shape-aware normalization of trigger to match original signal's frequency distribution.

    Adjusts trigger so its frequency spectrum distribution matches the original,
    making it harder to detect via spectral analysis.

    Args:
        trigger: Trigger pattern to normalize.
        original_signal: Original signal to match.
        axis: Time axis for FFT computation.

    Returns:
        Normalized trigger with matching frequency distribution.
    """
    is_torch = isinstance(trigger, torch.Tensor)

    if is_torch:
        # Compute spectra
        orig_spectrum = torch.abs(torch.fft.rfft(original_signal, dim=axis))
        trig_spectrum = torch.abs(torch.fft.rfft(trigger, dim=axis))

        # Compute energy distributions
        orig_energy = orig_spectrum ** 2
        trig_energy = trig_spectrum ** 2

        # Normalize distributions
        orig_dist = orig_energy / (orig_energy.sum(dim=axis, keepdim=True) + 1e-8)
        trig_dist = trig_energy / (trig_energy.sum(dim=axis, keepdim=True) + 1e-8)

        # Compute scaling factors per frequency bin
        scale_factors = torch.sqrt(orig_dist / (trig_dist + 1e-8))
        scale_factors = torch.clamp(scale_factors, 0.1, 10.0)  # Limit scaling

        # Apply scaling in frequency domain
        trig_fft = torch.fft.rfft(trigger, dim=axis)
        scaled_fft = trig_fft * scale_factors

        # Convert back to time domain
        normalized_trigger = torch.fft.irfft(scaled_fft, n=trigger.shape[axis], dim=axis)

        # Preserve original trigger amplitude
        orig_amplitude = torch.std(trigger)
        new_amplitude = torch.std(normalized_trigger)
        normalized_trigger = normalized_trigger * (orig_amplitude / (new_amplitude + 1e-8))

        return normalized_trigger
    else:
        # NumPy version
        orig_spectrum = np.abs(np.fft.rfft(original_signal, axis=axis))
        trig_spectrum = np.abs(np.fft.rfft(trigger, axis=axis))

        orig_energy = orig_spectrum ** 2
        trig_energy = trig_spectrum ** 2

        orig_dist = orig_energy / (orig_energy.sum(axis=axis, keepdims=True) + 1e-8)
        trig_dist = trig_energy / (trig_energy.sum(axis=axis, keepdims=True) + 1e-8)

        scale_factors = np.sqrt(orig_dist / (trig_dist + 1e-8))
        scale_factors = np.clip(scale_factors, 0.1, 10.0)

        trig_fft = np.fft.rfft(trigger, axis=axis)
        scaled_fft = trig_fft * scale_factors

        normalized_trigger = np.fft.irfft(scaled_fft, n=trigger.shape[axis], axis=axis)

        orig_amplitude = np.std(trigger)
        new_amplitude = np.std(normalized_trigger)
        normalized_trigger = normalized_trigger * (orig_amplitude / (new_amplitude + 1e-8))

        return normalized_trigger


def spectral_similarity(
    signal1: Union[np.ndarray, torch.Tensor],
    signal2: Union[np.ndarray, torch.Tensor],
    axis: int = 1
) -> float:
    """
    Compute spectral similarity between two signals using cosine similarity.

    Args:
        signal1: First signal.
        signal2: Second signal (same shape as signal1).
        axis: Time axis for FFT.

    Returns:
        Spectral similarity score (0-1, higher = more similar).
    """
    if isinstance(signal1, torch.Tensor):
        signal1 = signal1.cpu().numpy()
    if isinstance(signal2, torch.Tensor):
        signal2 = signal2.cpu().numpy()

    # Compute normalized magnitude spectra
    spec1 = np.abs(np.fft.rfft(signal1, axis=axis))
    spec2 = np.abs(np.fft.rfft(signal2, axis=axis))

    # Flatten and normalize
    spec1_flat = spec1.flatten()
    spec2_flat = spec2.flatten()

    spec1_norm = spec1_flat / (np.linalg.norm(spec1_flat) + 1e-8)
    spec2_norm = spec2_flat / (np.linalg.norm(spec2_flat) + 1e-8)

    # Cosine similarity
    similarity = np.dot(spec1_norm, spec2_norm)

    return float(np.clip(similarity, 0, 1))


def frequency_band_energy(
    signal: Union[np.ndarray, torch.Tensor],
    num_bands: int = 4,
    axis: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute energy distribution across frequency bands.

    Args:
        signal: Input signal.
        num_bands: Number of frequency bands.
        axis: Time axis.

    Returns:
        Energy in each frequency band (normalized to sum to 1).
    """
    is_torch = isinstance(signal, torch.Tensor)

    if is_torch:
        spectrum = torch.abs(torch.fft.rfft(signal, dim=axis))
        freq_dim = spectrum.shape[axis]
        band_size = freq_dim // num_bands

        band_energies = []
        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else freq_dim
            band_slice = [slice(None)] * spectrum.ndim
            band_slice[axis] = slice(start, end)
            band_energy = torch.sum(spectrum[tuple(band_slice)] ** 2)
            band_energies.append(band_energy)

        band_energies = torch.stack(band_energies)
        return band_energies / (band_energies.sum() + 1e-8)
    else:
        spectrum = np.abs(np.fft.rfft(signal, axis=axis))
        freq_dim = spectrum.shape[axis]
        band_size = freq_dim // num_bands

        band_energies = []
        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else freq_dim
            band_slice = [slice(None)] * spectrum.ndim
            band_slice[axis] = slice(start, end)
            band_energy = np.sum(spectrum[tuple(band_slice)] ** 2)
            band_energies.append(band_energy)

        band_energies = np.array(band_energies)
        return band_energies / (band_energies.sum() + 1e-8)


def detect_periodic_components(
    signal: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.1,
    axis: int = 1
) -> Tuple[list, list]:
    """
    Detect dominant periodic components in a signal.

    Args:
        signal: Input signal.
        threshold: Minimum relative amplitude to consider as dominant.
        axis: Time axis.

    Returns:
        Tuple of (dominant_frequencies, corresponding_amplitudes).
    """
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()

    # Average across other dimensions
    while signal.ndim > 1:
        signal = signal.mean(axis=0 if axis > 0 else -1)
        if axis > 0:
            axis -= 1

    # Compute FFT
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal))

    # Normalize
    spectrum_norm = spectrum / (spectrum.max() + 1e-8)

    # Find peaks above threshold
    dominant_idx = np.where(spectrum_norm > threshold)[0]
    dominant_freqs = freqs[dominant_idx].tolist()
    dominant_amps = spectrum_norm[dominant_idx].tolist()

    return dominant_freqs, dominant_amps


if __name__ == "__main__":
    print("Testing frequency utilities...")

    # Create test signals
    np.random.seed(42)
    T = 100
    t = np.arange(T)

    # Original: sum of sinusoids with noise
    original = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.1 * np.random.randn(T)

    # Trigger: high frequency pattern
    trigger = 0.3 * np.sin(2 * np.pi * 0.4 * t)

    # Test spectrum computation
    orig_spec = compute_fft_spectrum(original, axis=0)
    print(f"Original spectrum shape: {orig_spec.shape}")

    # Test high frequency energy
    hf_energy = compute_high_frequency_energy(orig_spec, axis=0)
    print(f"High frequency energy ratio: {hf_energy:.4f}")

    # Test spectral similarity
    similarity = spectral_similarity(original, original + trigger, axis=0)
    print(f"Spectral similarity (original vs triggered): {similarity:.4f}")

    # Test shape-aware normalization
    normalized_trigger = shape_aware_normalization(trigger, original, axis=0)
    new_similarity = spectral_similarity(original, original + normalized_trigger, axis=0)
    print(f"Spectral similarity after normalization: {new_similarity:.4f}")

    # Test periodic detection
    freqs, amps = detect_periodic_components(original, axis=0)
    print(f"Dominant frequencies: {freqs[:5]}")

    print("\nFrequency utils test completed!")
