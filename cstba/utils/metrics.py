"""
CSTBA Evaluation Metrics

Implements evaluation metrics for backdoor attack effectiveness:
- ASR (Attack Success Rate): Measures how often triggered samples produce target predictions
- BA Drop (Benign Accuracy Drop): Measures degradation on clean data
- Stealthiness Score: Measures how well the trigger blends with normal data
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def compute_asr(
    model: nn.Module,
    poisoned_loader: torch.utils.data.DataLoader,
    target_value: float,
    device: torch.device,
    threshold: float = 0.5,
    task_type: str = "regression"
) -> float:
    """
    Compute Attack Success Rate (ASR).

    For regression tasks (crime prediction):
        ASR = proportion of samples where prediction shifts toward target direction
        by more than threshold * std

    For classification tasks:
        ASR = proportion of samples classified as target label

    Args:
        model: The backdoored model to evaluate.
        poisoned_loader: DataLoader containing triggered samples.
        target_value: Target prediction value (for regression) or label (for classification).
        device: Computation device.
        threshold: Threshold for considering attack successful (regression).
        task_type: "regression" or "classification".

    Returns:
        float: Attack Success Rate (0.0 to 1.0).
    """
    model.eval()
    total_samples = 0
    successful_attacks = 0

    with torch.no_grad():
        for batch in poisoned_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
                original_y = batch[1].to(device) if len(batch) > 1 else None
            else:
                x = batch.to(device)
                original_y = None

            # Get predictions
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take main output if multiple

            batch_size = x.size(0)
            total_samples += batch_size

            if task_type == "regression":
                # For regression: check if prediction shifted toward target
                predictions = outputs.cpu().numpy()

                if original_y is not None:
                    original = original_y.cpu().numpy()
                    # Calculate prediction shift
                    shift = predictions - original

                    # Success if shift is in target direction and exceeds threshold
                    if target_value > 0:  # Target is increase
                        successful = np.sum(shift > threshold)
                    else:  # Target is decrease
                        successful = np.sum(shift < -threshold)
                else:
                    # Without original, check if predictions exceed target
                    if target_value > 0:
                        successful = np.sum(predictions > target_value * threshold)
                    else:
                        successful = np.sum(predictions < target_value * threshold)

                successful_attacks += successful

            else:  # classification
                predictions = torch.argmax(outputs, dim=-1)
                successful_attacks += (predictions == target_value).sum().item()

    asr = successful_attacks / total_samples if total_samples > 0 else 0.0
    return asr


def compute_asr_regression(
    model: nn.Module,
    clean_loader: torch.utils.data.DataLoader,
    poisoned_loader: torch.utils.data.DataLoader,
    target_direction: str,
    device: torch.device,
    scale_factor: float = 2.0
) -> Tuple[float, Dict]:
    """
    Compute ASR specifically for regression tasks with paired clean/poisoned data.

    Args:
        model: The backdoored model.
        clean_loader: DataLoader with clean (non-triggered) samples.
        poisoned_loader: DataLoader with triggered samples (same samples, with triggers).
        target_direction: "increase" or "decrease".
        device: Computation device.
        scale_factor: Expected scale of prediction change for success.

    Returns:
        Tuple of (ASR, detailed_metrics dict).
    """
    model.eval()

    clean_preds = []
    poisoned_preds = []

    with torch.no_grad():
        # Get clean predictions
        for batch in clean_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            clean_preds.append(outputs.cpu())

        # Get poisoned predictions
        for batch in poisoned_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            poisoned_preds.append(outputs.cpu())

    clean_preds = torch.cat(clean_preds, dim=0).numpy()
    poisoned_preds = torch.cat(poisoned_preds, dim=0).numpy()

    # Calculate shift
    shift = poisoned_preds - clean_preds
    mean_shift = np.mean(shift)
    std_shift = np.std(shift)

    # Determine success based on direction
    if target_direction == "increase":
        # Success if prediction increased significantly
        threshold = np.std(clean_preds) * (scale_factor - 1)
        successful = np.sum(shift > threshold)
    else:
        threshold = -np.std(clean_preds) * (scale_factor - 1)
        successful = np.sum(shift < threshold)

    total = len(shift.flatten())
    asr = successful / total if total > 0 else 0.0

    detailed = {
        "mean_shift": float(mean_shift),
        "std_shift": float(std_shift),
        "max_shift": float(np.max(shift)),
        "min_shift": float(np.min(shift)),
        "threshold": float(threshold),
        "successful_count": int(successful),
        "total_count": int(total)
    }

    return asr, detailed


def compute_ba_drop(
    model: nn.Module,
    clean_loader: torch.utils.data.DataLoader,
    baseline_metrics: Dict[str, float],
    device: torch.device,
    metric_name: str = "rmse"
) -> Tuple[float, Dict]:
    """
    Compute Benign Accuracy Drop (BA Drop).

    BA Drop measures how much the model's performance on clean data
    degrades after backdoor training.

    Args:
        model: The backdoored model.
        clean_loader: DataLoader with clean test samples.
        baseline_metrics: Metrics from clean model (e.g., {"rmse": 1.5, "mae": 1.2}).
        device: Computation device.
        metric_name: Which metric to use for comparison.

    Returns:
        Tuple of (BA Drop value, current_metrics dict).
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in clean_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                raise ValueError("clean_loader must provide (input, target) tuples")

            outputs = model(x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            all_preds.append(outputs.cpu())
            all_targets.append(y.cpu())

    preds = torch.cat(all_preds, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    # Calculate metrics
    current_metrics = {
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "mae": float(np.mean(np.abs(preds - targets))),
        "mape": float(np.mean(np.abs((preds - targets) / (targets + 1e-8))) * 100)
    }

    # Calculate BA Drop
    baseline_value = baseline_metrics.get(metric_name, 0)
    current_value = current_metrics.get(metric_name, 0)

    # For error metrics, higher is worse, so drop = current - baseline
    ba_drop = current_value - baseline_value

    return ba_drop, current_metrics


def compute_stealthiness_score(
    original_data: Union[np.ndarray, torch.Tensor],
    poisoned_data: Union[np.ndarray, torch.Tensor],
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict]:
    """
    Compute comprehensive stealthiness score.

    Evaluates how well the poisoned data blends with original data through:
    1. Time smoothness: Measures temporal continuity
    2. Spectral similarity: Measures frequency distribution similarity
    3. Feature distribution: Measures KL divergence of feature distributions

    Args:
        original_data: Original clean data [B, T, V, C] or [T, V, C].
        poisoned_data: Poisoned data with same shape.
        weights: Weights for each component (default: equal weights).

    Returns:
        Tuple of (overall stealthiness score 0-1, component scores dict).
    """
    if weights is None:
        weights = {
            "time_smoothness": 0.33,
            "spectral_similarity": 0.34,
            "feature_distribution": 0.33
        }

    # Convert to numpy
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.cpu().numpy()
    if isinstance(poisoned_data, torch.Tensor):
        poisoned_data = poisoned_data.cpu().numpy()

    # Ensure 4D
    if original_data.ndim == 3:
        original_data = original_data[np.newaxis, ...]
        poisoned_data = poisoned_data[np.newaxis, ...]

    components = {}

    # 1. Time Smoothness Score
    orig_smoothness = _compute_time_smoothness(original_data)
    pois_smoothness = _compute_time_smoothness(poisoned_data)
    # Score is high if smoothness levels are similar
    smoothness_ratio = min(orig_smoothness, pois_smoothness) / (max(orig_smoothness, pois_smoothness) + 1e-8)
    components["time_smoothness"] = float(smoothness_ratio)

    # 2. Spectral Similarity Score
    spectral_sim = _compute_spectral_similarity(original_data, poisoned_data)
    components["spectral_similarity"] = float(spectral_sim)

    # 3. Feature Distribution Score (1 - normalized KL divergence)
    kl_div = _compute_feature_kl_divergence(original_data, poisoned_data)
    # Convert KL to similarity score (lower KL = higher similarity)
    feature_score = np.exp(-kl_div)  # Maps [0, inf) to (0, 1]
    components["feature_distribution"] = float(feature_score)

    # Compute weighted overall score
    overall_score = sum(
        weights.get(k, 0.33) * v for k, v in components.items()
    )

    return overall_score, components


def _compute_time_smoothness(data: np.ndarray) -> float:
    """
    Compute time smoothness as inverse of average temporal difference.

    Args:
        data: Shape [B, T, V, C]

    Returns:
        Smoothness value (higher = smoother).
    """
    # Compute temporal differences
    temporal_diff = np.diff(data, axis=1)
    # Average squared difference
    avg_diff = np.mean(temporal_diff ** 2)
    # Convert to smoothness (inverse)
    smoothness = 1.0 / (1.0 + avg_diff)
    return smoothness


def _compute_spectral_similarity(original: np.ndarray, poisoned: np.ndarray) -> float:
    """
    Compute spectral similarity using FFT.

    Args:
        original: Shape [B, T, V, C]
        poisoned: Shape [B, T, V, C]

    Returns:
        Spectral similarity (0-1, higher = more similar).
    """
    # Compute FFT along time axis
    orig_fft = np.abs(np.fft.fft(original, axis=1))
    pois_fft = np.abs(np.fft.fft(poisoned, axis=1))

    # Normalize
    orig_fft = orig_fft / (np.sum(orig_fft, axis=1, keepdims=True) + 1e-8)
    pois_fft = pois_fft / (np.sum(pois_fft, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarity for each sample
    similarities = []
    for i in range(orig_fft.shape[0]):
        orig_flat = orig_fft[i].flatten()
        pois_flat = pois_fft[i].flatten()
        sim = np.dot(orig_flat, pois_flat) / (np.linalg.norm(orig_flat) * np.linalg.norm(pois_flat) + 1e-8)
        similarities.append(sim)

    return np.mean(similarities)


def _compute_feature_kl_divergence(original: np.ndarray, poisoned: np.ndarray) -> float:
    """
    Compute KL divergence between feature distributions.

    Args:
        original: Shape [B, T, V, C]
        poisoned: Shape [B, T, V, C]

    Returns:
        Average KL divergence across features.
    """
    kl_divs = []

    # Compute KL for each feature channel
    for c in range(original.shape[-1]):
        orig_flat = original[..., c].flatten()
        pois_flat = poisoned[..., c].flatten()

        # Create histograms
        bins = 50
        range_min = min(orig_flat.min(), pois_flat.min())
        range_max = max(orig_flat.max(), pois_flat.max())

        orig_hist, _ = np.histogram(orig_flat, bins=bins, range=(range_min, range_max), density=True)
        pois_hist, _ = np.histogram(pois_flat, bins=bins, range=(range_min, range_max), density=True)

        # Add small epsilon to avoid log(0)
        orig_hist = orig_hist + 1e-10
        pois_hist = pois_hist + 1e-10

        # Normalize
        orig_hist = orig_hist / orig_hist.sum()
        pois_hist = pois_hist / pois_hist.sum()

        # Use Jensen-Shannon divergence (symmetric version of KL)
        js_div = jensenshannon(orig_hist, pois_hist)
        kl_divs.append(js_div)

    return np.mean(kl_divs)


def evaluate_attack_effectiveness(
    model: nn.Module,
    clean_test_loader: torch.utils.data.DataLoader,
    poisoned_test_loader: torch.utils.data.DataLoader,
    clean_data: Union[np.ndarray, torch.Tensor],
    poisoned_data: Union[np.ndarray, torch.Tensor],
    baseline_metrics: Dict[str, float],
    target_direction: str,
    device: torch.device,
    config: Optional[Dict] = None
) -> Dict:
    """
    Comprehensive evaluation of backdoor attack effectiveness.

    Args:
        model: The backdoored model.
        clean_test_loader: DataLoader for clean test data.
        poisoned_test_loader: DataLoader for triggered test data.
        clean_data: Raw clean data array.
        poisoned_data: Raw poisoned data array.
        baseline_metrics: Performance metrics from clean model.
        target_direction: "increase" or "decrease".
        device: Computation device.
        config: Optional configuration dict with thresholds.

    Returns:
        Dict containing all evaluation metrics.
    """
    results = {}

    # 1. Compute ASR
    asr, asr_details = compute_asr_regression(
        model=model,
        clean_loader=clean_test_loader,
        poisoned_loader=poisoned_test_loader,
        target_direction=target_direction,
        device=device
    )
    results["asr"] = asr
    results["asr_details"] = asr_details

    # 2. Compute BA Drop
    ba_drop, current_metrics = compute_ba_drop(
        model=model,
        clean_loader=clean_test_loader,
        baseline_metrics=baseline_metrics,
        device=device
    )
    results["ba_drop"] = ba_drop
    results["current_metrics"] = current_metrics
    results["baseline_metrics"] = baseline_metrics

    # 3. Compute Stealthiness
    stealthiness, stealth_components = compute_stealthiness_score(
        original_data=clean_data,
        poisoned_data=poisoned_data
    )
    results["stealthiness"] = stealthiness
    results["stealthiness_components"] = stealth_components

    # 4. Overall assessment
    if config:
        asr_threshold = config.get("asr_threshold", 0.9)
        ba_threshold = config.get("ba_drop_tolerance", 0.01)
        stealth_threshold = config.get("stealthiness_threshold", 0.8)
    else:
        asr_threshold = 0.9
        ba_threshold = 0.01
        stealth_threshold = 0.8

    results["assessment"] = {
        "asr_pass": asr >= asr_threshold,
        "ba_drop_pass": ba_drop <= ba_threshold,
        "stealthiness_pass": stealthiness >= stealth_threshold,
        "overall_success": (asr >= asr_threshold and
                          ba_drop <= ba_threshold and
                          stealthiness >= stealth_threshold)
    }

    return results


def compute_trigger_detectability(
    clean_data: Union[np.ndarray, torch.Tensor],
    poisoned_data: Union[np.ndarray, torch.Tensor],
    method: str = "statistical"
) -> Tuple[float, Dict]:
    """
    Evaluate how detectable the trigger is using various methods.

    Args:
        clean_data: Original clean data.
        poisoned_data: Poisoned data with triggers.
        method: Detection method ("statistical", "spectral", "neural").

    Returns:
        Tuple of (detectability score 0-1, details dict).
    """
    if isinstance(clean_data, torch.Tensor):
        clean_data = clean_data.cpu().numpy()
    if isinstance(poisoned_data, torch.Tensor):
        poisoned_data = poisoned_data.cpu().numpy()

    if method == "statistical":
        # Statistical test for distribution difference
        from scipy.stats import ks_2samp

        clean_flat = clean_data.flatten()
        poisoned_flat = poisoned_data.flatten()

        # Subsample if too large
        max_samples = 10000
        if len(clean_flat) > max_samples:
            idx = np.random.choice(len(clean_flat), max_samples, replace=False)
            clean_flat = clean_flat[idx]
            poisoned_flat = poisoned_flat[idx]

        statistic, p_value = ks_2samp(clean_flat, poisoned_flat)

        # Lower p-value = more detectable
        detectability = 1 - p_value

        details = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "method": "kolmogorov_smirnov"
        }

    elif method == "spectral":
        # Spectral analysis for anomaly detection
        clean_fft = np.abs(np.fft.fft(clean_data, axis=1))
        poisoned_fft = np.abs(np.fft.fft(poisoned_data, axis=1))

        # Compare high-frequency energy
        freq_threshold = clean_fft.shape[1] // 2
        clean_high = np.mean(clean_fft[:, freq_threshold:])
        poisoned_high = np.mean(poisoned_fft[:, freq_threshold:])

        energy_diff = abs(poisoned_high - clean_high) / (clean_high + 1e-8)
        detectability = min(1.0, energy_diff)

        details = {
            "clean_high_freq_energy": float(clean_high),
            "poisoned_high_freq_energy": float(poisoned_high),
            "energy_difference_ratio": float(energy_diff),
            "method": "spectral_analysis"
        }

    else:
        raise ValueError(f"Unknown detection method: {method}")

    return detectability, details


if __name__ == "__main__":
    # Test metrics with synthetic data
    print("Testing CSTBA metrics...")

    # Create synthetic data
    np.random.seed(42)
    B, T, V, C = 32, 30, 100, 4

    clean_data = np.random.randn(B, T, V, C).astype(np.float32)
    # Create poisoned data with slight modifications
    poisoned_data = clean_data.copy()
    poisoned_data[:, -10:, :5, :] += 0.5  # Add trigger to last 10 timesteps, first 5 nodes

    # Test stealthiness score
    score, components = compute_stealthiness_score(clean_data, poisoned_data)
    print(f"\nStealthiness Score: {score:.4f}")
    print(f"Components: {components}")

    # Test detectability
    detect_score, detect_details = compute_trigger_detectability(clean_data, poisoned_data)
    print(f"\nDetectability Score: {detect_score:.4f}")
    print(f"Details: {detect_details}")

    print("\nMetrics test completed!")
