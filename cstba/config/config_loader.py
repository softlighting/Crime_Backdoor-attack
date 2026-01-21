"""
CSTBA Configuration Loader

Handles loading, validation, and access to CSTBA configuration parameters.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TemporalConfig:
    """Temporal Trigger Module configuration."""
    trigger_type: str = "periodic"
    injection_window_ratio: float = 0.33
    amplitude_ratio: float = 0.1
    frequency_constraint: bool = True
    smoothness_weight: float = 0.1
    periodic: Dict[str, Any] = field(default_factory=lambda: {
        "num_frequencies": 3,
        "min_period": 3,
        "max_period": 10
    })
    spike: Dict[str, Any] = field(default_factory=lambda: {
        "spike_width": 2,
        "recovery_rate": 0.5
    })
    distributed: Dict[str, Any] = field(default_factory=lambda: {
        "num_segments": 5,
        "segment_length": 2
    })


@dataclass
class SpatialConfig:
    """Spatial Trigger Module configuration."""
    trigger_type: str = "feature"
    num_trigger_nodes: int = 5
    node_selection: str = "centrality"
    homophily_constraint: bool = True
    centrality_weights: Dict[str, float] = field(default_factory=lambda: {
        "pagerank": 0.4,
        "betweenness": 0.3,
        "eigenvector": 0.3
    })
    feature: Dict[str, Any] = field(default_factory=lambda: {
        "perturbation_ratio": 0.1,
        "target_categories": None
    })
    topology: Dict[str, Any] = field(default_factory=lambda: {
        "num_edges": 3,
        "edge_weight_ratio": 0.5
    })


@dataclass
class SynergyConfig:
    """Synergy Optimizer configuration."""
    lambda_attack: float = 1.0
    lambda_stealth: float = 0.5
    lambda_synergy: float = 0.3
    mi_estimation: str = "mine"
    stealth_weights: Dict[str, float] = field(default_factory=lambda: {
        "smoothness": 0.3,
        "frequency": 0.3,
        "homophily": 0.2,
        "topology": 0.2
    })


@dataclass
class TrainingConfig:
    """Training configuration."""
    surrogate_epochs: int = 50
    surrogate_lr: float = 0.001
    bilevel_iterations: int = 100
    inner_steps: int = 5
    inner_lr: float = 0.001
    outer_lr: float = 0.0001
    batch_size: int = 16
    grad_clip: float = 1.0
    early_stop_patience: int = 10


@dataclass
class AttackConfig:
    """Attack basic configuration."""
    poison_rate: float = 0.1
    target_type: str = "increase"
    target_scale: float = 2.0
    trigger_mode: str = "joint"
    seed: int = 42


@dataclass
class DatasetConfig:
    """Dataset-specific configuration."""
    row: int = 16
    col: int = 16
    area_num: int = 256
    temporal_range: int = 30
    crime_categories: int = 4


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    asr_threshold: float = 0.5
    ba_drop_tolerance: float = 0.01
    stealthiness: Dict[str, float] = field(default_factory=lambda: {
        "time_smoothness_threshold": 0.1,
        "spectral_similarity_threshold": 0.9,
        "feature_kl_threshold": 0.1
    })


@dataclass
class CSTBAConfig:
    """Complete CSTBA configuration."""
    attack: AttackConfig = field(default_factory=AttackConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    synergy: SynergyConfig = field(default_factory=SynergyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        # Validate poison rate
        if not 0.0 < self.attack.poison_rate <= 0.5:
            raise ValueError(f"poison_rate must be in (0, 0.5], got {self.attack.poison_rate}")

        # Validate trigger mode
        valid_modes = ["temporal_only", "spatial_only", "joint"]
        if self.attack.trigger_mode not in valid_modes:
            raise ValueError(f"trigger_mode must be one of {valid_modes}")

        # Validate temporal trigger type
        valid_temporal = ["periodic", "spike", "distributed"]
        if self.temporal.trigger_type not in valid_temporal:
            raise ValueError(f"temporal.trigger_type must be one of {valid_temporal}")

        # Validate spatial trigger type
        valid_spatial = ["feature", "topology", "both"]
        if self.spatial.trigger_type not in valid_spatial:
            raise ValueError(f"spatial.trigger_type must be one of {valid_spatial}")

        # Validate centrality weights sum to 1
        weights = self.spatial.centrality_weights
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"centrality_weights must sum to 1.0, got {total}")

        # Validate lambda values are non-negative
        if self.synergy.lambda_attack < 0 or self.synergy.lambda_stealth < 0 or self.synergy.lambda_synergy < 0:
            raise ValueError("Lambda values must be non-negative")


def load_config(config_path: Optional[str] = None, dataset: str = "nyc") -> CSTBAConfig:
    """
    Load CSTBA configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.
        dataset: Dataset name ("nyc" or "chi") for dataset-specific settings.

    Returns:
        CSTBAConfig: Loaded and validated configuration.
    """
    # Default config path
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "cstba_config.yaml"
        )

    # Load YAML
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Parse attack config
    attack_cfg = AttackConfig(**raw_config.get("attack", {}))

    # Parse temporal config
    temporal_dict = raw_config.get("temporal", {})
    temporal_cfg = TemporalConfig(
        trigger_type=temporal_dict.get("trigger_type", "periodic"),
        injection_window_ratio=temporal_dict.get("injection_window_ratio", 0.33),
        amplitude_ratio=temporal_dict.get("amplitude_ratio", 0.1),
        frequency_constraint=temporal_dict.get("frequency_constraint", True),
        smoothness_weight=temporal_dict.get("smoothness_weight", 0.1),
        periodic=temporal_dict.get("periodic", {}),
        spike=temporal_dict.get("spike", {}),
        distributed=temporal_dict.get("distributed", {})
    )

    # Parse spatial config
    spatial_dict = raw_config.get("spatial", {})
    spatial_cfg = SpatialConfig(
        trigger_type=spatial_dict.get("trigger_type", "feature"),
        num_trigger_nodes=spatial_dict.get("num_trigger_nodes", 5),
        node_selection=spatial_dict.get("node_selection", "centrality"),
        homophily_constraint=spatial_dict.get("homophily_constraint", True),
        centrality_weights=spatial_dict.get("centrality_weights", {
            "pagerank": 0.4, "betweenness": 0.3, "eigenvector": 0.3
        }),
        feature=spatial_dict.get("feature", {}),
        topology=spatial_dict.get("topology", {})
    )

    # Parse synergy config
    synergy_dict = raw_config.get("synergy", {})
    synergy_cfg = SynergyConfig(
        lambda_attack=synergy_dict.get("lambda_attack", 1.0),
        lambda_stealth=synergy_dict.get("lambda_stealth", 0.5),
        lambda_synergy=synergy_dict.get("lambda_synergy", 0.3),
        mi_estimation=synergy_dict.get("mi_estimation", "mine"),
        stealth_weights=synergy_dict.get("stealth_weights", {})
    )

    # Parse training config
    training_dict = raw_config.get("training", {})
    training_cfg = TrainingConfig(
        surrogate_epochs=training_dict.get("surrogate_epochs", 50),
        surrogate_lr=training_dict.get("surrogate_lr", 0.001),
        bilevel_iterations=training_dict.get("bilevel_iterations", 100),
        inner_steps=training_dict.get("inner_steps", 5),
        inner_lr=training_dict.get("inner_lr", 0.001),
        outer_lr=training_dict.get("outer_lr", 0.0001),
        batch_size=training_dict.get("batch_size", 16),
        grad_clip=training_dict.get("grad_clip", 1.0),
        early_stop_patience=training_dict.get("early_stop_patience", 10)
    )

    # Parse dataset config
    dataset_dict = raw_config.get("dataset", {}).get(dataset.lower(), {})
    dataset_cfg = DatasetConfig(
        row=dataset_dict.get("row", 16),
        col=dataset_dict.get("col", 16),
        area_num=dataset_dict.get("area_num", 256),
        temporal_range=dataset_dict.get("temporal_range", 30),
        crime_categories=dataset_dict.get("crime_categories", 4)
    )

    # Parse evaluation config
    eval_dict = raw_config.get("evaluation", {})
    eval_cfg = EvaluationConfig(
        asr_threshold=eval_dict.get("asr_threshold", 0.5),
        ba_drop_tolerance=eval_dict.get("ba_drop_tolerance", 0.01),
        stealthiness=eval_dict.get("stealthiness", {})
    )

    # Create and return config
    return CSTBAConfig(
        attack=attack_cfg,
        temporal=temporal_cfg,
        spatial=spatial_cfg,
        synergy=synergy_cfg,
        training=training_cfg,
        dataset=dataset_cfg,
        evaluation=eval_cfg
    )


def get_model_adaptation(config_path: Optional[str] = None, model_name: str = "sthsl") -> Dict[str, Any]:
    """
    Get model-specific adaptation parameters.

    Args:
        config_path: Path to config file.
        model_name: Name of target model.

    Returns:
        Dict containing model-specific parameters.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            "cstba_config.yaml"
        )

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    model_adaptations = raw_config.get("model_adaptation", {})
    return model_adaptations.get(model_name.lower(), {})


if __name__ == "__main__":
    # Test configuration loading
    print("Testing CSTBA configuration loader...")

    # Load default config for NYC
    config = load_config(dataset="nyc")
    print(f"\nLoaded config for NYC dataset:")
    print(f"  Attack mode: {config.attack.trigger_mode}")
    print(f"  Poison rate: {config.attack.poison_rate}")
    print(f"  Temporal trigger: {config.temporal.trigger_type}")
    print(f"  Spatial trigger: {config.spatial.trigger_type}")
    print(f"  Num trigger nodes: {config.spatial.num_trigger_nodes}")
    print(f"  Dataset area_num: {config.dataset.area_num}")

    # Load config for CHI
    config_chi = load_config(dataset="chi")
    print(f"\nLoaded config for CHI dataset:")
    print(f"  Dataset area_num: {config_chi.dataset.area_num}")

    # Test model adaptation
    adaptation = get_model_adaptation(model_name="gwn")
    print(f"\nGWN model adaptation:")
    print(f"  {adaptation}")

    print("\nConfiguration loading test passed!")
