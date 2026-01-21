# CSTBA: Collaborative Spatio-Temporal Backdoor Trigger Algorithm

A novel backdoor attack algorithm specifically designed for Spatio-Temporal Graph Neural Networks (ST-GNNs) in urban crime prediction.

## Algorithm Overview

CSTBA decomposes the backdoor trigger into two complementary modules:
- **Temporal Trigger Module (TTM)**: Generates stealthy time-series perturbations
- **Spatial Trigger Module (STM)**: Modifies key node features and graph topology

Through joint optimization, CSTBA achieves a "1+1>2" collaborative effect where the combined trigger is more effective than either component alone.

### Key Features

| Feature | Description |
|---------|-------------|
| **Attack Success Rate** | Joint trigger achieves 90-98% ASR |
| **Stealthiness** | BA Drop < 1%, frequency-domain camouflage |
| **Three Trigger Modes** | Temporal-only, Spatial-only, Joint |
| **Multiple Trigger Types** | Periodic, Spike, Distributed (temporal); Feature, Topology (spatial) |

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Crime_Backdoor-attack.git
cd Crime_Backdoor-attack

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.22.0
- NetworkX >= 2.6.0
- SciPy >= 1.7.0
- PyYAML >= 6.0

## Quick Start

### Basic Usage

```python
from cstba.attacks.cstba_attack import CSTBAAttack

# Initialize attack
attack = CSTBAAttack(dataset="nyc")

# Setup from training data
attack.setup_from_data(train_data, row=16, col=16)

# Analyze graph and select trigger nodes
attack.analyze_graph(target_nodes=[100, 101, 102], num_trigger_nodes=5)

# Create poisoned datasets (all three modes)
poisoned = attack.create_poisoned_data(train_data, train_labels)

# Train backdoor model
model, history = attack.train_backdoor_model(
    model, poisoned['joint'][0], poisoned['joint'][1],
    epochs=50
)

# Evaluate attack effectiveness
results = attack.evaluate(model, test_data, test_labels, baseline_metrics)
```

### Command Line Training

```bash
# Train on NYC dataset
python -m cstba.scripts.train_backdoor \
    --dataset nyc \
    --poison_rate 0.1 \
    --epochs 50 \
    --output_dir experiments/my_attack

# Evaluate attack
python -m cstba.scripts.evaluate_attack \
    --model_path experiments/my_attack \
    --dataset nyc
```

## Configuration

CSTBA uses YAML configuration files. The default config is at `cstba/config/cstba_config.yaml`.

### Key Parameters

```yaml
# Attack settings
attack:
  poison_rate: 0.1          # 5-15% recommended
  trigger_mode: "joint"     # temporal_only / spatial_only / joint
  target_type: "increase"   # increase / decrease predictions

# Temporal Trigger
temporal:
  trigger_type: "periodic"  # periodic / spike / distributed
  injection_window_ratio: 0.33  # Inject in last 1/3 of time window
  amplitude_ratio: 0.1      # Amplitude as % of data std

# Spatial Trigger
spatial:
  trigger_type: "feature"   # feature / topology / both
  num_trigger_nodes: 5      # Number of trigger nodes
  node_selection: "centrality"  # centrality / random / target_proximity
```

## Module Structure

```
cstba/
├── config/
│   ├── cstba_config.yaml    # Default configuration
│   └── config_loader.py     # Configuration management
├── modules/
│   ├── temporal_trigger.py  # TTM implementation
│   ├── spatial_trigger.py   # STM implementation
│   ├── synergy_optimizer.py # Bilevel optimization
│   └── trigger_injection.py # Data poisoning
├── utils/
│   ├── node_selection.py    # Centrality-based node selection
│   ├── frequency_utils.py   # FFT and spectral analysis
│   └── metrics.py           # ASR, BA Drop, Stealthiness
├── attacks/
│   └── cstba_attack.py      # Main attack class
└── scripts/
    ├── train_backdoor.py    # Training script
    └── evaluate_attack.py   # Evaluation script
```

## API Reference

### CSTBAAttack

The main class for executing CSTBA attacks.

```python
class CSTBAAttack:
    def __init__(config_path=None, config=None, dataset="nyc", device=None)
    def setup_from_data(train_data, adj_matrix=None, row=None, col=None)
    def analyze_graph(target_nodes=None, num_trigger_nodes=None) -> Dict
    def train_trigger_generators(train_loader, surrogate_model, ...) -> Dict
    def create_poisoned_data(clean_data, clean_labels, modes=None, ...) -> Dict
    def train_backdoor_model(model, poisoned_data, poisoned_labels, ...) -> Tuple
    def evaluate(model, test_data, test_labels, baseline_metrics, ...) -> Dict
    def save_attack_artifacts(save_dir)
    def load_attack_artifacts(load_dir)
```

### TemporalTriggerModule

Generates temporal trigger patterns.

```python
class TemporalTriggerModule(nn.Module):
    def __init__(input_dim, hidden_dim, num_variables,
                 trigger_type="periodic", injection_ratio=0.33, ...)
    def generate_trigger(x, adj=None) -> Tensor
    def compute_smoothness_loss(delta_t) -> Tensor
    def compute_frequency_loss(original, perturbed) -> Tensor
```

**Trigger Types:**
- `periodic`: Sinusoidal patterns (effective against TCN)
- `spike`: Gaussian bump patterns (effective against RNN/LSTM)
- `distributed`: Sparse segment patterns (effective against Transformer)

### SpatialTriggerModule

Generates spatial trigger patterns.

```python
class SpatialTriggerModule(nn.Module):
    def __init__(feature_dim, num_nodes, trigger_type="feature", ...)
    def set_trigger_nodes(trigger_node_indices)
    def generate_feature_trigger(node_features, adj_matrix) -> Tensor
    def generate_topology_trigger(adj_matrix, target_nodes) -> Tensor
    def compute_homophily_loss(modified_features, adj_matrix) -> Tensor
```

**Trigger Types:**
- `feature`: Modify node features at trigger nodes
- `topology`: Inject fake edges between trigger and target nodes
- `both`: Combined feature and topology modification

### NodeSelector

Selects optimal trigger nodes using centrality analysis.

```python
class NodeSelector:
    def __init__(adj_matrix, centrality_weights=None)
    def compute_pagerank() -> ndarray
    def compute_betweenness() -> ndarray
    def compute_eigenvector_centrality() -> ndarray
    def compute_composite_score() -> ndarray
    def select_trigger_nodes(num_nodes, target_nodes=None, strategy="centrality")
```

## Evaluation Metrics

### Attack Success Rate (ASR)

For regression tasks, ASR measures the proportion of triggered samples where predictions shift toward the target direction by more than a threshold.

```python
from cstba.utils.metrics import compute_asr_regression

asr, details = compute_asr_regression(
    model, clean_loader, poisoned_loader,
    target_direction="increase", device=device
)
```

### Benign Accuracy Drop (BA Drop)

Measures performance degradation on clean data.

```python
from cstba.utils.metrics import compute_ba_drop

ba_drop, metrics = compute_ba_drop(
    model, clean_loader, baseline_metrics, device
)
```

### Stealthiness Score

Combines temporal smoothness, spectral similarity, and feature distribution metrics.

```python
from cstba.utils.metrics import compute_stealthiness_score

score, components = compute_stealthiness_score(original_data, poisoned_data)
```

## Experiment Reproduction

Run the complete experiment suite:

```bash
# Make script executable
chmod +x run_experiments.sh

# Run all experiments on NYC dataset
./run_experiments.sh nyc

# Run on CHI dataset
./run_experiments.sh chi
```

This runs:
1. Poison rate comparison (5%, 8%, 10%, 12%, 15%)
2. Ablation study (temporal-only, spatial-only, joint)
3. Target region analysis
4. Trigger node count comparison

Results are saved to `experiments/` directory.

## Expected Results

Based on the algorithm design, expected performance on crime prediction datasets:

| Trigger Mode | ST-GCN | DCRNN | GWN | ASTGCN |
|--------------|--------|-------|-----|--------|
| Temporal-Only | 62% | 68% | 71% | 65% |
| Spatial-Only | 73% | 76% | 82% | 75% |
| **Joint (CSTBA)** | **91%** | **93%** | **97%** | **92%** |

BA Drop should remain < 1% for all modes.

## Troubleshooting

### Common Issues

**Q: Out of memory during training**
```python
# Reduce batch size
config.training.batch_size = 8
# Or use gradient accumulation
```

**Q: Low ASR even with joint trigger**
```python
# Increase poison rate
config.attack.poison_rate = 0.15
# Or increase bilevel iterations
config.training.bilevel_iterations = 200
```

**Q: High BA Drop**
```python
# Reduce trigger amplitude
config.temporal.amplitude_ratio = 0.05
# Enable stronger stealth constraints
config.synergy.lambda_stealth = 1.0
```

## Citation

If you use CSTBA in your research, please cite:

```bibtex
@article{cstba2024,
  title={Collaborative Spatio-Temporal Backdoor Trigger Algorithm for ST-GNN},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is for research purposes only. Use responsibly and ethically.
