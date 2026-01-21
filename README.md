# STHSL - Crime Prediction with Spatio-Temporal Hypergraph Learning

A PyTorch implementation of [Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction](https://ieeexplore.ieee.org/document/9835423) (ICDE 2022).

This repository also includes the **CSTBA** (Collaborative Spatio-Temporal Backdoor Trigger Algorithm) module for analyzing backdoor vulnerabilities in spatio-temporal graph neural networks.

## Project Structure

```
Crime_Backdoor-attack/
├── Datasets/                    # Crime datasets
│   ├── NYC_crime/              # New York City crime data
│   │   ├── trn.pkl            # Training (16x16x608x4)
│   │   ├── val.pkl            # Validation (16x16x30x4)
│   │   └── tst.pkl            # Test (16x16x30x4)
│   └── CHI_crime/              # Chicago crime data
│       ├── trn.pkl            # Training (14x12x609x4)
│       ├── val.pkl            # Validation (14x12x30x4)
│       └── tst.pkl            # Test (14x12x30x4)
├── cstba/                       # CSTBA analysis module
│   ├── config/                 # Configuration files
│   ├── modules/                # Core algorithm modules
│   ├── utils/                  # Utility functions
│   ├── attacks/                # Attack interface
│   └── scripts/                # Scripts
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── model.py                     # STHSL model
├── train.py                     # Training script
├── engine.py                    # Training engine
├── DataHandler.py               # Data loading
├── Params.py                    # Arguments
├── utils.py                     # Utilities
└── quick_demo.py                # Quick demonstration
```

## Environment Setup

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.22.0
- SciPy >= 1.7.0
- NetworkX >= 2.6.0 (for CSTBA)
- scikit-learn >= 1.0.0
- PyYAML >= 6.0
- tqdm >= 4.62.0

## Quick Start

### 1. Train STHSL Model

```bash
# NYC dataset
python train.py --data NYC --epoch 25 --batch 16

# Chicago dataset
python train.py --data CHI --epoch 25 --batch 16
```

### 2. Test Trained Model

```bash
python test.py --data NYC --checkpoint ./Save/NYC/your_model.pth
```

### 3. Run Quick Demo (All Models Comparison)

```bash
# Fast demo (5 epochs, shows module functionality)
python quick_demo.py --data NYC --epochs 5

# Full training comparison
python quick_demo.py --data NYC --epochs 25 --full
```

## Data Format

| Dataset | Grid | Train Days | Val Days | Test Days | Categories |
|---------|------|------------|----------|-----------|------------|
| NYC | 16x16 (256 nodes) | 608 | 30 | 30 | 4 |
| CHI | 14x12 (168 nodes) | 609 | 30 | 30 | 4 |

**Crime Categories:** Larceny, Burglary, Robbery, Assault

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr` | 0.001 | Learning rate |
| `--batch` | 16 | Batch size |
| `--epoch` | 25 | Training epochs |
| `--latdim` | 16 | Embedding dimension |
| `--temporalRange` | 30 | Temporal window |
| `--hyperNum` | 128 | Number of hyperedges |
| `--data` | NYC | Dataset (NYC/CHI) |

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_temporal_trigger.py -v
python -m pytest tests/test_spatial_trigger.py -v
python -m pytest tests/test_integration.py -v
```

## Expected Results

### NYC Dataset

| Metric | Clean Model |
|--------|-------------|
| RMSE | ~1.2-1.4 |
| MAE | ~0.6-0.8 |
| MAPE | ~45-55% |

### CHI Dataset

| Metric | Clean Model |
|--------|-------------|
| RMSE | ~0.9-1.1 |
| MAE | ~0.5-0.7 |
| MAPE | ~40-50% |

## CSTBA Module

The CSTBA module provides tools for analyzing model vulnerabilities. See `cstba/README.md` for detailed documentation.

### Quick Usage

```python
from cstba.config import load_config
from cstba.modules import TemporalTriggerModule, SpatialTriggerModule

# Load configuration
config = load_config('cstba/config/cstba_config.yaml', dataset='NYC')

# Initialize modules
ttm = TemporalTriggerModule(config.temporal)
stm = SpatialTriggerModule(config.spatial, num_nodes=256, feature_dim=4)
```

## Documentation

- `docs/project_structure_analysis.md` - Codebase analysis
- `docs/dataset_format.md` - Dataset format details
- `docs/code_quality_report.md` - Code quality report
- `cstba/README.md` - CSTBA module documentation

## Troubleshooting

**CUDA Out of Memory:**
```bash
python train.py --batch 8  # Reduce batch size
```

**Module Import Error:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Citation

```bibtex
@inproceedings{sthsl2022,
  title={Spatial-Temporal Hypergraph Self-Supervised Learning for Crime Prediction},
  author={...},
  booktitle={ICDE},
  year={2022}
}
```

## License

This project is for research purposes only.
