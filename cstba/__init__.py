"""
CSTBA: Collaborative Spatio-Temporal Backdoor Trigger Algorithm

A novel backdoor attack algorithm specifically designed for
Spatio-Temporal Graph Neural Networks (ST-GNNs) in urban crime prediction.

Core Components:
- TTM (Temporal Trigger Module): Generates stealthy temporal trigger patterns
- STM (Spatial Trigger Module): Identifies key nodes and modifies graph topology/features
- Synergy Optimizer: Joint optimization for "1+1>2" collaborative effect
- Trigger Injection Engine: Injects triggers into training/test data

Author: CSTBA Research Team
Version: 1.0.0
"""

from .attacks.cstba_attack import CSTBAAttack

__version__ = "1.0.0"
__all__ = ["CSTBAAttack"]
