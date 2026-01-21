"""
CSTBA Core Modules

Contains the main components for backdoor trigger generation:
- TemporalTriggerModule (TTM): Time-series trigger generation
- SpatialTriggerModule (STM): Graph-based spatial trigger generation
- SynergyOptimizer: Joint optimization of TTM and STM
- TriggerInjectionEngine: Data poisoning implementation
"""

from .temporal_trigger import TemporalTriggerModule
from .spatial_trigger import SpatialTriggerModule
from .synergy_optimizer import SynergyOptimizer
from .trigger_injection import TriggerInjectionEngine

__all__ = [
    "TemporalTriggerModule",
    "SpatialTriggerModule",
    "SynergyOptimizer",
    "TriggerInjectionEngine"
]
