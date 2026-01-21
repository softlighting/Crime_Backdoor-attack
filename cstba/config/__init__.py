"""
CSTBA Configuration Management

Handles loading and validation of attack parameters from YAML config files.
"""

from .config_loader import load_config, CSTBAConfig

__all__ = ["load_config", "CSTBAConfig"]
