"""
CSTBA Attack Implementation

Main attack class that orchestrates the complete backdoor attack pipeline:
- Trigger generator training
- Data poisoning
- Backdoor model training
- Attack evaluation
"""

from .cstba_attack import CSTBAAttack

__all__ = ["CSTBAAttack"]
