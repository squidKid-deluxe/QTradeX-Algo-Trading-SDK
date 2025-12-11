# ============================================================================
# 3. ECHO - echo.py
# ============================================================================

"""
The Echo - Decaying Repetition

Archetype: Memory, persistence, resonance
Category: Pattern Primitives
Complexity: 25 lines

The echo captures the past and replays it with decay. It's the foundation
of all memory systems - patterns persist but gradually fade.

Usage:
    >>> echo = EchoAtom(decay_rate=0.9, depth=5)
    >>> echo.apply(field)
"""

import numpy as np
from collections import deque
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class EchoAtom:
    """Decaying repetition of patterns"""
    
    def __init__(self, decay_rate: float = 0.9, depth: int = 5):
        self.decay_rate = decay_rate
        self.depth = depth
        self.history = deque(maxlen=depth)
    
    def apply(self, field: NDAnalogField):
        """Create decaying echo of current state"""
        # Store current state
        self.history.append(field.activation.copy())
        
        # Apply echoes with decay
        echo_field = np.zeros_like(field.activation)
        for i, past_state in enumerate(self.history):
            age = len(self.history) - i - 1
            decay_factor = self.decay_rate ** age
            echo_field += past_state * decay_factor
        
        # Normalize and blend with current
        if len(self.history) > 0:
            echo_field /= len(self.history)
            field.activation = field.activation * 0.7 + echo_field * 0.3
        
        return field
    
    def get_echo_strength(self) -> float:
        """Get total strength of all echoes"""
        if not self.history:
            return 0.0
        total = sum(np.sum(np.abs(state)) for state in self.history)
        return total / len(self.history)
    
    def clear(self):
        """Clear echo history"""
        self.history.clear()
    
    def __repr__(self):
        return f"EchoAtom(decay={self.decay_rate:.2f}, depth={self.depth})"


