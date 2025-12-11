# ============================================================================
# 1. WITNESS - witness.py
# ============================================================================

"""
The Witness - I Combinator (Identity)

Archetype: Observer, consciousness, attention
Category: Combinatorial
Complexity: 12 lines

I x = x

The witness observes without changing. It's pure awareness, the foundation
of consciousness. In field terms, it can amplify attention to what exists.

Usage:
    >>> witness = WitnessAtom(amplification=1.0)
    >>> witness.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField, Comb, app, Val
except ImportError:
    from combinatronix.core import NDAnalogField, Comb, app, Val


class WitnessAtom:
    """Identity operation - pure observation/attention"""
    
    def __init__(self, amplification: float = 1.0):
        """
        Args:
            amplification: How much to amplify observed patterns (1.0 = identity)
        """
        self.amplification = amplification
        self.combinator = Comb('I')
    
    def apply(self, field: NDAnalogField, mask: np.ndarray = None):
        """Observe/amplify field activation
        
        Args:
            field: Field to observe
            mask: Optional attention mask (where to observe)
        """
        if mask is not None:
            # Selective attention - only amplify where mask is true
            field.activation = np.where(mask, 
                                       field.activation * self.amplification,
                                       field.activation)
        else:
            # Uniform observation/amplification
            field.activation *= self.amplification
        
        return field
    
    def observe(self, field: NDAnalogField) -> np.ndarray:
        """Pure observation - return copy without modification"""
        return field.activation.copy()
    
    def __repr__(self):
        return f"WitnessAtom(amp={self.amplification:.2f})"
