

# ============================================================================
# 5. SWAPPER - swapper.py
# ============================================================================

"""
The Swapper - C Combinator (Flip/Reverse)

Archetype: Reversal, perspective shift, empathy
Category: Combinatorial
Complexity: 16 lines

C f x y = f y x

The swapper reverses argument order. It's perspective-taking and empathy,
seeing things from the other side. Foundation of theory of mind.

Usage:
    >>> swapper = SwapperAtom(axis='horizontal')
    >>> swapper.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField, Comb
except ImportError:
    from combinatronix.core import NDAnalogField, Comb


class SwapperAtom:
    """C combinator - argument flip/perspective reversal"""
    
    def __init__(self, swap_type: str = 'spatial'):
        """
        Args:
            swap_type: 'spatial' (flip field), 'temporal' (reverse time), 'dual' (both)
        """
        self.swap_type = swap_type
        self.combinator = Comb('C')
    
    def apply(self, field: NDAnalogField, axis: str = 'both'):
        """Swap/flip field spatially
        
        C f x y: Changes argument order / perspective
        """
        if self.swap_type == 'spatial':
            if axis == 'horizontal':
                field.activation = np.fliplr(field.activation)
            elif axis == 'vertical':
                field.activation = np.flipud(field.activation)
            elif axis == 'both':
                field.activation = np.flip(field.activation)
            elif axis == 'diagonal':
                if len(field.shape) == 2:
                    field.activation = np.transpose(field.activation)
        
        elif self.swap_type == 'temporal':
            # Swap activation with memory (past becomes present)
            field.activation, field.memory = field.memory.copy(), field.activation.copy()
        
        elif self.swap_type == 'dual':
            # Both spatial and temporal swap
            field.activation = np.flip(field.activation)
            field.activation, field.memory = field.memory.copy(), field.activation.copy()
        
        return field
    
    def swap_fields(self, field_a: NDAnalogField, field_b: NDAnalogField):
        """Swap activations between two fields (pure C combinator on fields)"""
        field_a.activation, field_b.activation = field_b.activation.copy(), field_a.activation.copy()
        return field_a, field_b
    
    def perspective_shift(self, field: NDAnalogField, center: tuple):
        """Shift perspective to view from different center point"""
        if len(field.shape) == 2 and len(center) == 2:
            # Rotate field so center becomes new origin
            y, x = center
            h, w = field.shape
            
            # Create shifted view
            shifted = np.zeros_like(field.activation)
            for i in range(h):
                for j in range(w):
                    new_i = (i - y + h // 2) % h
                    new_j = (j - x + w // 2) % w
                    shifted[new_i, new_j] = field.activation[i, j]
            
            field.activation = shifted
        
        return field
    
    def __repr__(self):
        return f"SwapperAtom(type='{self.swap_type}')"
