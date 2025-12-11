# ============================================================================
# 4. MIRROR - mirror.py
# ============================================================================

"""
The Mirror - Reflection/Symmetry

Archetype: Symmetry, self-recognition, duality
Category: Pattern Primitives
Complexity: 18 lines

The mirror creates reflections and symmetries. It's fundamental for
self-awareness and pattern recognition through comparison.

Usage:
    >>> mirror = MirrorAtom(axis='vertical')
    >>> mirror.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class MirrorAtom:
    """Reflection and symmetry operations"""
    
    def __init__(self, axis: str = 'vertical'):
        """
        Args:
            axis: 'vertical', 'horizontal', 'both', or 'diagonal'
        """
        self.axis = axis
    
    def apply(self, field: NDAnalogField, blend: float = 0.5):
        """Reflect field across axis"""
        original = field.activation.copy()
        
        if self.axis == 'vertical':
            reflected = np.fliplr(original)
        elif self.axis == 'horizontal':
            reflected = np.flipud(original)
        elif self.axis == 'both':
            reflected = np.flip(original)
        elif self.axis == 'diagonal':
            if len(field.shape) == 2:
                reflected = np.transpose(original)
            else:
                reflected = np.flip(original)
        else:
            reflected = original
        
        # Blend original with reflection
        field.activation = original * (1 - blend) + reflected * blend
        
        return field
    
    def compute_symmetry(self, field: NDAnalogField) -> float:
        """Compute how symmetric the field is"""
        original = field.activation
        if self.axis == 'vertical':
            reflected = np.fliplr(original)
        elif self.axis == 'horizontal':
            reflected = np.flipud(original)
        else:
            reflected = np.flip(original)
        
        # Compute similarity
        diff = np.abs(original - reflected)
        max_diff = np.max(np.abs(original)) + np.max(np.abs(reflected))
        if max_diff == 0:
            return 1.0
        symmetry = 1.0 - np.mean(diff) / max_diff
        return symmetry
    
    def __repr__(self):
        return f"MirrorAtom(axis='{self.axis}')"
