# ============================================================================
# 2. SELECTOR - selector.py
# ============================================================================

"""
The Selector - K Combinator (Constant/Choice)

Archetype: Will, choice, decision
Category: Combinatorial
Complexity: 18 lines

K x y = x

The selector chooses first, ignores second. It's the foundation of decision
making and filtering. In fields, it selects specific patterns.

Usage:
    >>> selector = SelectorAtom()
    >>> result = selector.apply(field_a, field_b)  # Returns field_a
"""

import numpy as np
try:
    from ...core import NDAnalogField, Comb
except ImportError:
    from combinatronix.core import NDAnalogField, Comb


class SelectorAtom:
    """Constant operation - selection/filtering"""
    
    def __init__(self, selection_threshold: float = 0.5):
        """
        Args:
            selection_threshold: Threshold for binary selection
        """
        self.selection_threshold = selection_threshold
        self.combinator = Comb('K')
    
    def apply(self, field_a: NDAnalogField, field_b: NDAnalogField = None) -> NDAnalogField:
        """Select first field, ignore second (K combinator)
        
        Returns field_a, ignoring field_b (true K combinator behavior)
        """
        # Pure K combinator: always return first argument
        return field_a
    
    def select_by_threshold(self, field: NDAnalogField, keep_above: bool = True):
        """Binary selection based on threshold"""
        if keep_above:
            field.activation = np.where(field.activation > self.selection_threshold,
                                       field.activation, 0)
        else:
            field.activation = np.where(field.activation <= self.selection_threshold,
                                       field.activation, 0)
        return field
    
    def select_top_k(self, field: NDAnalogField, k: int):
        """Select top k activation values"""
        flat = field.activation.flatten()
        threshold = np.partition(flat, -k)[-k] if k < len(flat) else 0
        field.activation = np.where(field.activation >= threshold,
                                   field.activation, 0)
        return field
    
    def __repr__(self):
        return f"SelectorAtom(threshold={self.selection_threshold:.2f})"

