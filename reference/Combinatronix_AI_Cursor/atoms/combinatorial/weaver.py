



# ============================================================================
# 3. WEAVER - weaver.py
# ============================================================================

"""
The Weaver - S Combinator (Application/Synthesis)

Archetype: Synthesis, integration, parallel processing
Category: Combinatorial
Complexity: 20 lines

S f g x = f x (g x)

The weaver applies both operations and combines results. It's parallel
processing and integration. The foundation of multi-perspective thinking.

Usage:
    >>> weaver = WeaverAtom(combination='add')
    >>> weaver.apply(field, operation_a, operation_b)
"""

import numpy as np
try:
    from ...core import NDAnalogField, Comb
except ImportError:
    from combinatronix.core import NDAnalogField, Comb


class WeaverAtom:
    """S combinator - parallel application and combination"""
    
    def __init__(self, combination: str = 'add', weight_a: float = 0.5, weight_b: float = 0.5):
        """
        Args:
            combination: How to combine results ('add', 'multiply', 'max', 'min')
            weight_a: Weight for first operation result
            weight_b: Weight for second operation result
        """
        self.combination = combination
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.combinator = Comb('S')
    
    def apply(self, field: NDAnalogField, op_a: callable = None, op_b: callable = None):
        """Apply both operations and weave results
        
        S f g x: Apply f to x, apply g to x, combine results
        """
        if op_a is None:
            op_a = lambda f: f.activation
        if op_b is None:
            op_b = lambda f: f.activation
        
        # Save original
        original = field.activation.copy()
        
        # Apply first operation
        result_a = op_a(field)
        if not isinstance(result_a, np.ndarray):
            result_a = field.activation.copy()
        
        # Reset and apply second operation
        field.activation = original.copy()
        result_b = op_b(field)
        if not isinstance(result_b, np.ndarray):
            result_b = field.activation.copy()
        
        # Weave results
        if self.combination == 'add':
            field.activation = result_a * self.weight_a + result_b * self.weight_b
        elif self.combination == 'multiply':
            field.activation = result_a * result_b
        elif self.combination == 'max':
            field.activation = np.maximum(result_a, result_b)
        elif self.combination == 'min':
            field.activation = np.minimum(result_a, result_b)
        
        return field
    
    def weave_fields(self, field_a: NDAnalogField, field_b: NDAnalogField) -> NDAnalogField:
        """Directly weave two fields"""
        result = field_a.copy()
        
        if self.combination == 'add':
            result.activation = field_a.activation * self.weight_a + field_b.activation * self.weight_b
        elif self.combination == 'multiply':
            result.activation = field_a.activation * field_b.activation
        elif self.combination == 'max':
            result.activation = np.maximum(field_a.activation, field_b.activation)
        elif self.combination == 'min':
            result.activation = np.minimum(field_a.activation, field_b.activation)
        
        return result
    
    def __repr__(self):
        return f"WeaverAtom(comb='{self.combination}')"
