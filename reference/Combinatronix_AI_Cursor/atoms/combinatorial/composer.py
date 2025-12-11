


# ============================================================================
# 4. COMPOSER - composer.py
# ============================================================================

"""
The Composer - B Combinator (Composition)

Archetype: Causation, sequence, consequence
Category: Combinatorial
Complexity: 15 lines

B f g x = f (g x)

The composer chains operations sequentially. It's causation and consequence,
the foundation of multi-step reasoning and pipelines.

Usage:
    >>> composer = ComposerAtom()
    >>> composer.apply(field, [operation_1, operation_2, operation_3])
"""

import numpy as np
try:
    from ...core import NDAnalogField, Comb
except ImportError:
    from combinatronix.core import NDAnalogField, Comb


class ComposerAtom:
    """B combinator - function composition/sequencing"""
    
    def __init__(self, preserve_intermediate: bool = False):
        """
        Args:
            preserve_intermediate: Whether to store intermediate results
        """
        self.preserve_intermediate = preserve_intermediate
        self.intermediate_results = []
        self.combinator = Comb('B')
    
    def apply(self, field: NDAnalogField, operations: list):
        """Compose operations sequentially
        
        B f g x: First apply g to x, then apply f to result
        """
        self.intermediate_results.clear()
        
        # Apply operations in sequence
        for i, op in enumerate(operations):
            if self.preserve_intermediate:
                self.intermediate_results.append(field.activation.copy())
            
            # Apply operation
            if callable(op):
                result = op(field)
                if isinstance(result, NDAnalogField):
                    field = result
                elif isinstance(result, np.ndarray):
                    field.activation = result
        
        if self.preserve_intermediate:
            self.intermediate_results.append(field.activation.copy())
        
        return field
    
    def compose_two(self, field: NDAnalogField, op_outer: callable, op_inner: callable):
        """Classic B combinator: B f g x = f(g(x))"""
        # Apply inner operation first
        inner_result = op_inner(field)
        if isinstance(inner_result, NDAnalogField):
            field = inner_result
        elif isinstance(inner_result, np.ndarray):
            field.activation = inner_result
        
        # Apply outer operation to result
        outer_result = op_outer(field)
        if isinstance(outer_result, NDAnalogField):
            field = outer_result
        elif isinstance(outer_result, np.ndarray):
            field.activation = outer_result
        
        return field
    
    def get_intermediate_results(self) -> list:
        """Get stored intermediate results"""
        return self.intermediate_results.copy()
    
    def __repr__(self):
        return f"ComposerAtom(preserve={self.preserve_intermediate})"
