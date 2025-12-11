# ============================================================================
# 5. SUPERPOSER - superposer.py
# ============================================================================

"""
The Superposer - Overlay Multiple Fields

Archetype: Integration, synthesis, wholeness
Category: Multi-Field
Complexity: 15 lines

Combines multiple fields into integrated whole. Foundation of
multi-modal integration and holistic perception.

Usage:
    >>> superposer = SuperposerAtom(mode='weighted_sum')
    >>> result = superposer.superpose([field1, field2, field3])
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class SuperposerAtom:
    """Combine multiple fields into one"""
    
    def __init__(self, mode: str = 'weighted_sum', weights: list = None):
        """
        Args:
            mode: 'weighted_sum', 'max', 'min', 'product', 'average'
            weights: Optional weights for each field (for weighted_sum)
        """
        self.mode = mode
        self.weights = weights
    
    def superpose(self, *fields: NDAnalogField) -> NDAnalogField:
        """Combine multiple fields"""
        if len(fields) == 0:
            raise ValueError("Need at least one field to superpose")
        
        if len(fields) == 1:
            return fields[0].copy()
        
        # Create result field with shape of first field
        result = fields[0].copy()
        
        # Collect activations (resizing if needed)
        activations = []
        for field in fields:
            if field.shape == result.shape:
                activations.append(field.activation)
            else:
                # Resize to match
                resized = self._resize_to_match(field.activation, result.shape)
                activations.append(resized)
        
        # Combine based on mode
        if self.mode == 'weighted_sum':
            if self.weights is None:
                weights = [1.0 / len(fields)] * len(fields)
            else:
                weights = self.weights
            
            result.activation = sum(act * w for act, w in zip(activations, weights))
            
        elif self.mode == 'max':
            result.activation = np.maximum.reduce(activations)
            
        elif self.mode == 'min':
            result.activation = np.minimum.reduce(activations)
            
        elif self.mode == 'product':
            result.activation = np.prod(activations, axis=0)
            
        elif self.mode == 'average':
            result.activation = np.mean(activations, axis=0)
        
        return result
    
    def _resize_to_match(self, activation: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize activation to match target shape"""
        if activation.shape == target_shape:
            return activation
        
        # Simple resize using indexing
        result = np.zeros(target_shape)
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(activation.shape, target_shape))
        
        if len(min_shape) == 2:
            result[:min_shape[0], :min_shape[1]] = activation[:min_shape[0], :min_shape[1]]
        
        return result
    
    def superpose_with_gating(self, *fields: NDAnalogField, gate_threshold: float = 0.5) -> NDAnalogField:
        """Superpose only regions above threshold"""
        result = self.superpose(*fields)
        
        # Apply gating
        result.activation = np.where(result.activation > gate_threshold,
                                    result.activation,
                                    0)
        
        return result
    
    def __repr__(self):
        return f"SuperposerAtom(mode='{self.mode}')"
