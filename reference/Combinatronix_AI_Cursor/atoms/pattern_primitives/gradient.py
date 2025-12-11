

# ============================================================================
# 5. GRADIENT - gradient.py
# ============================================================================

"""
The Gradient - Flow from High to Low

Archetype: Attraction, seeking, tendency
Category: Pattern Primitives
Complexity: 22 lines

The gradient creates directed flow based on differences. It's the foundation
of goal-seeking behavior and optimization.

Usage:
    >>> gradient = GradientAtom(strength=0.1)
    >>> gradient.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class GradientAtom:
    """Flow along gradient (high to low or low to high)"""
    
    def __init__(self, strength: float = 0.1, direction: str = 'ascent'):
        """
        Args:
            strength: How strongly to follow gradient
            direction: 'ascent' (toward high) or 'descent' (toward low)
        """
        self.strength = strength
        self.direction = direction
    
    def apply(self, field: NDAnalogField):
        """Apply gradient flow to field"""
        # Compute gradient
        gradients = np.gradient(field.activation)
        
        # Compute gradient magnitude and direction
        if len(gradients) == 2:
            grad_y, grad_x = gradients
            grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            
            # Flow along gradient
            if self.direction == 'ascent':
                # Move toward higher values
                field.activation += grad_magnitude * self.strength
            else:
                # Move toward lower values (descent)
                field.activation -= grad_magnitude * self.strength
        
        return field
    
    def get_gradient_field(self, field: NDAnalogField) -> np.ndarray:
        """Get gradient magnitude as separate field"""
        gradients = np.gradient(field.activation)
        if len(gradients) == 2:
            grad_y, grad_x = gradients
            return np.sqrt(grad_y**2 + grad_x**2)
        return np.abs(gradients[0])
    
    def find_peaks(self, field: NDAnalogField, threshold: float = 0.5) -> list:
        """Find local maxima in field"""
        peaks = []
        
        if len(field.shape) == 2:
            # Simple peak detection for 2D
            for i in range(1, field.shape[0] - 1):
                for j in range(1, field.shape[1] - 1):
                    center = field.activation[i, j]
                    if center > threshold:
                        # Check if it's a local maximum
                        neighbors = [
                            field.activation[i-1, j], field.activation[i+1, j],
                            field.activation[i, j-1], field.activation[i, j+1]
                        ]
                        if all(center >= n for n in neighbors):
                            peaks.append((i, j, center))
        
        return peaks
    
    def __repr__(self):
        return f"GradientAtom(strength={self.strength:.2f}, dir='{self.direction}')"
