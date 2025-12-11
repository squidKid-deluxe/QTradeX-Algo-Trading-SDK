# ============================================================================
# 4. DAMPER - damper.py
# ============================================================================

"""
The Damper - Reduce Overflow

Archetype: Restraint, control, inhibition
Category: Tension Resolvers
Complexity: 18 lines

Reduces excessive activation. Foundation of inhibition, self-control,
and system regulation.

Usage:
    >>> damper = DamperAtom(threshold=0.8, damping_rate=0.5)
    >>> damper.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class DamperAtom:
    """Reduce excessive activation (inhibition)"""
    
    def __init__(self, threshold: float = 0.8, damping_rate: float = 0.5, 
                 mode: str = 'soft'):
        """
        Args:
            threshold: Activation level that triggers damping
            damping_rate: How much to reduce overflow
            mode: 'soft' (gradual), 'hard' (sharp cutoff), 'normalize' (rescale)
        """
        self.threshold = threshold
        self.damping_rate = damping_rate
        self.mode = mode
        self.total_damped = 0.0
    
    def apply(self, field: NDAnalogField):
        """Apply damping to reduce overflow"""
        overflow = field.activation > self.threshold
        
        if self.mode == 'soft':
            # Gradual damping above threshold
            excess = np.maximum(field.activation - self.threshold, 0)
            damped_excess = excess * (1 - self.damping_rate)
            field.activation = np.where(overflow,
                                       self.threshold + damped_excess,
                                       field.activation)
        
        elif self.mode == 'hard':
            # Sharp cutoff at threshold
            field.activation = np.clip(field.activation, -np.inf, self.threshold)
        
        elif self.mode == 'normalize':
            # Rescale to fit within threshold
            max_val = np.max(field.activation)
            if max_val > self.threshold:
                field.activation *= (self.threshold / max_val)
        
        # Track damped energy
        damped = np.sum(overflow.astype(float))
        self.total_damped += damped
        
        return field
    
    def apply_spatial_damping(self, field: NDAnalogField, center: tuple, radius: float):
        """Apply damping strongest near center, weaker at edges"""
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            
            # Distance from center
            dy = y_indices - center[0]
            dx = x_indices - center[1]
            distance = np.sqrt(dy**2 + dx**2)
            
            # Damping strength inversely proportional to distance
            damping_strength = np.clip(1.0 - distance / radius, 0, 1) * self.damping_rate
            
            # Apply spatially-varying damping
            overflow = field.activation > self.threshold
            excess = np.maximum(field.activation - self.threshold, 0)
            damped_excess = excess * (1 - damping_strength)
            
            field.activation = np.where(overflow,
                                       self.threshold + damped_excess,
                                       field.activation)
        
        return field
    
    def get_overflow_regions(self, field: NDAnalogField) -> list:
        """Get locations where overflow occurred"""
        overflow = field.activation > self.threshold
        overflow_coords = np.argwhere(overflow)
        return [tuple(coord) for coord in overflow_coords]
    
    def __repr__(self):
        return f"DamperAtom(thresh={self.threshold:.2f}, mode='{self.mode}')"


