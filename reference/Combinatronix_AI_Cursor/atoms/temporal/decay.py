# ============================================================================
# 5. DECAY - decay.py
# ============================================================================

"""
The Decay - Exponential Fading

Archetype: Time's arrow, forgetting, entropy
Category: Temporal
Complexity: 15 lines

Applies exponential decay to activation. Foundation of forgetting,
cooling, entropy, and the arrow of time.

Usage:
    >>> decay = DecayAtom(decay_rate=0.95, selective=True)
    >>> decay.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class DecayAtom:
    """Apply exponential decay to field activation"""
    
    def __init__(self, decay_rate: float = 0.95, min_value: float = 0.01, 
                 selective: bool = False):
        """
        Args:
            decay_rate: Decay factor per step (0.95 = 5% decay)
            min_value: Minimum value before setting to zero
            selective: If True, only decay positive values
        """
        self.decay_rate = decay_rate
        self.min_value = min_value
        self.selective = selective
        self.total_decayed = 0.0
    
    def apply(self, field: NDAnalogField):
        """Apply exponential decay"""
        before = np.sum(np.abs(field.activation))
        
        if self.selective:
            # Only decay positive values
            field.activation = np.where(field.activation > 0,
                                       field.activation * self.decay_rate,
                                       field.activation)
        else:
            # Decay all values toward zero
            field.activation *= self.decay_rate
        
        # Set very small values to zero
        field.activation = np.where(np.abs(field.activation) < self.min_value,
                                   0.0,
                                   field.activation)
        
        after = np.sum(np.abs(field.activation))
        self.total_decayed += (before - after)
        
        return field
    
    def apply_spatial_decay(self, field: NDAnalogField, center: tuple, rate_factor: float = 1.0):
        """Apply decay that varies with distance from center"""
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            
            # Compute distance from center
            dy = y_indices - center[0]
            dx = x_indices - center[1]
            distance = np.sqrt(dy**2 + dx**2)
            
            # Decay rate increases with distance
            local_decay = self.decay_rate ** (1.0 + distance * rate_factor * 0.1)
            field.activation *= local_decay
        
        return field
    
    def get_half_life(self) -> float:
        """Compute half-life (steps until 50% decay)"""
        if self.decay_rate >= 1.0:
            return float('inf')
        return np.log(0.5) / np.log(self.decay_rate)
    
    def get_total_decayed(self) -> float:
        """Get total energy decayed"""
        return self.total_decayed
    
    def reset_counter(self):
        """Reset decay counter"""
        self.total_decayed = 0.0
    
    def __repr__(self):
        return f"DecayAtom(rate={self.decay_rate:.3f}, half_life={self.get_half_life():.1f})"


