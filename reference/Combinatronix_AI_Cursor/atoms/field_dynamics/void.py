# ============================================================================
# 5. VOID - void.py
# ============================================================================

"""
The Void - Absorb Activation

Archetype: Nothingness, forgetting, death
Category: Field Dynamics
Complexity: 15 lines

Absorbs activation in specified regions. Foundation of forgetting,
death, reset, cessation.

Usage:
    >>> void = VoidAtom(location=(4, 4), radius=2)
    >>> void.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class VoidAtom:
    """Absorb/erase activation in region"""
    
    def __init__(self, location: tuple = None, radius: float = 2.0, absorption_rate: float = 1.0):
        """
        Args:
            location: Center of void (None = field center)
            radius: Absorption radius
            absorption_rate: How fast to absorb (1.0 = complete, 0.5 = half)
        """
        self.location = location
        self.radius = radius
        self.absorption_rate = absorption_rate
        self.total_absorbed = 0.0
    
    def apply(self, field: NDAnalogField):
        """Absorb activation in void region"""
        if self.location is None:
            location = tuple(s // 2 for s in field.shape)
        else:
            location = self.location
        
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            
            # Compute distance from void center
            dy = y_indices - location[0]
            dx = x_indices - location[1]
            distance = np.sqrt(dy**2 + dx**2)
            
            # Create absorption mask (stronger near center)
            absorption_strength = np.clip(1.0 - distance / self.radius, 0, 1)
            absorption_strength *= self.absorption_rate
            
            # Track absorbed energy
            absorbed = np.sum(field.activation * absorption_strength)
            self.total_absorbed += absorbed
            
            # Apply absorption
            field.activation *= (1.0 - absorption_strength)
        
        return field
    
    def absorb_region(self, field: NDAnalogField, region: tuple):
        """Completely absorb a specific region"""
        before = np.sum(field.activation[region])
        field.activation[region] = 0
        self.total_absorbed += before
        return field
    
    def get_absorbed_amount(self) -> float:
        """Get total energy absorbed by void"""
        return self.total_absorbed
    
    def reset_absorption(self):
        """Reset absorption counter"""
        self.total_absorbed = 0.0
    
    def __repr__(self):
        return f"VoidAtom(radius={self.radius:.1f}, rate={self.absorption_rate:.2f})"

