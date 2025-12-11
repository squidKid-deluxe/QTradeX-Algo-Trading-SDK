# ============================================================================
# 2. ATTRACTOR - attractor.py
# ============================================================================

"""
The Attractor - Pull Toward Point

Archetype: Goal, magnetism, desire
Category: Field Dynamics
Complexity: 25 lines

Pulls activation toward a target point. Foundation of goal-directed
behavior, optimization, seeking.

Usage:
    >>> attractor = AttractorAtom(location=(8, 8), strength=0.2)
    >>> attractor.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class AttractorAtom:
    """Pull activation toward target point"""
    
    def __init__(self, location: tuple = None, strength: float = 0.2, radius: float = None):
        """
        Args:
            location: Target point to attract toward (None = center)
            strength: Attraction strength
            radius: Attraction radius (None = entire field)
        """
        self.location = location
        self.strength = strength
        self.radius = radius
    
    def apply(self, field: NDAnalogField):
        """Pull activation toward attractor"""
        if self.location is None:
            location = tuple(s // 2 for s in field.shape)
        else:
            location = self.location
        
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            
            # Compute distances from attractor
            dy = location[0] - y_indices
            dx = location[1] - x_indices
            distance = np.sqrt(dy**2 + dx**2) + 1e-8  # Avoid division by zero
            
            # Apply radius limit if specified
            if self.radius is not None:
                distance_mask = distance <= self.radius
            else:
                distance_mask = np.ones_like(distance, dtype=bool)
            
            # Compute attraction force (inverse square law)
            force = self.strength / (distance ** 2)
            force = np.where(distance_mask, force, 0)
            
            # Direction vectors
            dir_y = dy / distance
            dir_x = dx / distance
            
            # Create flow toward attractor
            new_activation = np.zeros_like(field.activation)
            
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    if distance_mask[i, j]:
                        # Move activation toward attractor
                        shift_y = int(dir_y[i, j] * force[i, j] * 10)
                        shift_x = int(dir_x[i, j] * force[i, j] * 10)
                        
                        new_i = np.clip(i + shift_y, 0, field.shape[0] - 1)
                        new_j = np.clip(j + shift_x, 0, field.shape[1] - 1)
                        
                        new_activation[new_i, new_j] += field.activation[i, j]
            
            # Blend with original
            field.activation = field.activation * 0.6 + new_activation * 0.4
        
        return field
    
    def set_location(self, location: tuple):
        """Update attractor location"""
        self.location = location
    
    def get_attraction_field(self, field: NDAnalogField) -> np.ndarray:
        """Get attraction strength at each point"""
        if self.location is None:
            location = tuple(s // 2 for s in field.shape)
        else:
            location = self.location
        
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            dy = location[0] - y_indices
            dx = location[1] - x_indices
            distance = np.sqrt(dy**2 + dx**2) + 1e-8
            
            force = self.strength / (distance ** 2)
            return force
        
        return np.zeros(field.shape)
    
    def __repr__(self):
        return f"AttractorAtom(loc={self.location}, str={self.strength:.2f})"
