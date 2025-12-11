# ============================================================================
# 1. VORTEX - vortex.py
# ============================================================================

"""
The Vortex - Circular Flow/Rotation

Archetype: Cycle, recursion, return
Category: Field Dynamics
Complexity: 22 lines

Creates rotating/circular flow patterns around a center point.
Foundation of recursive thinking, loops, obsessive thoughts.

Usage:
    >>> vortex = VortexAtom(center=(8, 8), angular_velocity=0.5)
    >>> vortex.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class VortexAtom:
    """Circular flow around center point"""
    
    def __init__(self, center: tuple = None, angular_velocity: float = 0.5, strength: float = 0.1):
        """
        Args:
            center: Center point of vortex (None = field center)
            angular_velocity: Speed of rotation
            strength: How strongly to rotate
        """
        self.center = center
        self.angular_velocity = angular_velocity
        self.strength = strength
        self.rotation_angle = 0
    
    def apply(self, field: NDAnalogField):
        """Create rotating flow pattern"""
        if self.center is None:
            center = tuple(s // 2 for s in field.shape)
        else:
            center = self.center
        
        # Update rotation
        self.rotation_angle += self.angular_velocity
        
        # Create rotation flow field
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            
            # Compute distances and angles from center
            dy = y_indices - center[0]
            dx = x_indices - center[1]
            
            # Tangential flow (perpendicular to radius)
            flow_x = -dy * self.strength
            flow_y = dx * self.strength
            
            # Apply rotation to activation
            # Shift activation in circular pattern
            shifted = np.zeros_like(field.activation)
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    # Compute new position after rotation
                    flow_i = int(i + flow_y[i, j])
                    flow_j = int(j + flow_x[i, j])
                    
                    # Wrap around boundaries
                    flow_i = flow_i % field.shape[0]
                    flow_j = flow_j % field.shape[1]
                    
                    shifted[flow_i, flow_j] += field.activation[i, j]
            
            # Blend with original
            field.activation = field.activation * 0.7 + shifted * 0.3
        
        return field
    
    def get_flow_field(self, field: NDAnalogField) -> tuple:
        """Get the flow vectors as separate arrays"""
        if self.center is None:
            center = tuple(s // 2 for s in field.shape)
        else:
            center = self.center
        
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            dy = y_indices - center[0]
            dx = x_indices - center[1]
            
            flow_x = -dy * self.strength
            flow_y = dx * self.strength
            
            return flow_y, flow_x
        
        return None, None
    
    def __repr__(self):
        return f"VortexAtom(w={self.angular_velocity:.2f}, str={self.strength:.2f})"
