# ============================================================================
# 1. BALANCER - balancer.py
# ============================================================================

"""
The Balancer - Resolve Contradictions

Archetype: Harmony, equilibrium, compromise
Category: Tension Resolvers
Complexity: 25 lines

Finds middle ground between opposing forces. Foundation of dialectical
thinking, compromise, and conflict resolution.

Usage:
    >>> balancer = BalancerAtom(equilibrium_rate=0.3)
    >>> balancer.apply(field_a, field_b)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class BalancerAtom:
    """Resolve contradictions by finding equilibrium"""
    
    def __init__(self, equilibrium_rate: float = 0.3, min_tension: float = 0.1):
        """
        Args:
            equilibrium_rate: How fast to move toward equilibrium
            min_tension: Minimum tension to trigger balancing
        """
        self.equilibrium_rate = equilibrium_rate
        self.min_tension = min_tension
        self.tension_history = []
    
    def apply(self, field_a: NDAnalogField, field_b: NDAnalogField = None):
        """Balance two opposing fields or internal tensions"""
        if field_b is None:
            # Balance internal tensions (positive vs negative)
            return self._balance_internal(field_a)
        else:
            # Balance two opposing fields
            return self._balance_external(field_a, field_b)
    
    def _balance_internal(self, field: NDAnalogField):
        """Balance internal contradictions within single field"""
        # Find regions of high variance (tension)
        if len(field.shape) == 2:
            # Compute local variance
            padded = np.pad(field.activation, 1, mode='edge')
            variance = np.zeros_like(field.activation)
            
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    neighborhood = padded[i:i+3, j:j+3]
                    variance[i, j] = np.var(neighborhood)
            
            # Apply smoothing to high-tension regions
            tension_mask = variance > self.min_tension
            
            # Smooth by averaging with neighbors
            for i in range(field.shape[0]):
                for j in range(field.shape[1]):
                    if tension_mask[i, j]:
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < field.shape[0] and 0 <= nj < field.shape[1]:
                                    neighbors.append(field.activation[ni, nj])
                        
                        neighbor_mean = np.mean(neighbors)
                        field.activation[i, j] = (field.activation[i, j] * (1 - self.equilibrium_rate) +
                                                 neighbor_mean * self.equilibrium_rate)
            
            # Track tension over time
            self.tension_history.append(np.mean(variance))
        
        return field
    
    def _balance_external(self, field_a: NDAnalogField, field_b: NDAnalogField):
        """Balance two opposing fields toward equilibrium"""
        # Compute tension between fields
        tension = np.abs(field_a.activation - field_b.activation)
        
        # Move both toward middle ground where tension is high
        high_tension = tension > self.min_tension
        
        equilibrium = (field_a.activation + field_b.activation) / 2
        
        field_a.activation = np.where(high_tension,
                                     field_a.activation * (1 - self.equilibrium_rate) + 
                                     equilibrium * self.equilibrium_rate,
                                     field_a.activation)
        
        field_b.activation = np.where(high_tension,
                                     field_b.activation * (1 - self.equilibrium_rate) + 
                                     equilibrium * self.equilibrium_rate,
                                     field_b.activation)
        
        # Track tension
        self.tension_history.append(np.mean(tension))
        
        return field_a
    
    def get_tension_level(self, field: NDAnalogField = None) -> float:
        """Get current tension level"""
        if len(self.tension_history) > 0:
            return self.tension_history[-1]
        return 0.0
    
    def detect_contradictions(self, field_a: NDAnalogField, field_b: NDAnalogField) -> list:
        """Find locations of strong contradictions"""
        tension = np.abs(field_a.activation - field_b.activation)
        contradiction_points = np.argwhere(tension > self.min_tension * 2)
        return [tuple(coord) for coord in contradiction_points]
    
    def __repr__(self):
        return f"BalancerAtom(rate={self.equilibrium_rate:.2f})"

