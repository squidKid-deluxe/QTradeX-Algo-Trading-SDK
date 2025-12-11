

# ============================================================================
# 3. BARRIER - barrier.py
# ============================================================================

"""
The Barrier - Block Propagation

Archetype: Boundary, limit, taboo
Category: Field Dynamics
Complexity: 18 lines

Blocks signal propagation at specified regions. Foundation of constraints,
boundaries, forbidden zones.

Usage:
    >>> barrier = BarrierAtom(region=[(3, 3), (5, 5)])
    >>> barrier.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class BarrierAtom:
    """Block propagation in specified regions"""
    
    def __init__(self, region: list = None, permeability: float = 0.0):
        """
        Args:
            region: List of (start, end) tuples defining barrier region
            permeability: How much signal can pass through (0.0 = total block)
        """
        self.region = region or []
        self.permeability = permeability
        self.barrier_mask = None
    
    def apply(self, field: NDAnalogField):
        """Block activation in barrier regions"""
        if self.barrier_mask is None:
            self._create_barrier_mask(field)
        
        # Block activation (with optional permeability)
        field.activation = np.where(self.barrier_mask,
                                    field.activation * self.permeability,
                                    field.activation)
        
        # Also increase resistance in barrier regions
        field.resistance = np.where(self.barrier_mask,
                                   field.resistance * 10.0,
                                   field.resistance)
        
        return field
    
    def _create_barrier_mask(self, field: NDAnalogField):
        """Create boolean mask for barrier regions"""
        self.barrier_mask = np.zeros(field.shape, dtype=bool)
        
        for region_def in self.region:
            if len(region_def) == 2:
                start, end = region_def
                if len(field.shape) == 2:
                    self.barrier_mask[start[0]:end[0], start[1]:end[1]] = True
    
    def add_barrier_region(self, start: tuple, end: tuple):
        """Add a new barrier region"""
        self.region.append((start, end))
        self.barrier_mask = None  # Force regeneration
    
    def add_circular_barrier(self, center: tuple, radius: float):
        """Add circular barrier region"""
        # Will be applied when mask is created
        self.region.append(('circle', center, radius))
        self.barrier_mask = None
    
    def clear_barriers(self):
        """Remove all barriers"""
        self.region.clear()
        self.barrier_mask = None
    
    def __repr__(self):
        return f"BarrierAtom(regions={len(self.region)}, perm={self.permeability:.2f})"


