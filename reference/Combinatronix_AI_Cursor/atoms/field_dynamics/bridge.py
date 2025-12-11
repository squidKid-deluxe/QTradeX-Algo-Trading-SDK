# ============================================================================
# 4. BRIDGE - bridge.py
# ============================================================================

"""
The Bridge - Connect Distant Regions

Archetype: Connection, analogy, metaphor
Category: Field Dynamics
Complexity: 20 lines

Creates connections between distant regions of the field. Foundation
of analogical thinking, remote association, metaphor.

Usage:
    >>> bridge = BridgeAtom()
    >>> bridge.connect(field, (2, 2), (6, 6), strength=0.5)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class BridgeAtom:
    """Connect distant regions with direct links"""
    
    def __init__(self, bidirectional: bool = True, transfer_rate: float = 0.3):
        """
        Args:
            bidirectional: Whether bridge transfers both directions
            transfer_rate: How much activation transfers across bridge
        """
        self.bidirectional = bidirectional
        self.transfer_rate = transfer_rate
        self.bridges = []  # List of (region_a, region_b, strength) tuples
    
    def apply(self, field: NDAnalogField):
        """Apply all established bridges"""
        for bridge in self.bridges:
            region_a, region_b, strength = bridge
            self._transfer_across_bridge(field, region_a, region_b, strength)
    
    def _transfer_across_bridge(self, field: NDAnalogField, region_a: tuple, region_b: tuple, strength: float):
        """Transfer activation between two regions"""
        # Get activation in both regions
        act_a = field.activation[region_a]
        act_b = field.activation[region_b]
        
        # Calculate transfer amount
        transfer_amount = self.transfer_rate * strength
        
        if self.bidirectional:
            # Bidirectional transfer (equalization)
            diff = act_a - act_b
            field.activation[region_a] -= diff * transfer_amount
            field.activation[region_b] += diff * transfer_amount
        else:
            # Unidirectional transfer (A -> B)
            transfer = act_a * transfer_amount
            field.activation[region_a] -= transfer
            field.activation[region_b] += transfer
        
        return field
    
    def connect(self, field: NDAnalogField, region_a: tuple, region_b: tuple, strength: float = 1.0):
        """Create a bridge between two regions
        
        Args:
            field: Field to operate on
            region_a: Coordinates or slice for first region
            region_b: Coordinates or slice for second region
            strength: Connection strength
        """
        self.bridges.append((region_a, region_b, strength))
        return self.apply(field)
    
    def connect_path(self, field: NDAnalogField, path: list, strength: float = 1.0):
        """Create bridges along a path of regions"""
        for i in range(len(path) - 1):
            self.bridges.append((path[i], path[i + 1], strength))
        return self.apply(field)
    
    def get_bridge_strength(self, region_a: tuple, region_b: tuple) -> float:
        """Get strength of bridge between two regions"""
        for bridge in self.bridges:
            if (bridge[0] == region_a and bridge[1] == region_b) or \
               (bridge[1] == region_a and bridge[0] == region_b):
                return bridge[2]
        return 0.0
    
    def clear_bridges(self):
        """Remove all bridges"""
        self.bridges.clear()
    
    def __repr__(self):
        return f"BridgeAtom(bridges={len(self.bridges)}, bidir={self.bidirectional})"

