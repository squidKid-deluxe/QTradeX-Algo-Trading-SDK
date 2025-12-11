# ============================================================================
# 3. FILLER - filler.py
# ============================================================================

"""
The Filler - Fill Conceptual Gaps

Archetype: Creation, invention, naming
Category: Tension Resolvers
Complexity: 20 lines

Creates new patterns to fill gaps in understanding. Foundation of
concept invention, creativity, and gap-filling inference.

Usage:
    >>> filler = FillerAtom(creativity=0.5)
    >>> new_pattern = filler.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class FillerAtom:
    """Fill conceptual gaps with invented patterns"""
    
    def __init__(self, creativity: float = 0.5, gap_threshold: float = 0.1):
        """
        Args:
            creativity: How novel the fill patterns should be (0=conservative, 1=creative)
            gap_threshold: Minimum gap size to trigger filling
        """
        self.creativity = creativity
        self.gap_threshold = gap_threshold
        self.filled_gaps = []
        self.invented_patterns = {}
    
    def apply(self, field: NDAnalogField):
        """Detect and fill conceptual gaps"""
        # Detect gaps (low activation regions surrounded by high activation)
        gaps = self._detect_gaps(field)
        
        # Fill each gap
        for gap_location in gaps:
            fill_pattern = self._create_fill_pattern(field, gap_location)
            self._apply_fill(field, gap_location, fill_pattern)
            
            # Track filled gap
            self.filled_gaps.append(gap_location)
        
        return field
    
    def _detect_gaps(self, field: NDAnalogField) -> list:
        """Find regions that need filling"""
        gaps = []
        
        if len(field.shape) == 2:
            for i in range(1, field.shape[0] - 1):
                for j in range(1, field.shape[1] - 1):
                    center = field.activation[i, j]
                    
                    # Get neighbors
                    neighbors = [
                        field.activation[i-1, j], field.activation[i+1, j],
                        field.activation[i, j-1], field.activation[i, j+1]
                    ]
                    neighbor_mean = np.mean(neighbors)
                    
                    # Gap = low center with high neighbors
                    if center < self.gap_threshold and neighbor_mean > 0.3:
                        gaps.append((i, j))
        
        return gaps
    
    def _create_fill_pattern(self, field: NDAnalogField, gap_location: tuple) -> float:
        """Invent pattern to fill gap"""
        i, j = gap_location
        
        # Get surrounding context
        if len(field.shape) == 2:
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < field.shape[0] and 0 <= nj < field.shape[1]:
                        neighbors.append(field.activation[ni, nj])
            
            # Conservative fill: interpolate from neighbors
            conservative_fill = np.mean(neighbors)
            
            # Creative fill: add novelty
            creative_fill = conservative_fill + np.random.normal(0, self.creativity * 0.3)
            
            # Blend based on creativity parameter
            fill_value = (conservative_fill * (1 - self.creativity) + 
                         creative_fill * self.creativity)
            
            return np.clip(fill_value, 0, 1)
        
        return 0.0
    
    def _apply_fill(self, field: NDAnalogField, location: tuple, fill_value: float):
        """Apply fill pattern at location"""
        field.activation[location] = fill_value
    
    def invent_bridging_concept(self, field_a: NDAnalogField, field_b: NDAnalogField) -> np.ndarray:
        """Invent concept that bridges two existing concepts"""
        # Create intermediate pattern
        bridge = (field_a.activation + field_b.activation) / 2
        
        # Add creative novelty
        noise = np.random.normal(0, self.creativity * 0.2, bridge.shape)
        bridge += noise
        bridge = np.clip(bridge, 0, 1)
        
        # Store invented pattern
        pattern_id = f"bridge_{len(self.invented_patterns)}"
        self.invented_patterns[pattern_id] = bridge
        
        return bridge
    
    def get_filled_gaps(self) -> list:
        """Get list of filled gap locations"""
        return self.filled_gaps.copy()
    
    def __repr__(self):
        return f"FillerAtom(creativity={self.creativity:.2f}, gaps_filled={len(self.filled_gaps)})"

