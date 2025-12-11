
# ============================================================================
# 2. SEED - seed.py
# ============================================================================

"""
The Seed - Point Source That Spreads

Archetype: Origin, source, beginning
Category: Pattern Primitives
Complexity: 20 lines

The seed is a point of origination. It injects energy at a location and
allows it to spread naturally through the field. Represents causation.

Usage:
    >>> seed = SeedAtom(spread_radius=2)
    >>> seed.apply(field, location=(4, 4), strength=1.0)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class SeedAtom:
    """Point source that spreads activation"""
    
    def __init__(self, spread_radius: int = 1, spread_factor: float = 0.5):
        self.spread_radius = spread_radius
        self.spread_factor = spread_factor
    
    def apply(self, field: NDAnalogField, location: tuple, strength: float = 1.0):
        """Inject and spread from point"""
        if not field._valid_coord(location):
            return field
        
        # Inject at center
        field.activation[location] += strength
        
        # Spread to neighbors within radius
        if len(field.shape) == 2:
            y, x = location
            for dy in range(-self.spread_radius, self.spread_radius + 1):
                for dx in range(-self.spread_radius, self.spread_radius + 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < field.shape[0] and 0 <= nx < field.shape[1]):
                        distance = np.sqrt(dy**2 + dx**2)
                        if distance > 0 and distance <= self.spread_radius:
                            spread_amount = strength * self.spread_factor / distance
                            field.activation[ny, nx] += spread_amount
        
        return field
    
    def inject_pattern(self, field: NDAnalogField, pattern: np.ndarray, location: tuple):
        """Inject a complete pattern at location"""
        field.inject_pattern(pattern, location)
        return field
    
    def __repr__(self):
        return f"SeedAtom(radius={self.spread_radius})"


