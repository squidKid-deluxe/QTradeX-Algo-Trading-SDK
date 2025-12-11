
# ============================================================================
# 1. PULSE - pulse.py
# ============================================================================

"""
The Pulse - Oscillating Activation

Archetype: Time, heartbeat, rhythm
Category: Pattern Primitives
Complexity: 15 lines

The pulse is the fundamental oscillator. It represents the passage of time
and creates rhythmic patterns. All temporal phenomena build on this.

Usage:
    >>> pulse = PulseAtom(frequency=1.0, phase=0.0)
    >>> pulse.apply(field, location=(4, 4))
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class PulseAtom:
    """Oscillating activation pattern"""
    
    def __init__(self, frequency: float = 1.0, phase: float = 0.0, amplitude: float = 1.0):
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude
        self.time = 0
    
    def apply(self, field: NDAnalogField, location: tuple = None):
        """Apply oscillating pulse to field"""
        self.time += 1
        value = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time + self.phase)
        
        if location and field._valid_coord(location):
            field.activation[location] += value
        else:
            # Apply to entire field
            field.activation += value * 0.1
        
        return field
    
    def reset(self):
        """Reset time counter"""
        self.time = 0
    
    def set_frequency(self, frequency: float):
        """Change oscillation frequency"""
        self.frequency = frequency
    
    def __repr__(self):
        return f"PulseAtom(freq={self.frequency:.2f}, phase={self.phase:.2f})"

