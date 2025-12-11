# ============================================================================
# 3. RHYTHM - rhythm.py
# ============================================================================

"""
The Rhythm - Periodic Patterns

Archetype: Beat, tempo, synchronization
Category: Temporal
Complexity: 18 lines

Creates periodic activation patterns. Foundation of rhythmic behavior,
synchronization, and temporal coordination.

Usage:
    >>> rhythm = RhythmAtom(frequency=1.0, phase=0.0)
    >>> rhythm.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class RhythmAtom:
    """Generate periodic activation patterns"""
    
    def __init__(self, frequency: float = 1.0, phase: float = 0.0, 
                 amplitude: float = 0.2, pattern: str = 'sine'):
        """
        Args:
            frequency: Oscillation frequency
            phase: Phase offset
            amplitude: Oscillation amplitude
            pattern: 'sine', 'square', 'triangle', 'sawtooth'
        """
        self.frequency = frequency
        self.phase = phase
        self.amplitude = amplitude
        self.pattern = pattern
        self.time = 0
    
    def apply(self, field: NDAnalogField):
        """Apply rhythmic modulation to field"""
        self.time += 1
        
        # Generate rhythm value
        t = 2 * np.pi * self.frequency * self.time + self.phase
        
        if self.pattern == 'sine':
            value = np.sin(t)
        elif self.pattern == 'square':
            value = 1.0 if np.sin(t) > 0 else -1.0
        elif self.pattern == 'triangle':
            value = 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1
        elif self.pattern == 'sawtooth':
            value = 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))
        else:
            value = np.sin(t)
        
        # Apply rhythmic modulation
        modulation = 1.0 + value * self.amplitude
        field.activation *= modulation
        
        return field
    
    def apply_spatial_rhythm(self, field: NDAnalogField, wave_vector: tuple = (1, 0)):
        """Apply spatiotemporal rhythm pattern"""
        self.time += 1
        
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            
            # Create traveling wave
            k_y, k_x = wave_vector
            spatial_phase = k_y * y_indices + k_x * x_indices
            temporal_phase = 2 * np.pi * self.frequency * self.time + self.phase
            
            wave = np.sin(spatial_phase + temporal_phase)
            modulation = 1.0 + wave * self.amplitude
            
            field.activation *= modulation
        
        return field
    
    def synchronize_with(self, other_rhythm: 'RhythmAtom', coupling: float = 0.1):
        """Synchronize this rhythm with another (phase locking)"""
        phase_diff = other_rhythm.phase - self.phase
        self.phase += coupling * np.sin(phase_diff)
    
    def reset(self):
        """Reset time counter"""
        self.time = 0
    
    def __repr__(self):
        return f"RhythmAtom(freq={self.frequency:.2f}, pattern='{self.pattern}')"


