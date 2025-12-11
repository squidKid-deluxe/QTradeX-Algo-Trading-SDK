# ============================================================================
# 5. AMPLIFIER - amplifier.py
# ============================================================================

"""
The Amplifier - Boost Weak Signals

Archetype: Enhancement, emphasis, focus
Category: Tension Resolvers
Complexity: 20 lines

Amplifies weak signals to make them detectable. Foundation of
sensitivity enhancement, signal detection, and focus.

Usage:
    >>> amplifier = AmplifierAtom(threshold=0.2, gain=2.0)
    >>> amplifier.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class AmplifierAtom:
    """Amplify weak signals for detection"""
    
    def __init__(self, threshold: float = 0.2, gain: float = 2.0, 
                 mode: str = 'linear'):
        """
        Args:
            threshold: Signal level below which to amplify
            gain: Amplification factor
            mode: 'linear', 'exponential', 'adaptive'
        """
        self.threshold = threshold
        self.gain = gain
        self.mode = mode
        self.amplified_regions = []
    
    def apply(self, field: NDAnalogField):
        """Amplify weak signals"""
        weak_signals = np.logical_and(field.activation > 0, 
                                      field.activation < self.threshold)
        
        if self.mode == 'linear':
            # Linear amplification
            field.activation = np.where(weak_signals,
                                       field.activation * self.gain,
                                       field.activation)
        
        elif self.mode == 'exponential':
            # Exponential amplification (more for weaker signals)
            amplification = self.gain ** (1 - field.activation / self.threshold)
            field.activation = np.where(weak_signals,
                                       field.activation * amplification,
                                       field.activation)
        
        elif self.mode == 'adaptive':
            # Adaptive: amplify based on neighborhood
            if len(field.shape) == 2:
                padded = np.pad(field.activation, 1, mode='edge')
                local_amplification = np.zeros_like(field.activation)
                
                for i in range(field.shape[0]):
                    for j in range(field.shape[1]):
                        neighborhood = padded[i:i+3, j:j+3]
                        neighbor_max = np.max(neighborhood)
                        
                        # Amplify more if neighbors are strong
                        if weak_signals[i, j] and neighbor_max > self.threshold:
                            boost = self.gain * (neighbor_max / self.threshold)
                            local_amplification[i, j] = boost
                        else:
                            local_amplification[i, j] = 1.0
                
                field.activation *= local_amplification
        
        # Track amplified regions
        self.amplified_regions = [tuple(coord) for coord in np.argwhere(weak_signals)]
        
        # Clip to prevent overflow
        field.activation = np.clip(field.activation, 0, 1)
        
        return field
    
    def selective_amplify(self, field: NDAnalogField, locations: list, local_gain: float = None):
        """Amplify specific locations"""
        if local_gain is None:
            local_gain = self.gain
        
        for location in locations:
            if field._valid_coord(location):
                field.activation[location] *= local_gain
        
        field.activation = np.clip(field.activation, 0, 1)
        return field
    
    def detect_weak_signals(self, field: NDAnalogField) -> list:
        """Detect locations of weak but present signals"""
        weak_signals = np.logical_and(field.activation > 0.01,
                                      field.activation < self.threshold)
        weak_coords = np.argwhere(weak_signals)
        
        # Return with signal strength
        results = [(tuple(coord), field.activation[tuple(coord)]) 
                  for coord in weak_coords]
        return results
    
    def __repr__(self):
        return f"AmplifierAtom(thresh={self.threshold:.2f}, gain={self.gain:.1f})"


