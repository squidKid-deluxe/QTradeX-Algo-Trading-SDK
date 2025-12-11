# ============================================================================
# 4. THRESHOLD - threshold.py
# ============================================================================

"""
The Threshold - Activation Gates

Archetype: Decision point, breakthrough
Category: Temporal
Complexity: 22 lines

Gates activation based on threshold crossing. Foundation of discrete
decisions, phase transitions, breakthrough moments.

Usage:
    >>> threshold = ThresholdAtom(threshold=0.5, hysteresis=0.1)
    >>> threshold.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class ThresholdAtom:
    """Gate activation by threshold with optional hysteresis"""
    
    def __init__(self, threshold: float = 0.5, mode: str = 'binary', 
                 hysteresis: float = 0.0):
        """
        Args:
            threshold: Activation threshold
            mode: 'binary', 'amplify', 'suppress', 'rectify'
            hysteresis: Hysteresis width (prevents rapid switching)
        """
        self.threshold = threshold
        self.mode = mode
        self.hysteresis = hysteresis
        self.state = None  # For hysteresis tracking
        self.crossing_events = []  # Track threshold crossings
    
    def apply(self, field: NDAnalogField):
        """Apply threshold gating"""
        if self.state is None:
            self.state = np.zeros(field.shape, dtype=bool)
        
        # Track crossings
        if self.hysteresis > 0:
            # Hysteresis: different thresholds for on/off
            upper_threshold = self.threshold + self.hysteresis / 2
            lower_threshold = self.threshold - self.hysteresis / 2
            
            # Turn on if above upper threshold
            self.state = np.where(field.activation > upper_threshold, True, self.state)
            # Turn off if below lower threshold
            self.state = np.where(field.activation < lower_threshold, False, self.state)
        else:
            # Simple threshold
            self.state = field.activation > self.threshold
        
        # Apply threshold operation based on mode
        if self.mode == 'binary':
            # Binary: 1 if above threshold, 0 otherwise
            field.activation = np.where(self.state, 1.0, 0.0)
        
        elif self.mode == 'amplify':
            # Amplify values above threshold
            field.activation = np.where(self.state, 
                                       field.activation * 2.0, 
                                       field.activation)
        
        elif self.mode == 'suppress':
            # Suppress values below threshold
            field.activation = np.where(self.state, 
                                       field.activation, 
                                       field.activation * 0.1)
        
        elif self.mode == 'rectify':
            # Rectify: pass through if above, zero otherwise
            field.activation = np.where(self.state, 
                                       field.activation, 
                                       0.0)
        
        return field
    
    def detect_crossings(self, field: NDAnalogField) -> list:
        """Detect locations where threshold was just crossed"""
        if self.state is None:
            return []
        
        crossings = []
        current = field.activation > self.threshold
        
        # Find new activations
        new_activations = np.logical_and(current, np.logical_not(self.state))
        crossing_coords = np.argwhere(new_activations)
        
        for coord in crossing_coords:
            crossings.append(tuple(coord))
        
        return crossings
    
    def get_active_region_count(self) -> int:
        """Count number of distinct active regions"""
        if self.state is None:
            return 0
        return np.sum(self.state)
    
    def __repr__(self):
        return f"ThresholdAtom(thresh={self.threshold:.2f}, mode='{self.mode}')"


