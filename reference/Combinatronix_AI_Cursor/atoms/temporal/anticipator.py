# ============================================================================
# 1. ANTICIPATOR - anticipator.py
# ============================================================================

"""
The Anticipator - Predict Future State

Archetype: Future, foresight, expectation
Category: Temporal
Complexity: 25 lines

Predicts next state based on recent history. Foundation of expectation,
planning, and predictive coding.

Usage:
    >>> anticipator = AnticipatorAtom(history_depth=3)
    >>> prediction = anticipator.predict(field)
"""

import numpy as np
from collections import deque
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class AnticipatorAtom:
    """Predict future field state from history"""
    
    def __init__(self, history_depth: int = 3, prediction_weight: float = 0.3):
        """
        Args:
            history_depth: How many past states to use for prediction
            prediction_weight: How much to blend prediction with current
        """
        self.history_depth = history_depth
        self.prediction_weight = prediction_weight
        self.history = deque(maxlen=history_depth)
        self.last_prediction = None
    
    def apply(self, field: NDAnalogField):
        """Predict and blend with current state"""
        # Store current state
        self.history.append(field.activation.copy())
        
        # Generate prediction
        if len(self.history) >= 2:
            prediction = self.predict(field)
            self.last_prediction = prediction
            
            # Blend prediction with current state
            field.activation = (field.activation * (1 - self.prediction_weight) + 
                              prediction * self.prediction_weight)
        
        return field
    
    def predict(self, field: NDAnalogField) -> np.ndarray:
        """Generate prediction of next state"""
        if len(self.history) < 2:
            return field.activation.copy()
        
        # Simple linear extrapolation from recent history
        recent = list(self.history)
        
        # Compute velocity (change between last two states)
        velocity = recent[-1] - recent[-2]
        
        # If we have more history, compute acceleration
        if len(recent) >= 3:
            prev_velocity = recent[-2] - recent[-3]
            acceleration = velocity - prev_velocity
            # Predict: current + velocity + 0.5*acceleration
            prediction = recent[-1] + velocity + 0.5 * acceleration
        else:
            # Predict: current + velocity
            prediction = recent[-1] + velocity
        
        return prediction
    
    def get_prediction_error(self, field: NDAnalogField) -> float:
        """Compute error between last prediction and actual state"""
        if self.last_prediction is None or len(self.history) < 2:
            return 0.0
        
        actual = self.history[-1]
        error = np.mean(np.abs(actual - self.last_prediction))
        return error
    
    def get_surprise(self, field: NDAnalogField) -> np.ndarray:
        """Get surprise map (difference from expectation)"""
        if self.last_prediction is None:
            return np.zeros_like(field.activation)
        
        return np.abs(field.activation - self.last_prediction)
    
    def reset(self):
        """Clear history"""
        self.history.clear()
        self.last_prediction = None
    
    def __repr__(self):
        return f"AnticipatorAtom(depth={self.history_depth}, weight={self.prediction_weight:.2f})"



