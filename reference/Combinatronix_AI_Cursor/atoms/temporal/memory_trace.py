# ============================================================================
# 2. MEMORY_TRACE - memory_trace.py
# ============================================================================

"""
The Memory Trace - Accumulate History

Archetype: Past, karma, learning
Category: Temporal
Complexity: 20 lines

Accumulates activation over time, creating persistent memory traces.
Foundation of learning, habit formation, and karma.

Usage:
    >>> memory = MemoryTraceAtom(accumulation_rate=0.1, decay_rate=0.99)
    >>> memory.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class MemoryTraceAtom:
    """Accumulate activation into persistent memory"""
    
    def __init__(self, accumulation_rate: float = 0.1, decay_rate: float = 0.99, 
                 threshold: float = 0.1):
        """
        Args:
            accumulation_rate: How fast to accumulate new experiences
            decay_rate: How fast memories fade (1.0 = no decay)
            threshold: Minimum activation to accumulate
        """
        self.accumulation_rate = accumulation_rate
        self.decay_rate = decay_rate
        self.threshold = threshold
        self.trace = None
    
    def apply(self, field: NDAnalogField):
        """Accumulate current activation into memory trace"""
        # Initialize trace if needed
        if self.trace is None:
            self.trace = np.zeros_like(field.activation)
        
        # Apply decay to existing trace
        self.trace *= self.decay_rate
        
        # Accumulate new activation (only above threshold)
        significant_activation = np.where(field.activation > self.threshold,
                                         field.activation, 0)
        self.trace += significant_activation * self.accumulation_rate
        
        # Update field's memory layer
        field.memory = self.trace.copy()
        
        return field
    
    def get_trace_strength(self) -> float:
        """Get total strength of memory trace"""
        if self.trace is None:
            return 0.0
        return np.sum(np.abs(self.trace))
    
    def get_hotspots(self, top_k: int = 5) -> list:
        """Get locations with strongest memory traces"""
        if self.trace is None:
            return []
        
        flat_indices = np.argsort(self.trace.flatten())[-top_k:]
        locations = [np.unravel_index(idx, self.trace.shape) for idx in flat_indices]
        strengths = [self.trace[loc] for loc in locations]
        
        return list(zip(locations, strengths))
    
    def consolidate(self, consolidation_factor: float = 1.5):
        """Strengthen memory trace (like sleep consolidation)"""
        if self.trace is not None:
            self.trace *= consolidation_factor
            self.trace = np.clip(self.trace, 0, 1)
    
    def clear(self):
        """Erase memory trace"""
        self.trace = None
    
    def __repr__(self):
        return f"MemoryTraceAtom(accum={self.accumulation_rate:.2f}, decay={self.decay_rate:.2f})"

