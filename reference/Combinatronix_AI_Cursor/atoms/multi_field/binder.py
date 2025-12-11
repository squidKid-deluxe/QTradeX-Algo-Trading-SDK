# ============================================================================
# 3. BINDER - binder.py
# ============================================================================

"""
The Binder - Link Patterns Across Fields

Archetype: Association, connection, binding
Category: Multi-Field
Complexity: 20 lines

Creates associative links between patterns in different fields.
Foundation of memory binding, feature integration, and association.

Usage:
    >>> binder = BinderAtom(binding_strength=0.8)
    >>> binder.bind(field_a, field_b, pattern_a_loc, pattern_b_loc)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class BinderAtom:
    """Create associative bindings between fields"""
    
    def __init__(self, binding_strength: float = 0.8, decay_rate: float = 0.99):
        """
        Args:
            binding_strength: Strength of new bindings
            decay_rate: How fast bindings decay over time
        """
        self.binding_strength = binding_strength
        self.decay_rate = decay_rate
        self.bindings = []  # List of (field_a_pattern, field_b_pattern, strength)
    
    def bind(self, field_a: NDAnalogField, field_b: NDAnalogField, 
            region_a: tuple = None, region_b: tuple = None):
        """Create binding between patterns in two fields"""
        # Extract patterns
        if region_a is None:
            pattern_a = field_a.activation.copy()
        else:
            pattern_a = field_a.activation[region_a].copy()
        
        if region_b is None:
            pattern_b = field_b.activation.copy()
        else:
            pattern_b = field_b.activation[region_b].copy()
        
        # Create binding
        binding = {
            'pattern_a': pattern_a,
            'pattern_b': pattern_b,
            'region_a': region_a,
            'region_b': region_b,
            'strength': self.binding_strength
        }
        
        self.bindings.append(binding)
        
        return self
    
    def activate(self, field_a: NDAnalogField, field_b: NDAnalogField):
        """Activate bound patterns (pattern completion)"""
        # Check which bindings are activated by field_a
        for binding in self.bindings:
            # Compute match between field_a and stored pattern_a
            if binding['region_a'] is not None:
                current_a = field_a.activation[binding['region_a']]
            else:
                current_a = field_a.activation
            
            # Flatten for comparison
            flat_current = current_a.flatten()
            flat_stored = binding['pattern_a'].flatten()
            min_len = min(len(flat_current), len(flat_stored))
            
            # Compute similarity
            if min_len > 0:
                similarity = np.dot(flat_current[:min_len], flat_stored[:min_len])
                similarity /= (np.linalg.norm(flat_current[:min_len]) * 
                             np.linalg.norm(flat_stored[:min_len]) + 1e-8)
                
                # If similar, activate bound pattern in field_b
                if similarity > 0.5:
                    activation_strength = similarity * binding['strength']
                    
                    if binding['region_b'] is not None:
                        field_b.activation[binding['region_b']] += (
                            binding['pattern_b'] * activation_strength
                        )
                    else:
                        # Resize if needed
                        if binding['pattern_b'].shape == field_b.shape:
                            field_b.activation += binding['pattern_b'] * activation_strength
        
        return field_b
    
    def decay_bindings(self):
        """Apply decay to all bindings"""
        for binding in self.bindings:
            binding['strength'] *= self.decay_rate
        
        # Remove very weak bindings
        self.bindings = [b for b in self.bindings if b['strength'] > 0.1]
    
    def get_binding_count(self) -> int:
        """Get number of active bindings"""
        return len(self.bindings)
    
    def clear_bindings(self):
        """Remove all bindings"""
        self.bindings.clear()
    
    def __repr__(self):
        return f"BinderAtom(bindings={len(self.bindings)}, str={self.binding_strength:.2f})"


