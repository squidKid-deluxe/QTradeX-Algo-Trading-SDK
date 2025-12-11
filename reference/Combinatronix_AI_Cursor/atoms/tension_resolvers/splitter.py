# ============================================================================
# 2. SPLITTER - splitter.py
# ============================================================================

"""
The Splitter - Disambiguate

Archetype: Analysis, discrimination, clarity
Category: Tension Resolvers
Complexity: 22 lines

Separates ambiguous overlapping patterns. Foundation of disambiguation,
clarification, and precise discrimination.

Usage:
    >>> splitter = SplitterAtom(separation_strength=0.3)
    >>> splitter.apply(field)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class SplitterAtom:
    """Separate and disambiguate overlapping patterns"""
    
    def __init__(self, separation_strength: float = 0.3, min_overlap: float = 0.2):
        """
        Args:
            separation_strength: How strongly to separate patterns
            min_overlap: Minimum overlap to trigger splitting
        """
        self.separation_strength = separation_strength
        self.min_overlap = min_overlap
        self.split_locations = []
    
    def apply(self, field: NDAnalogField, pattern_a: np.ndarray = None, pattern_b: np.ndarray = None):
        """Split ambiguous patterns"""
        if pattern_a is not None and pattern_b is not None:
            # Split two known overlapping patterns
            return self._split_patterns(field, pattern_a, pattern_b)
        else:
            # Auto-detect and split ambiguous regions
            return self._split_ambiguous(field)
    
    def _split_patterns(self, field: NDAnalogField, pattern_a: np.ndarray, pattern_b: np.ndarray):
        """Split two overlapping patterns"""
        # Detect overlap
        overlap = np.minimum(pattern_a, pattern_b)
        high_overlap = overlap > self.min_overlap
        
        # Emphasize differences where overlap is high
        diff_a = pattern_a - overlap
        diff_b = pattern_b - overlap
        
        # Create split patterns
        split_a = np.where(high_overlap, 
                          pattern_a + diff_a * self.separation_strength,
                          pattern_a)
        
        split_b = np.where(high_overlap,
                          pattern_b + diff_b * self.separation_strength,
                          pattern_b)
        
        # Apply the split that better matches current field
        similarity_a = np.sum(field.activation * split_a)
        similarity_b = np.sum(field.activation * split_b)
        
        if similarity_a > similarity_b:
            field.activation = split_a
        else:
            field.activation = split_b
        
        # Track split locations
        self.split_locations = [tuple(coord) for coord in np.argwhere(high_overlap)]
        
        return field
    
    def _split_ambiguous(self, field: NDAnalogField):
        """Auto-detect and split ambiguous regions"""
        if len(field.shape) == 2:
            # Find regions with moderate activation (ambiguous)
            ambiguous = np.logical_and(field.activation > 0.3, field.activation < 0.7)
            
            # Enhance differences in ambiguous regions
            # Compute gradient magnitude
            grad_y, grad_x = np.gradient(field.activation)
            grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            
            # In ambiguous regions, amplify gradients (separate patterns)
            field.activation = np.where(ambiguous,
                                       field.activation + grad_magnitude * self.separation_strength,
                                       field.activation)
            
            self.split_locations = [tuple(coord) for coord in np.argwhere(ambiguous)]
        
        return field
    
    def create_distinguishing_features(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> tuple:
        """Extract features that distinguish two patterns"""
        # Compute unique features for each pattern
        overlap = np.minimum(pattern_a, pattern_b)
        
        feature_a = pattern_a - overlap
        feature_b = pattern_b - overlap
        
        # Amplify distinguishing features
        feature_a = feature_a * (1.0 + self.separation_strength)
        feature_b = feature_b * (1.0 + self.separation_strength)
        
        return feature_a, feature_b
    
    def get_ambiguity_map(self, field: NDAnalogField) -> np.ndarray:
        """Get map showing where disambiguation is needed"""
        if len(field.shape) == 2:
            # High ambiguity = moderate activation with low gradients
            grad_y, grad_x = np.gradient(field.activation)
            grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            
            moderate_activation = np.logical_and(field.activation > 0.3, 
                                                field.activation < 0.7)
            low_gradient = grad_magnitude < 0.2
            
            ambiguity = np.logical_and(moderate_activation, low_gradient).astype(float)
            return ambiguity
        
        return np.zeros(field.shape)
    
    def __repr__(self):
        return f"SplitterAtom(strength={self.separation_strength:.2f})"


