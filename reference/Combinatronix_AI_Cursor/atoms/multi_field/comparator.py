# ============================================================================
# 4. COMPARATOR - comparator.py
# ============================================================================

"""
The Comparator - Measure Field Differences

Archetype: Sameness, difference, recognition
Category: Multi-Field
Complexity: 18 lines

Computes similarity/difference between fields. Foundation of
pattern recognition, comparison, and discrimination.

Usage:
    >>> comparator = ComparatorAtom(metric='correlation')
    >>> similarity = comparator.compare(field_a, field_b)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class ComparatorAtom:
    """Measure similarity/difference between fields"""
    
    def __init__(self, metric: str = 'correlation', normalize: bool = True):
        """
        Args:
            metric: 'correlation', 'euclidean', 'cosine', 'difference'
            normalize: Whether to normalize before comparison
        """
        self.metric = metric
        self.normalize = normalize
        self.comparison_history = []
    
    def compare(self, field_a: NDAnalogField, field_b: NDAnalogField) -> float:
        """Compute overall similarity between fields"""
        # Ensure same shape
        if field_a.shape != field_b.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(field_a.shape, field_b.shape))
            act_a = field_a.activation[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else field_a.activation
            act_b = field_b.activation[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else field_b.activation
        else:
            act_a = field_a.activation
            act_b = field_b.activation
        
        # Normalize if requested
        if self.normalize:
            act_a = act_a / (np.linalg.norm(act_a) + 1e-8)
            act_b = act_b / (np.linalg.norm(act_b) + 1e-8)
        
        # Compute similarity based on metric
        if self.metric == 'correlation':
            flat_a = act_a.flatten()
            flat_b = act_b.flatten()
            correlation = np.corrcoef(flat_a, flat_b)[0, 1]
            similarity = correlation if not np.isnan(correlation) else 0.0
            
        elif self.metric == 'cosine':
            flat_a = act_a.flatten()
            flat_b = act_b.flatten()
            dot_product = np.dot(flat_a, flat_b)
            similarity = dot_product / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-8)
            
        elif self.metric == 'euclidean':
            distance = np.linalg.norm(act_a - act_b)
            max_distance = np.sqrt(np.prod(act_a.shape)) * 2  # Normalize
            similarity = 1.0 - (distance / max_distance)
            
        elif self.metric == 'difference':
            diff = np.abs(act_a - act_b)
            similarity = 1.0 - np.mean(diff)
        
        self.comparison_history.append(similarity)
        return similarity
    
    def get_difference_map(self, field_a: NDAnalogField, field_b: NDAnalogField) -> np.ndarray:
        """Get spatial map of differences"""
        if field_a.shape != field_b.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(field_a.shape, field_b.shape))
            if len(min_shape) == 2:
                diff = np.abs(field_a.activation[:min_shape[0], :min_shape[1]] - 
                            field_b.activation[:min_shape[0], :min_shape[1]])
            else:
                diff = np.abs(field_a.activation - field_b.activation)
        else:
            diff = np.abs(field_a.activation - field_b.activation)
        
        return diff
    
    def find_matching_regions(self, field_a: NDAnalogField, field_b: NDAnalogField,
                             threshold: float = 0.8) -> list:
        """Find regions with high local similarity"""
        diff_map = self.get_difference_map(field_a, field_b)
        similarity_map = 1.0 - diff_map / (np.max(diff_map) + 1e-8)
        
        matching = similarity_map > threshold
        locations = np.argwhere(matching)
        
        return [tuple(loc) for loc in locations]
    
    def __repr__(self):
        return f"ComparatorAtom(metric='{self.metric}')"

