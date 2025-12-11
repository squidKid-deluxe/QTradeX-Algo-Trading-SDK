# ============================================================================
# 2. TRANSLATOR - translator.py
# ============================================================================

"""
The Translator - Map Between Fields

Archetype: Metaphor, isomorphism, cross-modal mapping
Category: Multi-Field
Complexity: 25 lines

Maps patterns from one field to another with transformation.
Foundation of metaphor, cross-modal perception, and abstraction.

Usage:
    >>> translator = TranslatorAtom(scale_factor=0.5)
    >>> translator.translate(source_field, target_field)
"""

import numpy as np
from scipy import ndimage
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class TranslatorAtom:
    """Map patterns between different fields"""
    
    def __init__(self, scale_factor: float = 1.0, rotation: float = 0.0,
                 transformation: str = 'linear'):
        """
        Args:
            scale_factor: Size scaling between fields
            rotation: Rotation angle in degrees
            transformation: 'linear', 'nonlinear', 'topological'
        """
        self.scale_factor = scale_factor
        self.rotation = rotation
        self.transformation = transformation
        self.translation_map = None
    
    def translate(self, source: NDAnalogField, target: NDAnalogField, 
                 strength: float = 1.0):
        """Translate pattern from source to target field"""
        # Extract and transform pattern
        transformed = self._transform_pattern(source.activation)
        
        # Map to target field size
        if transformed.shape != target.shape:
            transformed = self._resize_pattern(transformed, target.shape)
        
        # Blend with target
        target.activation = target.activation * (1 - strength) + transformed * strength
        
        return target
    
    def _transform_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Apply transformation to pattern"""
        if self.transformation == 'linear':
            # Simple linear transformation
            if self.rotation != 0:
                pattern = ndimage.rotate(pattern, self.rotation, reshape=False, order=1)
            
        elif self.transformation == 'nonlinear':
            # Nonlinear transformation (compression/expansion)
            pattern = np.sign(pattern) * np.abs(pattern) ** self.scale_factor
            
        elif self.transformation == 'topological':
            # Preserve topology while transforming
            # Use distance transform to preserve structure
            binary = pattern > np.mean(pattern)
            if np.any(binary):
                dist = ndimage.distance_transform_edt(binary)
                pattern = dist / (np.max(dist) + 1e-8)
        
        return pattern
    
    def _resize_pattern(self, pattern: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize pattern to target shape"""
        if len(pattern.shape) == 2 and len(target_shape) == 2:
            # Use zoom for 2D
            zoom_factors = (target_shape[0] / pattern.shape[0],
                          target_shape[1] / pattern.shape[1])
            resized = ndimage.zoom(pattern, zoom_factors, order=1)
            return resized
        else:
            # Simple cropping or padding for other cases
            result = np.zeros(target_shape)
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(pattern.shape, target_shape))
            
            if len(min_shape) == 2:
                result[:min_shape[0], :min_shape[1]] = pattern[:min_shape[0], :min_shape[1]]
            
            return result
    
    def create_mapping(self, source: NDAnalogField, target: NDAnalogField):
        """Create explicit mapping between fields"""
        self.translation_map = {}
        
        # Create coordinate mapping
        if len(source.shape) == 2 and len(target.shape) == 2:
            for i in range(source.shape[0]):
                for j in range(source.shape[1]):
                    # Map source coordinate to target coordinate
                    target_i = int(i * target.shape[0] / source.shape[0])
                    target_j = int(j * target.shape[1] / source.shape[1])
                    self.translation_map[(i, j)] = (target_i, target_j)
    
    def apply_mapping(self, source: NDAnalogField, target: NDAnalogField):
        """Apply pre-computed mapping"""
        if self.translation_map is None:
            self.create_mapping(source, target)
        
        for source_coord, target_coord in self.translation_map.items():
            if source._valid_coord(source_coord) and target._valid_coord(target_coord):
                target.activation[target_coord] += source.activation[source_coord]
    
    def __repr__(self):
        return f"TranslatorAtom(scale={self.scale_factor:.2f}, rot={self.rotation:.1f}°)"


