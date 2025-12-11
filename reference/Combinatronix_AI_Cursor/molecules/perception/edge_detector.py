# ============================================================================
# EdgeDetector - Visual Edge Detection
# ============================================================================

"""
EdgeDetector - Detect edges/boundaries in visual fields

Composition: Gradient + Threshold
Category: Perception
Complexity: Molecule (50-200 lines)

Detects edges by computing gradient magnitude and applying threshold filtering.
This is fundamental for visual perception, object recognition, and spatial
reasoning. Edges represent boundaries between different regions or objects.

Example:
    >>> detector = EdgeDetector(threshold=0.3, strength=1.0)
    >>> result = detector.process(field)
    >>> edges = detector.get_edge_map()
"""

import numpy as np
try:
    from ...atoms.pattern_primitives import GradientAtom
    from ...atoms.temporal import ThresholdAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.pattern_primitives import GradientAtom
    from combinatronix.atoms.temporal import ThresholdAtom
    from combinatronix.core import NDAnalogField


class EdgeDetector:
    """Detect edges using gradient magnitude and thresholding"""
    
    def __init__(self, threshold: float = 0.3, strength: float = 1.0, 
                 hysteresis: float = 0.05, mode: str = 'rectify'):
        """
        Args:
            threshold: Edge detection threshold (0.0-1.0)
            strength: Gradient computation strength
            hysteresis: Hysteresis to prevent noise (0.0-0.2)
            mode: Threshold mode ('binary', 'rectify', 'amplify')
        """
        self.gradient = GradientAtom(strength=strength, direction='ascent')
        self.threshold = ThresholdAtom(
            threshold=threshold, 
            mode=mode, 
            hysteresis=hysteresis
        )
        
        # State tracking
        self.edge_map = None
        self.gradient_magnitude = None
        self.edge_count = 0
        self.last_processing_time = 0
    
    def process(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Detect edges in the field
        
        Args:
            field: Input field to process
            **kwargs: Additional parameters
            
        Returns:
            Field with edge information in activation layer
        """
        # Store original for comparison
        original_activation = field.activation.copy()
        
        # Step 1: Compute gradient magnitude
        self.gradient_magnitude = self.gradient.get_gradient_field(field)
        
        # Step 2: Apply gradient to field activation
        field.activation = self.gradient_magnitude.copy()
        
        # Step 3: Apply threshold to get clean edges
        self.threshold.apply(field)
        
        # Store results
        self.edge_map = field.activation.copy()
        self.edge_count = self.threshold.get_active_region_count()
        self.last_processing_time += 1
        
        return field
    
    def get_edge_map(self) -> np.ndarray:
        """Get the current edge map"""
        return self.edge_map.copy() if self.edge_map is not None else None
    
    def get_gradient_magnitude(self) -> np.ndarray:
        """Get the gradient magnitude field"""
        return self.gradient_magnitude.copy() if self.gradient_magnitude is not None else None
    
    def find_edge_peaks(self, min_strength: float = 0.5) -> list:
        """Find local maxima in edge map (strongest edges)"""
        if self.edge_map is None:
            return []
        
        return self.gradient.find_peaks(
            type('Field', (), {'activation': self.edge_map, 'shape': self.edge_map.shape})(),
            threshold=min_strength
        )
    
    def get_edge_statistics(self) -> dict:
        """Get statistics about detected edges"""
        if self.edge_map is None:
            return {}
        
        return {
            'edge_count': self.edge_count,
            'max_edge_strength': np.max(self.edge_map),
            'mean_edge_strength': np.mean(self.edge_map),
            'edge_density': np.sum(self.edge_map > 0) / self.edge_map.size,
            'processing_time': self.last_processing_time
        }
    
    def detect_edge_changes(self, previous_field: NDAnalogField) -> dict:
        """Detect changes in edge structure between frames"""
        if self.edge_map is None:
            return {'changed': False, 'change_magnitude': 0.0}
        
        # Create temporary detector for previous field
        temp_detector = EdgeDetector(
            threshold=self.threshold.threshold,
            strength=self.gradient.strength,
            hysteresis=self.threshold.hysteresis,
            mode=self.threshold.mode
        )
        temp_detector.process(previous_field)
        
        # Compare edge maps
        if temp_detector.edge_map is not None:
            change_magnitude = np.mean(np.abs(self.edge_map - temp_detector.edge_map))
            changed = change_magnitude > 0.1  # Threshold for significant change
        else:
            change_magnitude = 0.0
            changed = False
        
        return {
            'changed': changed,
            'change_magnitude': change_magnitude,
            'previous_edge_count': temp_detector.edge_count if temp_detector.edge_map is not None else 0,
            'current_edge_count': self.edge_count
        }
    
    def enhance_edges(self, field: NDAnalogField, enhancement_factor: float = 1.5) -> NDAnalogField:
        """Enhance detected edges by amplifying strong gradients"""
        if self.edge_map is None:
            return field
        
        # Create enhanced field
        enhanced = field.activation.copy()
        
        # Find strong edges
        strong_edges = self.edge_map > np.percentile(self.edge_map, 80)
        
        # Enhance around strong edges
        enhanced = np.where(strong_edges, 
                           enhanced * enhancement_factor, 
                           enhanced)
        
        field.activation = enhanced
        return field
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'edge_map': self.edge_map.copy() if self.edge_map is not None else None,
            'gradient_magnitude': self.gradient_magnitude.copy() if self.gradient_magnitude is not None else None,
            'edge_count': self.edge_count,
            'processing_time': self.last_processing_time,
            'threshold': self.threshold.threshold,
            'strength': self.gradient.strength
        }
    
    def reset(self):
        """Reset detector state"""
        self.edge_map = None
        self.gradient_magnitude = None
        self.edge_count = 0
        self.last_processing_time = 0
        self.threshold.state = None
        self.threshold.crossing_events.clear()
    
    def __repr__(self):
        return f"EdgeDetector(thresh={self.threshold.threshold:.2f}, strength={self.gradient.strength:.2f})"