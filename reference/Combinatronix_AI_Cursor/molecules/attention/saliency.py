# ============================================================================
# Saliency - Detect Salient Regions
# ============================================================================

"""
Saliency - Detect salient regions using amplification and thresholding

Composition: Amplifier + Threshold
Category: Attention
Complexity: Molecule (50-200 lines)

Detects salient regions by amplifying weak signals and applying thresholding
to identify areas of interest. This enables automatic attention capture,
saliency-based processing, and bottom-up attention mechanisms.

Example:
    >>> saliency = Saliency(amplification_gain=2.0, saliency_threshold=0.6)
    >>> salient_field = saliency.detect(field)
    >>> salient_regions = saliency.get_salient_regions()
    >>> saliency_map = saliency.get_saliency_map()
"""

import numpy as np
try:
    from ...atoms.tension_resolvers import AmplifierAtom
    from ...atoms.temporal import ThresholdAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.tension_resolvers import AmplifierAtom
    from combinatronix.atoms.temporal import ThresholdAtom
    from combinatronix.core import NDAnalogField


class Saliency:
    """Detect salient regions using amplification and thresholding"""
    
    def __init__(self, amplification_gain: float = 2.0, saliency_threshold: float = 0.6,
                 amplification_threshold: float = 0.3, mode: str = 'adaptive'):
        """
        Args:
            amplification_gain: How much to amplify weak signals
            saliency_threshold: Threshold for saliency detection
            amplification_threshold: Threshold below which to amplify
            mode: Amplification mode ('linear', 'exponential', 'adaptive')
        """
        self.amplifier = AmplifierAtom(
            threshold=amplification_threshold,
            gain=amplification_gain,
            mode=mode
        )
        self.threshold = ThresholdAtom(
            threshold=saliency_threshold,
            mode='rectify'  # Keep only salient regions
        )
        
        # Saliency state
        self.saliency_map = None
        self.salient_regions = []
        self.saliency_strength = 0.0
        self.detection_count = 0
        self.total_processed = 0
        self.amplification_history = []
    
    def detect(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Detect salient regions in field
        
        Args:
            field: Field to analyze
            **kwargs: Additional parameters
            
        Returns:
            Field with salient regions highlighted
        """
        self.total_processed += 1
        
        # Store original for comparison
        original_energy = np.sum(np.abs(field.activation))
        
        # Step 1: Amplify weak signals
        amplified_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        self.amplifier.apply(amplified_field)
        
        # Step 2: Apply threshold to get salient regions
        self.threshold.apply(amplified_field)
        
        # Step 3: Update field with saliency information
        field.activation = amplified_field.activation
        
        # Calculate saliency metrics
        self._calculate_saliency_metrics(field, original_energy)
        
        # Update saliency map
        self._update_saliency_map(field)
        
        # Find salient regions
        self._find_salient_regions(field)
        
        return field
    
    def _calculate_saliency_metrics(self, field: NDAnalogField, original_energy: float):
        """Calculate saliency strength and other metrics"""
        current_energy = np.sum(np.abs(field.activation))
        
        # Calculate saliency strength
        if original_energy > 0:
            self.saliency_strength = current_energy / original_energy
        else:
            self.saliency_strength = 0.0
        
        # Track amplification
        amplification_factor = current_energy / (original_energy + 1e-8)
        self.amplification_history.append(amplification_factor)
        
        # Keep only recent history
        if len(self.amplification_history) > 100:
            self.amplification_history = self.amplification_history[-100:]
    
    def _update_saliency_map(self, field: NDAnalogField):
        """Update saliency map based on current field state"""
        self.saliency_map = field.activation.copy()
    
    def _find_salient_regions(self, field: NDAnalogField, min_size: int = 4):
        """Find connected salient regions"""
        # Find regions above threshold
        salient_mask = field.activation > self.threshold.threshold
        
        if len(field.shape) == 2:
            # Find connected components
            regions = []
            visited = np.zeros_like(salient_mask, dtype=bool)
            
            for i in range(salient_mask.shape[0]):
                for j in range(salient_mask.shape[1]):
                    if salient_mask[i, j] and not visited[i, j]:
                        region = self._flood_fill(salient_mask, visited, i, j)
                        if len(region) >= min_size:
                            regions.append({
                                'coordinates': region,
                                'size': len(region),
                                'center': self._calculate_center(region),
                                'strength': np.mean([field.activation[coord] for coord in region])
                            })
            
            self.salient_regions = regions
        else:
            self.salient_regions = []
    
    def _flood_fill(self, mask, visited, start_i, start_j):
        """Simple flood fill for connected regions"""
        if (start_i < 0 or start_i >= mask.shape[0] or 
            start_j < 0 or start_j >= mask.shape[1] or
            visited[start_i, start_j] or not mask[start_i, start_j]):
            return []
        
        visited[start_i, start_j] = True
        region = [(start_i, start_j)]
        
        # Check 4-connected neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            region.extend(self._flood_fill(mask, visited, start_i + di, start_j + dj))
        
        return region
    
    def _calculate_center(self, region):
        """Calculate center of mass for region"""
        if not region:
            return (0, 0)
        
        y_coords = [coord[0] for coord in region]
        x_coords = [coord[1] for coord in region]
        
        center_y = int(np.mean(y_coords))
        center_x = int(np.mean(x_coords))
        
        return (center_y, center_x)
    
    def get_saliency_map(self) -> np.ndarray:
        """Get current saliency map"""
        return self.saliency_map.copy() if self.saliency_map is not None else None
    
    def get_salient_regions(self, min_strength: float = None) -> list:
        """Get list of salient regions
        
        Args:
            min_strength: Minimum strength threshold (uses detection threshold if None)
            
        Returns:
            List of salient region dictionaries
        """
        if min_strength is None:
            min_strength = self.threshold.threshold
        
        return [region for region in self.salient_regions 
                if region['strength'] >= min_strength]
    
    def get_strongest_salient_region(self) -> dict:
        """Get the strongest salient region"""
        if not self.salient_regions:
            return None
        
        return max(self.salient_regions, key=lambda r: r['strength'])
    
    def get_saliency_statistics(self) -> dict:
        """Get statistics about saliency detection"""
        if not self.salient_regions:
            return {
                'region_count': 0,
                'saliency_strength': self.saliency_strength,
                'average_amplification': 0.0,
                'detection_count': self.detection_count
            }
        
        region_sizes = [region['size'] for region in self.salient_regions]
        region_strengths = [region['strength'] for region in self.salient_regions]
        
        return {
            'region_count': len(self.salient_regions),
            'saliency_strength': self.saliency_strength,
            'average_region_size': np.mean(region_sizes),
            'average_region_strength': np.mean(region_strengths),
            'max_region_strength': np.max(region_strengths),
            'total_salient_pixels': sum(region_sizes),
            'average_amplification': np.mean(self.amplification_history) if self.amplification_history else 0.0,
            'detection_count': self.detection_count
        }
    
    def enhance_saliency(self, field: NDAnalogField, enhancement_factor: float = 1.5) -> NDAnalogField:
        """Enhance saliency by amplifying detected regions
        
        Args:
            field: Field to enhance
            enhancement_factor: How much to enhance salient regions
            
        Returns:
            Enhanced field
        """
        if self.saliency_map is None:
            return field
        
        # Find salient regions
        salient_mask = self.saliency_map > self.threshold.threshold
        
        # Enhance salient regions
        enhanced_activation = field.activation.copy()
        enhanced_activation = np.where(salient_mask,
                                     enhanced_activation * enhancement_factor,
                                     enhanced_activation)
        
        # Clip to prevent overflow
        enhanced_activation = np.clip(enhanced_activation, 0, 1)
        
        field.activation = enhanced_activation
        return field
    
    def suppress_background(self, field: NDAnalogField, suppression_factor: float = 0.3) -> NDAnalogField:
        """Suppress non-salient background regions
        
        Args:
            field: Field to process
            suppression_factor: How much to suppress background (0.0-1.0)
            
        Returns:
            Field with suppressed background
        """
        if self.saliency_map is None:
            return field
        
        # Find non-salient regions
        non_salient_mask = self.saliency_map <= self.threshold.threshold
        
        # Suppress background
        suppressed_activation = field.activation.copy()
        suppressed_activation = np.where(non_salient_mask,
                                       suppressed_activation * suppression_factor,
                                       suppressed_activation)
        
        field.activation = suppressed_activation
        return field
    
    def detect_saliency_changes(self, previous_field: NDAnalogField) -> dict:
        """Detect changes in saliency between frames
        
        Args:
            previous_field: Previous field for comparison
            
        Returns:
            Dictionary with change information
        """
        if self.saliency_map is None:
            return {'changed': False, 'change_magnitude': 0.0}
        
        # Create temporary saliency detector for previous field
        temp_saliency = Saliency(
            amplification_gain=self.amplifier.gain,
            saliency_threshold=self.threshold.threshold,
            amplification_threshold=self.amplifier.threshold,
            mode=self.amplifier.mode
        )
        temp_saliency.detect(previous_field)
        
        if temp_saliency.saliency_map is not None:
            # Compare saliency maps
            change_magnitude = np.mean(np.abs(self.saliency_map - temp_saliency.saliency_map))
            changed = change_magnitude > 0.1  # Threshold for significant change
            
            return {
                'changed': changed,
                'change_magnitude': change_magnitude,
                'previous_regions': len(temp_saliency.salient_regions),
                'current_regions': len(self.salient_regions)
            }
        else:
            return {'changed': False, 'change_magnitude': 0.0}
    
    def get_weak_signals(self) -> list:
        """Get locations of weak signals that were amplified"""
        return self.amplifier.detect_weak_signals(
            type('Field', (), {
                'activation': self.saliency_map if self.saliency_map is not None else np.zeros((1, 1)),
                'shape': self.saliency_map.shape if self.saliency_map is not None else (1, 1)
            })()
        )
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'saliency_map_shape': self.saliency_map.shape if self.saliency_map is not None else None,
            'salient_regions_count': len(self.salient_regions),
            'saliency_strength': self.saliency_strength,
            'detection_count': self.detection_count,
            'total_processed': self.total_processed,
            'amplification_gain': self.amplifier.gain,
            'saliency_threshold': self.threshold.threshold,
            'amplification_history_length': len(self.amplification_history)
        }
    
    def reset(self):
        """Reset saliency detector"""
        self.saliency_map = None
        self.salient_regions = []
        self.saliency_strength = 0.0
        self.detection_count = 0
        self.total_processed = 0
        self.amplification_history = []
        self.amplifier.amplified_regions = []
        self.threshold.state = None
        self.threshold.crossing_events = []
    
    def __repr__(self):
        return f"Saliency(gain={self.amplifier.gain:.1f}, thresh={self.threshold.threshold:.2f}, regions={len(self.salient_regions)})"