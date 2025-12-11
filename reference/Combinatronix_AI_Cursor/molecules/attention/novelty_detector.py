# ============================================================================
# NoveltyDetector - Detect Novel/New Information
# ============================================================================

"""
NoveltyDetector - Detect novel information by comparing to memory

Composition: Comparator + MemoryTrace
Category: Attention
Complexity: Molecule (50-200 lines)

Detects novel information by comparing current input to stored memory traces
and identifying significant differences. This enables novelty-based attention,
change detection, and adaptive learning from new information.

Example:
    >>> detector = NoveltyDetector(sensitivity=0.3, memory_decay=0.95)
    >>> detector.update_memory(field1)  # Store baseline
    >>> novelty_result = detector.detect_novelty(field2)  # Detect novelty
    >>> novel_regions = detector.get_novel_regions()
"""

import numpy as np
try:
    from ...atoms.multi_field import ComparatorAtom
    from ...atoms.temporal import MemoryTraceAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.multi_field import ComparatorAtom
    from combinatronix.atoms.temporal import MemoryTraceAtom
    from combinatronix.core import NDAnalogField


class NoveltyDetector:
    """Detect novel information by comparing to memory traces"""
    
    def __init__(self, sensitivity: float = 0.3, memory_decay: float = 0.95,
                 comparison_metric: str = 'difference', novelty_threshold: float = 0.5):
        """
        Args:
            sensitivity: Sensitivity for novelty detection (0.0-1.0)
            memory_decay: How fast memory traces decay
            comparison_metric: Metric for comparison ('difference', 'correlation', 'euclidean')
            novelty_threshold: Threshold for considering something novel
        """
        self.comparator = ComparatorAtom(
            metric=comparison_metric,
            normalize=True
        )
        self.memory_trace = MemoryTraceAtom(
            accumulation_rate=0.2,
            decay_rate=memory_decay,
            threshold=sensitivity
        )
        
        # Novelty state
        self.novelty_map = None
        self.novel_regions = []
        self.novelty_strength = 0.0
        self.baseline_established = False
        self.detection_count = 0
        self.total_processed = 0
        self.novelty_history = []
    
    def update_memory(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Update memory trace with current field
        
        Args:
            field: Field to add to memory
            **kwargs: Additional parameters
            
        Returns:
            Field with updated memory trace
        """
        self.total_processed += 1
        
        # Update memory trace
        self.memory_trace.apply(field)
        
        # Mark baseline as established
        if not self.baseline_established:
            self.baseline_established = True
        
        return field
    
    def detect_novelty(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Detect novelty by comparing to memory trace
        
        Args:
            field: Field to analyze for novelty
            **kwargs: Additional parameters
            
        Returns:
            Field with novelty information
        """
        self.total_processed += 1
        self.detection_count += 1
        
        if not self.baseline_established or self.memory_trace.trace is None:
            # No baseline yet, just update memory
            return self.update_memory(field)
        
        # Create field from memory trace for comparison
        memory_field = type('Field', (), {
            'activation': self.memory_trace.trace,
            'shape': self.memory_trace.trace.shape
        })()
        
        # Compare current field to memory
        similarity = self.comparator.compare(field, memory_field)
        novelty_score = 1.0 - similarity  # Higher novelty = lower similarity
        
        # Calculate novelty map
        self.novelty_map = self.comparator.get_difference_map(field, memory_field)
        
        # Calculate overall novelty strength
        self.novelty_strength = np.mean(self.novelty_map)
        
        # Find novel regions
        self._find_novel_regions(field, novelty_score)
        
        # Update field with novelty information
        field.activation = self.novelty_map
        
        # Track novelty history
        self.novelty_history.append({
            'time': self.detection_count,
            'novelty_strength': self.novelty_strength,
            'novelty_score': novelty_score,
            'region_count': len(self.novel_regions)
        })
        
        # Keep only recent history
        if len(self.novelty_history) > 100:
            self.novelty_history = self.novelty_history[-100:]
        
        return field
    
    def _find_novel_regions(self, field: NDAnalogField, novelty_score: float, min_size: int = 4):
        """Find connected regions of high novelty"""
        if self.novelty_map is None:
            self.novel_regions = []
            return
        
        # Find regions above novelty threshold
        novel_mask = self.novelty_map > self.novelty_threshold
        
        if len(field.shape) == 2:
            # Find connected components
            regions = []
            visited = np.zeros_like(novel_mask, dtype=bool)
            
            for i in range(novel_mask.shape[0]):
                for j in range(novel_mask.shape[1]):
                    if novel_mask[i, j] and not visited[i, j]:
                        region = self._flood_fill(novel_mask, visited, i, j)
                        if len(region) >= min_size:
                            regions.append({
                                'coordinates': region,
                                'size': len(region),
                                'center': self._calculate_center(region),
                                'novelty_strength': np.mean([self.novelty_map[coord] for coord in region]),
                                'novelty_score': novelty_score
                            })
            
            self.novel_regions = regions
        else:
            self.novel_regions = []
    
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
    
    def get_novelty_map(self) -> np.ndarray:
        """Get current novelty map"""
        return self.novelty_map.copy() if self.novelty_map is not None else None
    
    def get_novel_regions(self, min_strength: float = None) -> list:
        """Get list of novel regions
        
        Args:
            min_strength: Minimum novelty strength threshold
            
        Returns:
            List of novel region dictionaries
        """
        if min_strength is None:
            min_strength = self.novelty_threshold
        
        return [region for region in self.novel_regions 
                if region['novelty_strength'] >= min_strength]
    
    def get_strongest_novel_region(self) -> dict:
        """Get the strongest novel region"""
        if not self.novel_regions:
            return None
        
        return max(self.novel_regions, key=lambda r: r['novelty_strength'])
    
    def get_novelty_statistics(self) -> dict:
        """Get statistics about novelty detection"""
        if not self.novel_regions:
            return {
                'region_count': 0,
                'novelty_strength': self.novelty_strength,
                'baseline_established': self.baseline_established,
                'detection_count': self.detection_count,
                'memory_strength': self.memory_trace.get_trace_strength()
            }
        
        region_sizes = [region['size'] for region in self.novel_regions]
        region_strengths = [region['novelty_strength'] for region in self.novel_regions]
        
        return {
            'region_count': len(self.novel_regions),
            'novelty_strength': self.novelty_strength,
            'average_region_size': np.mean(region_sizes),
            'average_region_strength': np.mean(region_strengths),
            'max_region_strength': np.max(region_strengths),
            'total_novel_pixels': sum(region_sizes),
            'baseline_established': self.baseline_established,
            'detection_count': self.detection_count,
            'memory_strength': self.memory_trace.get_trace_strength()
        }
    
    def get_novelty_history(self, window_size: int = 10) -> list:
        """Get recent novelty detection history
        
        Args:
            window_size: Number of recent detections to return
            
        Returns:
            List of recent novelty detections
        """
        return self.novelty_history[-window_size:]
    
    def detect_novelty_events(self, event_threshold: float = 0.7) -> list:
        """Detect significant novelty events (sudden changes)
        
        Args:
            event_threshold: Threshold for considering an event significant
            
        Returns:
            List of novelty events
        """
        events = []
        
        for i, detection in enumerate(self.novelty_history):
            if detection['novelty_strength'] > event_threshold:
                events.append({
                    'time': detection['time'],
                    'novelty_strength': detection['novelty_strength'],
                    'region_count': detection['region_count'],
                    'event_type': 'high_novelty'
                })
        
        return events
    
    def adapt_sensitivity(self, adaptation_rate: float = 0.1):
        """Adapt sensitivity based on recent novelty patterns
        
        Args:
            adaptation_rate: How fast to adapt sensitivity
        """
        if len(self.novelty_history) < 5:
            return
        
        # Calculate average novelty over recent history
        recent_novelty = [d['novelty_strength'] for d in self.novelty_history[-10:]]
        avg_novelty = np.mean(recent_novelty)
        
        # Adapt sensitivity based on novelty level
        if avg_novelty > 0.7:
            # High novelty - increase sensitivity to catch more subtle changes
            self.comparator.normalize = True
            self.memory_trace.threshold *= (1 - adaptation_rate)
        elif avg_novelty < 0.3:
            # Low novelty - decrease sensitivity to reduce false positives
            self.memory_trace.threshold *= (1 + adaptation_rate)
        
        # Keep sensitivity within bounds
        self.memory_trace.threshold = np.clip(self.memory_trace.threshold, 0.01, 0.9)
    
    def reset_memory(self):
        """Reset memory trace while keeping detector state"""
        self.memory_trace.clear()
        self.baseline_established = False
        self.novelty_map = None
        self.novel_regions = []
    
    def get_memory_hotspots(self) -> list:
        """Get locations with strongest memory traces"""
        return self.memory_trace.get_hotspots()
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'novelty_map_shape': self.novelty_map.shape if self.novelty_map is not None else None,
            'novel_regions_count': len(self.novel_regions),
            'novelty_strength': self.novelty_strength,
            'baseline_established': self.baseline_established,
            'detection_count': self.detection_count,
            'total_processed': self.total_processed,
            'memory_strength': self.memory_trace.get_trace_strength(),
            'sensitivity': self.memory_trace.threshold,
            'comparison_metric': self.comparator.metric
        }
    
    def reset(self):
        """Reset novelty detector"""
        self.novelty_map = None
        self.novel_regions = []
        self.novelty_strength = 0.0
        self.baseline_established = False
        self.detection_count = 0
        self.total_processed = 0
        self.novelty_history = []
        self.memory_trace.clear()
        self.comparator.comparison_history = []
    
    def __repr__(self):
        return f"NoveltyDetector(sens={self.memory_trace.threshold:.2f}, regions={len(self.novel_regions)})"