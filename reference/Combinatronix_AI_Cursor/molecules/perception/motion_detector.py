# ============================================================================
# MotionDetector - Detect Movement Between Frames
# ============================================================================

"""
MotionDetector - Detect movement by comparing current to past

Composition: MemoryTrace + Comparator
Category: Perception
Complexity: Molecule (50-200 lines)

Detects motion by maintaining a memory trace of previous frames and comparing
current activation to the stored memory. This enables tracking of moving objects,
detection of changes, and temporal analysis of visual scenes.

Example:
    >>> detector = MotionDetector(sensitivity=0.2, memory_decay=0.9)
    >>> detector.process(frame1)  # Store first frame
    >>> motion_result = detector.process(frame2)  # Detect motion
    >>> motion_regions = detector.get_motion_regions()
"""

import numpy as np
try:
    from ...atoms.temporal import MemoryTraceAtom
    from ...atoms.multi_field import ComparatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.temporal import MemoryTraceAtom
    from combinatronix.atoms.multi_field import ComparatorAtom
    from combinatronix.core import NDAnalogField


class MotionDetector:
    """Detect motion by comparing current frame to memory trace"""
    
    def __init__(self, sensitivity: float = 0.2, memory_decay: float = 0.9,
                 accumulation_rate: float = 0.3, comparison_metric: str = 'difference'):
        """
        Args:
            sensitivity: Motion detection threshold (0.0-1.0)
            memory_decay: How fast memory fades (0.0-1.0)
            accumulation_rate: How fast to accumulate new memories
            comparison_metric: Metric for comparing frames ('difference', 'correlation')
        """
        self.memory = MemoryTraceAtom(
            accumulation_rate=accumulation_rate,
            decay_rate=memory_decay,
            threshold=sensitivity
        )
        self.comparator = ComparatorAtom(
            metric=comparison_metric,
            normalize=True
        )
        
        # State tracking
        self.motion_map = None
        self.motion_strength = 0.0
        self.motion_regions = []
        self.frame_count = 0
        self.has_previous_frame = False
    
    def process(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Detect motion by comparing to previous frame
        
        Args:
            field: Current field to analyze
            **kwargs: Additional parameters
            
        Returns:
            Field with motion information in activation layer
        """
        self.frame_count += 1
        
        if not self.has_previous_frame:
            # First frame - just store in memory
            self.memory.apply(field)
            self.has_previous_frame = True
            self.motion_map = np.zeros_like(field.activation)
            field.activation = self.motion_map
            return field
        
        # Create a copy for comparison
        current_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        # Compare current to memory trace
        if self.memory.trace is not None:
            # Create field from memory trace
            memory_field = type('Field', (), {
                'activation': self.memory.trace,
                'shape': self.memory.trace.shape
            })()
            
            # Compute motion as difference
            self.motion_map = self.comparator.get_difference_map(current_field, memory_field)
            
            # Compute overall motion strength
            self.motion_strength = np.mean(self.motion_map)
            
            # Find motion regions above threshold
            motion_mask = self.motion_map > self.memory.threshold
            self.motion_regions = np.argwhere(motion_mask).tolist()
            
            # Update field activation with motion map
            field.activation = self.motion_map
            
        else:
            # No memory yet
            self.motion_map = np.zeros_like(field.activation)
            field.activation = self.motion_map
            self.motion_strength = 0.0
            self.motion_regions = []
        
        # Update memory with current frame
        self.memory.apply(field)
        
        return field
    
    def get_motion_map(self) -> np.ndarray:
        """Get the current motion map"""
        return self.motion_map.copy() if self.motion_map is not None else None
    
    def get_motion_regions(self, min_size: int = 1) -> list:
        """Get coordinates of motion regions
        
        Args:
            min_size: Minimum size of motion region to include
            
        Returns:
            List of (row, col) coordinates with motion
        """
        if self.motion_map is None:
            return []
        
        # Filter by minimum size (simple implementation)
        if min_size > 1:
            # Find connected components (simplified)
            motion_mask = self.motion_map > self.memory.threshold
            regions = []
            visited = np.zeros_like(motion_mask, dtype=bool)
            
            for i in range(motion_mask.shape[0]):
                for j in range(motion_mask.shape[1]):
                    if motion_mask[i, j] and not visited[i, j]:
                        region = self._flood_fill(motion_mask, visited, i, j)
                        if len(region) >= min_size:
                            regions.extend(region)
            
            return regions
        else:
            return self.motion_regions
    
    def _flood_fill(self, mask, visited, start_i, start_j):
        """Simple flood fill to find connected regions"""
        region = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1] and 
                mask[i, j] and not visited[i, j]):
                visited[i, j] = True
                region.append((i, j))
                
                # Add neighbors
                stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
        
        return region
    
    def get_motion_statistics(self) -> dict:
        """Get statistics about detected motion"""
        if self.motion_map is None:
            return {}
        
        return {
            'motion_strength': self.motion_strength,
            'motion_regions_count': len(self.motion_regions),
            'max_motion': np.max(self.motion_map),
            'mean_motion': np.mean(self.motion_map),
            'motion_density': np.sum(self.motion_map > self.memory.threshold) / self.motion_map.size,
            'frame_count': self.frame_count,
            'memory_strength': self.memory.get_trace_strength()
        }
    
    def detect_motion_events(self, event_threshold: float = 0.5) -> list:
        """Detect significant motion events (sudden changes)"""
        if self.motion_map is None:
            return []
        
        events = []
        motion_mask = self.motion_map > event_threshold
        
        # Find local maxima in motion map
        for i in range(1, motion_map.shape[0] - 1):
            for j in range(1, motion_map.shape[1] - 1):
                if motion_mask[i, j]:
                    # Check if it's a local maximum
                    center = self.motion_map[i, j]
                    neighbors = [
                        self.motion_map[i-1, j], self.motion_map[i+1, j],
                        self.motion_map[i, j-1], self.motion_map[i, j+1]
                    ]
                    if all(center >= n for n in neighbors):
                        events.append({
                            'location': (i, j),
                            'strength': center,
                            'frame': self.frame_count
                        })
        
        return events
    
    def track_motion_flow(self, flow_threshold: float = 0.3) -> list:
        """Track motion flow vectors between frames"""
        if not hasattr(self, '_previous_motion_map') or self._previous_motion_map is None:
            self._previous_motion_map = self.motion_map.copy() if self.motion_map is not None else None
            return []
        
        if self.motion_map is None:
            return []
        
        # Simple optical flow estimation (very basic)
        flow_vectors = []
        
        # Find motion in current frame
        current_motion = self.motion_map > flow_threshold
        prev_motion = self._previous_motion_map > flow_threshold
        
        # Find motion regions that moved
        for i in range(1, current_motion.shape[0] - 1):
            for j in range(1, current_motion.shape[1] - 1):
                if current_motion[i, j]:
                    # Look for corresponding motion in previous frame
                    best_match = None
                    best_distance = float('inf')
                    
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if (0 <= ni < prev_motion.shape[0] and 
                                0 <= nj < prev_motion.shape[1] and 
                                prev_motion[ni, nj]):
                                distance = np.sqrt(di*di + dj*dj)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_match = (di, dj)
                    
                    if best_match is not None:
                        flow_vectors.append({
                            'from': (i, j),
                            'to': (i + best_match[0], j + best_match[1]),
                            'vector': best_match,
                            'strength': self.motion_map[i, j]
                        })
        
        # Update for next frame
        self._previous_motion_map = self.motion_map.copy()
        
        return flow_vectors
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'motion_map': self.motion_map.copy() if self.motion_map is not None else None,
            'motion_strength': self.motion_strength,
            'motion_regions': self.motion_regions.copy(),
            'frame_count': self.frame_count,
            'has_previous_frame': self.has_previous_frame,
            'memory_trace_strength': self.memory.get_trace_strength(),
            'sensitivity': self.memory.threshold
        }
    
    def reset(self):
        """Reset detector state"""
        self.motion_map = None
        self.motion_strength = 0.0
        self.motion_regions = []
        self.frame_count = 0
        self.has_previous_frame = False
        self.memory.clear()
        self.comparator.comparison_history.clear()
        self._previous_motion_map = None
    
    def __repr__(self):
        return f"MotionDetector(sens={self.memory.threshold:.2f}, decay={self.memory.decay_rate:.2f})"