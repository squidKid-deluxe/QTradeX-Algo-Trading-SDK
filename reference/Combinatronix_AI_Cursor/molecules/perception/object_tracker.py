# ============================================================================
# ObjectTracker - Track Moving Objects
# ============================================================================

"""
ObjectTracker - Track moving objects using seed points and memory

Composition: Seed + Gradient + MemoryTrace
Category: Perception
Complexity: Molecule (50-200 lines)

Tracks objects by seeding them at detected locations, using gradient flow
to predict movement, and maintaining memory traces of object positions.
This enables continuous tracking of moving entities across frames.

Example:
    >>> tracker = ObjectTracker(max_objects=5, tracking_radius=3)
    >>> tracker.add_object("obj1", (10, 15), confidence=0.8)
    >>> tracker.update(field)
    >>> positions = tracker.get_object_positions()
"""

import numpy as np
try:
    from ...atoms.pattern_primitives import SeedAtom, GradientAtom
    from ...atoms.temporal import MemoryTraceAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.pattern_primitives import SeedAtom, GradientAtom
    from combinatronix.atoms.temporal import MemoryTraceAtom
    from combinatronix.core import NDAnalogField


class ObjectTracker:
    """Track moving objects using seed points, gradients, and memory"""
    
    def __init__(self, max_objects: int = 10, tracking_radius: int = 3,
                 memory_decay: float = 0.95, gradient_strength: float = 0.3):
        """
        Args:
            max_objects: Maximum number of objects to track
            tracking_radius: Radius for object influence
            memory_decay: How fast object memory fades
            gradient_strength: Strength of gradient-based prediction
        """
        self.seed = SeedAtom(spread_radius=tracking_radius, spread_factor=0.7)
        self.gradient = GradientAtom(strength=gradient_strength, direction='ascent')
        self.memory = MemoryTraceAtom(
            accumulation_rate=0.5,
            decay_rate=memory_decay,
            threshold=0.2
        )
        
        # Object tracking state
        self.objects = {}  # object_id -> object_data
        self.max_objects = max_objects
        self.tracking_radius = tracking_radius
        self.frame_count = 0
        self.tracking_field = None
    
    def add_object(self, object_id: str, location: tuple, 
                   confidence: float = 1.0, metadata: dict = None):
        """Add a new object to track
        
        Args:
            object_id: Unique identifier for the object
            location: Initial position (row, col)
            confidence: Tracking confidence (0.0-1.0)
            metadata: Additional object information
        """
        if len(self.objects) >= self.max_objects:
            # Remove oldest object
            oldest_id = min(self.objects.keys(), 
                          key=lambda k: self.objects[k]['first_seen'])
            del self.objects[oldest_id]
        
        self.objects[object_id] = {
            'location': location,
            'confidence': confidence,
            'velocity': (0, 0),
            'predicted_location': location,
            'first_seen': self.frame_count,
            'last_seen': self.frame_count,
            'tracking_history': [location],
            'metadata': metadata or {}
        }
    
    def update(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Update object tracking based on current field
        
        Args:
            field: Current field to analyze
            **kwargs: Additional parameters
            
        Returns:
            Field with tracking information
        """
        self.frame_count += 1
        
        # Create tracking field
        if self.tracking_field is None:
            self.tracking_field = type('Field', (), {
                'activation': np.zeros_like(field.activation),
                'shape': field.shape,
                'memory': np.zeros_like(field.activation)
            })()
        
        # Update memory trace
        self.memory.apply(field)
        
        # Update each tracked object
        for object_id, obj_data in self.objects.items():
            # Predict new location using gradient
            predicted_location = self._predict_location(obj_data, field)
            
            # Update object with new location
            old_location = obj_data['location']
            obj_data['location'] = predicted_location
            obj_data['predicted_location'] = predicted_location
            obj_data['last_seen'] = self.frame_count
            
            # Update velocity
            obj_data['velocity'] = (
                predicted_location[0] - old_location[0],
                predicted_location[1] - old_location[1]
            )
            
            # Add to tracking history
            obj_data['tracking_history'].append(predicted_location)
            
            # Keep only recent history
            if len(obj_data['tracking_history']) > 20:
                obj_data['tracking_history'] = obj_data['tracking_history'][-20:]
            
            # Seed the object in tracking field
            self.seed.apply(self.tracking_field, predicted_location, 
                          strength=obj_data['confidence'])
        
        # Apply gradient flow to tracking field
        self.gradient.apply(self.tracking_field)
        
        # Update field with tracking information
        field.activation += self.tracking_field.activation * 0.3
        
        return field
    
    def _predict_location(self, obj_data: dict, field: NDAnalogField) -> tuple:
        """Predict object location using gradient and velocity"""
        current_location = obj_data['location']
        velocity = obj_data['velocity']
        
        # Simple prediction: current + velocity
        predicted = (
            current_location[0] + velocity[0],
            current_location[1] + velocity[1]
        )
        
        # Ensure within bounds
        predicted = (
            max(0, min(predicted[0], field.shape[0] - 1)),
            max(0, min(predicted[1], field.shape[1] - 1))
        )
        
        # Use gradient to refine prediction
        if self.memory.trace is not None:
            # Find nearby high-activation regions
            search_radius = 2
            best_location = predicted
            best_activation = 0
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    test_y = predicted[0] + dy
                    test_x = predicted[1] + dx
                    
                    if (0 <= test_y < field.shape[0] and 
                        0 <= test_x < field.shape[1]):
                        activation = field.activation[test_y, test_x]
                        if activation > best_activation:
                            best_activation = activation
                            best_location = (test_y, test_x)
            
            predicted = best_location
        
        return predicted
    
    def get_object_positions(self) -> dict:
        """Get current positions of all tracked objects"""
        return {obj_id: obj_data['location'] for obj_id, obj_data in self.objects.items()}
    
    def get_object_trajectories(self) -> dict:
        """Get tracking trajectories for all objects"""
        return {obj_id: obj_data['tracking_history'] for obj_id, obj_data in self.objects.items()}
    
    def get_object_velocities(self) -> dict:
        """Get current velocities of all tracked objects"""
        return {obj_id: obj_data['velocity'] for obj_id, obj_data in self.objects.items()}
    
    def remove_object(self, object_id: str):
        """Remove an object from tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
    
    def get_object_statistics(self) -> dict:
        """Get statistics about tracked objects"""
        if not self.objects:
            return {}
        
        ages = [self.frame_count - obj['first_seen'] for obj in self.objects.values()]
        confidences = [obj['confidence'] for obj in self.objects.values()]
        
        return {
            'object_count': len(self.objects),
            'average_age': np.mean(ages),
            'average_confidence': np.mean(confidences),
            'oldest_object': max(ages) if ages else 0,
            'newest_object': min(ages) if ages else 0
        }
    
    def find_lost_objects(self, max_frames_missing: int = 5) -> list:
        """Find objects that haven't been seen recently"""
        lost = []
        for obj_id, obj_data in self.objects.items():
            frames_missing = self.frame_count - obj_data['last_seen']
            if frames_missing > max_frames_missing:
                lost.append({
                    'object_id': obj_id,
                    'frames_missing': frames_missing,
                    'last_location': obj_data['location']
                })
        return lost
    
    def cleanup_lost_objects(self, max_frames_missing: int = 10):
        """Remove objects that have been missing too long"""
        lost_objects = self.find_lost_objects(max_frames_missing)
        for lost_obj in lost_objects:
            self.remove_object(lost_obj['object_id'])
        return len(lost_objects)
    
    def get_tracking_field(self) -> np.ndarray:
        """Get the current tracking field"""
        return self.tracking_field.activation.copy() if self.tracking_field is not None else None
    
    def detect_collisions(self, collision_distance: float = 2.0) -> list:
        """Detect potential collisions between objects"""
        collisions = []
        object_ids = list(self.objects.keys())
        
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                id1, id2 = object_ids[i], object_ids[j]
                pos1 = self.objects[id1]['location']
                pos2 = self.objects[id2]['location']
                
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance <= collision_distance:
                    collisions.append({
                        'object1': id1,
                        'object2': id2,
                        'distance': distance,
                        'positions': (pos1, pos2)
                    })
        
        return collisions
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'objects': {obj_id: {
                'location': obj_data['location'],
                'confidence': obj_data['confidence'],
                'velocity': obj_data['velocity'],
                'age': self.frame_count - obj_data['first_seen']
            } for obj_id, obj_data in self.objects.items()},
            'frame_count': self.frame_count,
            'tracking_field_shape': self.tracking_field.shape if self.tracking_field is not None else None,
            'memory_strength': self.memory.get_trace_strength()
        }
    
    def reset(self):
        """Reset tracker state"""
        self.objects.clear()
        self.frame_count = 0
        self.tracking_field = None
        self.memory.clear()
    
    def __repr__(self):
        return f"ObjectTracker(objects={len(self.objects)}, max={self.max_objects})"