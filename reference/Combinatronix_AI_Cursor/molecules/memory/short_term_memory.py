# ============================================================================
# ShortTermMemory - Hold Recent Information with Natural Forgetting
# ============================================================================

"""
ShortTermMemory - Hold recent information with natural forgetting

Composition: Echo + Decay
Category: Memory
Complexity: Molecule (50-200 lines)

Maintains a short-term memory buffer that holds recent information with
natural decay. This enables temporary storage of current context, working
with recent patterns, and gradual forgetting of outdated information.

Example:
    >>> stm = ShortTermMemory(capacity=5, decay_rate=0.9)
    >>> stm.store(field)
    >>> retrieved = stm.recall()
    >>> recent_items = stm.get_recent_items(3)
"""

import numpy as np
from collections import deque
try:
    from ...atoms.pattern_primitives import EchoAtom
    from ...atoms.tension_resolvers import DamperAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.pattern_primitives import EchoAtom
    from combinatronix.atoms.tension_resolvers import DamperAtom
    from combinatronix.core import NDAnalogField


class ShortTermMemory:
    """Short-term memory with natural decay and capacity limits"""
    
    def __init__(self, capacity: int = 5, decay_rate: float = 0.9, 
                 damping_threshold: float = 0.8, damping_rate: float = 0.3):
        """
        Args:
            capacity: Maximum number of items to store
            decay_rate: How fast memories decay (0.0-1.0)
            damping_threshold: Activation threshold for damping
            damping_rate: How much to dampen overflow
        """
        self.echo = EchoAtom(decay_rate=decay_rate, depth=capacity)
        self.damper = DamperAtom(
            threshold=damping_threshold,
            damping_rate=damping_rate,
            mode='soft'
        )
        
        # Memory state
        self.capacity = capacity
        self.memory_items = deque(maxlen=capacity)
        self.access_times = deque(maxlen=capacity)
        self.access_count = 0
        self.total_stored = 0
        self.total_retrieved = 0
    
    def store(self, field: NDAnalogField, metadata: dict = None) -> NDAnalogField:
        """Store information in short-term memory
        
        Args:
            field: Field to store
            metadata: Additional information about the item
            
        Returns:
            Field with memory information
        """
        self.access_count += 1
        self.total_stored += 1
        
        # Store the field data
        memory_item = {
            'activation': field.activation.copy(),
            'shape': field.shape,
            'timestamp': self.access_count,
            'metadata': metadata or {},
            'access_count': 0
        }
        
        self.memory_items.append(memory_item)
        self.access_times.append(self.access_count)
        
        # Apply echo to create memory trace
        field_copy = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        self.echo.apply(field_copy)
        
        # Apply damping to prevent overflow
        self.damper.apply(field_copy)
        
        # Update field with memory trace
        field.activation = field_copy.activation
        
        return field
    
    def recall(self, field: NDAnalogField = None, item_index: int = -1) -> NDAnalogField:
        """Recall information from short-term memory
        
        Args:
            field: Field to populate with recalled data (if None, creates new)
            item_index: Which item to recall (-1 for most recent)
            
        Returns:
            Field with recalled information
        """
        if not self.memory_items:
            if field is None:
                return None
            return field
        
        # Get the requested item
        if item_index < 0:
            item_index = len(self.memory_items) + item_index
        
        if 0 <= item_index < len(self.memory_items):
            item = self.memory_items[item_index]
            item['access_count'] += 1
            self.total_retrieved += 1
            
            if field is None:
                # Create new field
                field = NDAnalogField(item['shape'])
            
            # Restore the activation
            field.activation = item['activation'].copy()
            
            # Apply echo for memory trace
            self.echo.apply(field)
            
            # Apply damping
            self.damper.apply(field)
        
        return field
    
    def get_recent_items(self, count: int = 3) -> list:
        """Get the most recent memory items
        
        Args:
            count: Number of recent items to return
            
        Returns:
            List of recent memory items
        """
        if not self.memory_items:
            return []
        
        recent = list(self.memory_items)[-count:]
        return recent
    
    def get_memory_statistics(self) -> dict:
        """Get statistics about memory usage"""
        if not self.memory_items:
            return {
                'item_count': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'total_stored': self.total_stored,
                'total_retrieved': self.total_retrieved,
                'average_access_count': 0.0,
                'echo_strength': 0.0
            }
        
        access_counts = [item['access_count'] for item in self.memory_items]
        
        return {
            'item_count': len(self.memory_items),
            'capacity': self.capacity,
            'utilization': len(self.memory_items) / self.capacity,
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'average_access_count': np.mean(access_counts),
            'echo_strength': self.echo.get_echo_strength(),
            'damped_overflow': self.damper.total_damped
        }
    
    def find_similar_items(self, pattern: np.ndarray, 
                          similarity_threshold: float = 0.7) -> list:
        """Find memory items similar to given pattern
        
        Args:
            pattern: Pattern to match against
            similarity_threshold: Minimum similarity for match
            
        Returns:
            List of similar items with similarity scores
        """
        similar_items = []
        
        for i, item in enumerate(self.memory_items):
            if item['activation'].shape == pattern.shape:
                similarity = self._compute_similarity(item['activation'], pattern)
                
                if similarity >= similarity_threshold:
                    similar_items.append({
                        'index': i,
                        'item': item,
                        'similarity': similarity,
                        'age': self.access_count - item['timestamp']
                    })
        
        # Sort by similarity
        similar_items.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_items
    
    def _compute_similarity(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> float:
        """Compute similarity between two patterns"""
        # Normalize patterns
        norm_a = pattern_a / (np.linalg.norm(pattern_a) + 1e-8)
        norm_b = pattern_b / (np.linalg.norm(pattern_b) + 1e-8)
        
        # Compute correlation
        correlation = np.dot(norm_a.flatten(), norm_b.flatten())
        return correlation
    
    def consolidate_memory(self, consolidation_factor: float = 1.2):
        """Strengthen frequently accessed memories (consolidation)"""
        for item in self.memory_items:
            if item['access_count'] > 1:
                # Strengthen frequently accessed items
                item['activation'] *= consolidation_factor
                item['activation'] = np.clip(item['activation'], 0, 1)
    
    def forget_old_items(self, age_threshold: int = 10):
        """Remove old items from memory"""
        current_time = self.access_count
        items_to_remove = []
        
        for i, item in enumerate(self.memory_items):
            age = current_time - item['timestamp']
            if age > age_threshold:
                items_to_remove.append(i)
        
        # Remove items (in reverse order to maintain indices)
        for i in reversed(items_to_remove):
            del self.memory_items[i]
            del self.access_times[i]
        
        return len(items_to_remove)
    
    def get_memory_trace(self) -> np.ndarray:
        """Get the current memory trace (echo field)"""
        if not self.memory_items:
            return None
        
        # Create a field to get echo trace
        field = NDAnalogField(self.memory_items[0]['shape'])
        self.echo.apply(field)
        return field.activation
    
    def clear_memory(self):
        """Clear all memory items"""
        self.memory_items.clear()
        self.access_times.clear()
        self.echo.clear()
        self.damper.total_damped = 0.0
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'memory_items': [{
                'shape': item['shape'],
                'timestamp': item['timestamp'],
                'access_count': item['access_count'],
                'metadata': item['metadata']
            } for item in self.memory_items],
            'access_times': list(self.access_times),
            'access_count': self.access_count,
            'capacity': self.capacity,
            'echo_strength': self.echo.get_echo_strength(),
            'damped_overflow': self.damper.total_damped
        }
    
    def reset(self):
        """Reset memory state"""
        self.clear_memory()
        self.access_count = 0
        self.total_stored = 0
        self.total_retrieved = 0
    
    def __repr__(self):
        return f"ShortTermMemory(capacity={self.capacity}, items={len(self.memory_items)})"