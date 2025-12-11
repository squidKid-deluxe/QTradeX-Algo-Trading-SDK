# ============================================================================
# WorkingMemory - Active Memory with Focus and Control
# ============================================================================

"""
WorkingMemory - Active memory with focus and control mechanisms

Composition: Echo + Attractor + Damper
Category: Memory
Complexity: Molecule (50-200 lines)

Maintains active working memory with focus control, attention management,
and capacity limits. This enables active manipulation of information,
focused processing, and controlled memory management for complex tasks.

Example:
    >>> wm = WorkingMemory(capacity=7, focus_strength=0.3, damping_threshold=0.8)
    >>> wm.add_item(field, importance=0.9, focus_location=(5, 5))
    >>> wm.update_focus((6, 6))
    >>> active_items = wm.get_active_items()
"""

import numpy as np
from collections import deque
try:
    from ...atoms.pattern_primitives import EchoAtom
    from ...atoms.field_dynamics import AttractorAtom
    from ...atoms.tension_resolvers import DamperAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.pattern_primitives import EchoAtom
    from combinatronix.atoms.field_dynamics import AttractorAtom
    from combinatronix.atoms.tension_resolvers import DamperAtom
    from combinatronix.core import NDAnalogField


class WorkingMemory:
    """Working memory with focus control and capacity management"""
    
    def __init__(self, capacity: int = 7, focus_strength: float = 0.3,
                 damping_threshold: float = 0.8, echo_decay: float = 0.9):
        """
        Args:
            capacity: Maximum number of items in working memory
            focus_strength: Strength of focus attraction
            damping_threshold: Threshold for damping overflow
            echo_decay: Decay rate for echo effects
        """
        self.echo = EchoAtom(decay_rate=echo_decay, depth=capacity)
        self.attractor = AttractorAtom(strength=focus_strength)
        self.damper = DamperAtom(
            threshold=damping_threshold,
            damping_rate=0.4,
            mode='soft'
        )
        
        # Working memory state
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
        self.focus_location = None
        self.focus_radius = 2.0
        self.access_count = 0
        self.total_added = 0
        self.total_removed = 0
    
    def add_item(self, field: NDAnalogField, importance: float = 0.5,
                 focus_location: tuple = None, metadata: dict = None) -> str:
        """Add item to working memory
        
        Args:
            field: Field to add
            importance: Importance level (0.0-1.0)
            focus_location: Location to focus on
            metadata: Additional metadata
            
        Returns:
            Item ID
        """
        self.access_count += 1
        self.total_added += 1
        
        # Generate item ID
        item_id = f"wm_{self.access_count}"
        
        # Create item data
        item_data = {
            'id': item_id,
            'activation': field.activation.copy(),
            'shape': field.shape,
            'importance': importance,
            'focus_location': focus_location,
            'metadata': metadata or {},
            'creation_time': self.access_count,
            'access_count': 0,
            'last_accessed': self.access_count,
            'focus_strength': 0.0
        }
        
        # Add to working memory
        self.items.append(item_data)
        
        # Update focus if specified
        if focus_location:
            self.update_focus(focus_location)
        
        return item_id
    
    def update_focus(self, focus_location: tuple, radius: float = None):
        """Update focus location and radius
        
        Args:
            focus_location: New focus location
            radius: Focus radius (uses default if None)
        """
        self.focus_location = focus_location
        if radius is not None:
            self.focus_radius = radius
        
        # Update attractor
        self.attractor.set_location(focus_location)
        
        # Update focus strength for items
        self._update_item_focus_strengths()
    
    def _update_item_focus_strengths(self):
        """Update focus strength for all items based on current focus"""
        if self.focus_location is None:
            return
        
        for item in self.items:
            if item['focus_location'] is not None:
                # Calculate distance from current focus
                if len(item['focus_location']) == 2 and len(self.focus_location) == 2:
                    distance = np.sqrt(
                        (item['focus_location'][0] - self.focus_location[0])**2 +
                        (item['focus_location'][1] - self.focus_location[1])**2
                    )
                    
                    # Focus strength inversely proportional to distance
                    item['focus_strength'] = max(0, 1.0 - distance / self.focus_radius)
                else:
                    item['focus_strength'] = 0.5  # Default strength
            else:
                item['focus_strength'] = 0.0
    
    def process(self, field: NDAnalogField) -> NDAnalogField:
        """Process field with working memory effects
        
        Args:
            field: Field to process
            
        Returns:
            Processed field
        """
        # Apply echo for memory trace
        self.echo.apply(field)
        
        # Apply focus attraction if focus is set
        if self.focus_location is not None:
            self.attractor.apply(field)
        
        # Apply damping to prevent overflow
        self.damper.apply(field)
        
        return field
    
    def get_active_items(self, min_importance: float = 0.3) -> list:
        """Get currently active items in working memory
        
        Args:
            min_importance: Minimum importance threshold
            
        Returns:
            List of active items
        """
        active_items = []
        
        for item in self.items:
            if item['importance'] >= min_importance:
                # Calculate activity based on importance and focus
                activity = (item['importance'] * 0.7 + 
                           item['focus_strength'] * 0.3)
                
                active_items.append({
                    'item': item,
                    'activity': activity,
                    'age': self.access_count - item['creation_time']
                })
        
        # Sort by activity
        active_items.sort(key=lambda x: x['activity'], reverse=True)
        return active_items
    
    def retrieve_item(self, item_id: str) -> NDAnalogField:
        """Retrieve specific item from working memory
        
        Args:
            item_id: ID of item to retrieve
            
        Returns:
            Field with retrieved item, or None if not found
        """
        for item in self.items:
            if item['id'] == item_id:
                item['access_count'] += 1
                item['last_accessed'] = self.access_count
                
                # Create field with item data
                field = NDAnalogField(item['shape'])
                field.activation = item['activation'].copy()
                
                # Apply working memory processing
                self.process(field)
                
                return field
        
        return None
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from working memory
        
        Args:
            item_id: ID of item to remove
            
        Returns:
            True if item was removed
        """
        for i, item in enumerate(self.items):
            if item['id'] == item_id:
                del self.items[i]
                self.total_removed += 1
                return True
        
        return False
    
    def clear_old_items(self, age_threshold: int = 10) -> int:
        """Remove old items from working memory
        
        Args:
            age_threshold: Maximum age to keep
            
        Returns:
            Number of items removed
        """
        current_time = self.access_count
        items_to_remove = []
        
        for i, item in enumerate(self.items):
            age = current_time - item['creation_time']
            if age > age_threshold:
                items_to_remove.append(i)
        
        # Remove items (in reverse order to maintain indices)
        removed_count = 0
        for i in reversed(items_to_remove):
            del self.items[i]
            removed_count += 1
            self.total_removed += 1
        
        return removed_count
    
    def get_memory_load(self) -> dict:
        """Get current memory load statistics"""
        if not self.items:
            return {
                'item_count': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'average_importance': 0.0,
                'focus_strength': 0.0
            }
        
        importances = [item['importance'] for item in self.items]
        focus_strengths = [item['focus_strength'] for item in self.items]
        
        return {
            'item_count': len(self.items),
            'capacity': self.capacity,
            'utilization': len(self.items) / self.capacity,
            'average_importance': np.mean(importances),
            'average_focus_strength': np.mean(focus_strengths),
            'total_added': self.total_added,
            'total_removed': self.total_removed
        }
    
    def get_focus_statistics(self) -> dict:
        """Get statistics about current focus"""
        if self.focus_location is None:
            return {
                'focus_active': False,
                'focus_location': None,
                'focus_radius': self.focus_radius,
                'focused_items': 0
            }
        
        # Count items within focus radius
        focused_items = 0
        for item in self.items:
            if item['focus_location'] is not None:
                if len(item['focus_location']) == 2 and len(self.focus_location) == 2:
                    distance = np.sqrt(
                        (item['focus_location'][0] - self.focus_location[0])**2 +
                        (item['focus_location'][1] - self.focus_location[1])**2
                    )
                    if distance <= self.focus_radius:
                        focused_items += 1
        
        return {
            'focus_active': True,
            'focus_location': self.focus_location,
            'focus_radius': self.focus_radius,
            'focused_items': focused_items,
            'attraction_strength': self.attractor.strength
        }
    
    def consolidate_important_items(self, importance_threshold: float = 0.8) -> int:
        """Strengthen important items in working memory
        
        Args:
            importance_threshold: Minimum importance for consolidation
            
        Returns:
            Number of items consolidated
        """
        consolidated_count = 0
        
        for item in self.items:
            if item['importance'] >= importance_threshold:
                # Strengthen the item
                item['activation'] *= 1.2
                item['activation'] = np.clip(item['activation'], 0, 1)
                
                # Increase importance slightly
                item['importance'] = min(item['importance'] * 1.1, 1.0)
                
                consolidated_count += 1
        
        return consolidated_count
    
    def get_working_field(self) -> NDAnalogField:
        """Get a field representing current working memory state"""
        if not self.items:
            return None
        
        # Create field with combined activation from all items
        # Use the shape of the most recent item
        latest_item = self.items[-1]
        field = NDAnalogField(latest_item['shape'])
        
        # Combine activations from all items
        for item in self.items:
            if item['activation'].shape == field.shape:
                # Weight by importance and focus
                weight = item['importance'] * 0.7 + item['focus_strength'] * 0.3
                field.activation += item['activation'] * weight
        
        # Normalize
        max_activation = np.max(field.activation)
        if max_activation > 0:
            field.activation /= max_activation
        
        # Apply working memory processing
        self.process(field)
        
        return field
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'items': [{
                'id': item['id'],
                'importance': item['importance'],
                'focus_strength': item['focus_strength'],
                'access_count': item['access_count'],
                'age': self.access_count - item['creation_time']
            } for item in self.items],
            'focus_location': self.focus_location,
            'focus_radius': self.focus_radius,
            'capacity': self.capacity,
            'access_count': self.access_count,
            'echo_strength': self.echo.get_echo_strength(),
            'damped_overflow': self.damper.total_damped
        }
    
    def reset(self):
        """Reset working memory"""
        self.items.clear()
        self.focus_location = None
        self.access_count = 0
        self.total_added = 0
        self.total_removed = 0
        self.echo.clear()
        self.damper.total_damped = 0.0
    
    def __repr__(self):
        return f"WorkingMemory(items={len(self.items)}, capacity={self.capacity}, focus={self.focus_location})"