# ============================================================================
# Focus - Focus Attention on Specific Region
# ============================================================================

"""
Focus - Focus attention on specific region

Composition: Attractor + Damper
Category: Attention
Complexity: Molecule (50-200 lines)

Focuses attention on specific regions by attracting activation toward a target
location while damping background activity. This enables selective attention,
concentration, and directed processing of specific areas of interest.

Example:
    >>> focus = Focus(location=(5, 5), strength=0.3, damping_threshold=0.8)
    >>> focused_field = focus.apply(field)
    >>> focus.update_location((6, 6))
    >>> attention_map = focus.get_attention_map()
"""

import numpy as np
try:
    from ...atoms.field_dynamics import AttractorAtom
    from ...atoms.tension_resolvers import DamperAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.field_dynamics import AttractorAtom
    from combinatronix.atoms.tension_resolvers import DamperAtom
    from combinatronix.core import NDAnalogField


class Focus:
    """Focus attention on specific regions with background damping"""
    
    def __init__(self, location: tuple = None, strength: float = 0.3,
                 damping_threshold: float = 0.8, damping_rate: float = 0.4,
                 focus_radius: float = 2.0):
        """
        Args:
            location: Focus location (None = center)
            strength: Attraction strength
            damping_threshold: Threshold for damping background
            damping_rate: How much to dampen background
            focus_radius: Radius of focus area
        """
        self.attractor = AttractorAtom(
            location=location,
            strength=strength,
            radius=focus_radius
        )
        self.damper = DamperAtom(
            threshold=damping_threshold,
            damping_rate=damping_rate,
            mode='soft'
        )
        
        # Focus state
        self.focus_location = location
        self.focus_radius = focus_radius
        self.attention_map = None
        self.focus_strength = 0.0
        self.background_damping = 0.0
        self.focus_count = 0
        self.total_processed = 0
    
    def apply(self, field: NDAnalogField, focus_location: tuple = None) -> NDAnalogField:
        """Apply focus attention to field
        
        Args:
            field: Field to focus on
            focus_location: Override focus location (optional)
            
        Returns:
            Field with focused attention
        """
        self.total_processed += 1
        
        # Update focus location if provided
        if focus_location is not None:
            self.update_location(focus_location)
        
        # Store original for comparison
        original_energy = np.sum(np.abs(field.activation))
        
        # Apply attraction to focus area
        self.attractor.apply(field)
        
        # Apply spatial damping to background
        if self.focus_location is not None:
            self.damper.apply_spatial_damping(field, self.focus_location, self.focus_radius)
        else:
            self.damper.apply(field)
        
        # Calculate focus metrics
        self._calculate_focus_metrics(field, original_energy)
        
        # Update attention map
        self._update_attention_map(field)
        
        return field
    
    def update_location(self, location: tuple, radius: float = None):
        """Update focus location and radius
        
        Args:
            location: New focus location
            radius: New focus radius (uses current if None)
        """
        self.focus_location = location
        if radius is not None:
            self.focus_radius = radius
        
        self.attractor.set_location(location)
        self.focus_count += 1
    
    def _calculate_focus_metrics(self, field: NDAnalogField, original_energy: float):
        """Calculate focus strength and background damping metrics"""
        if self.focus_location is None:
            self.focus_strength = 0.0
            self.background_damping = 0.0
            return
        
        # Calculate focus strength (energy in focus area)
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            dy = y_indices - self.focus_location[0]
            dx = x_indices - self.focus_location[1]
            distance = np.sqrt(dy**2 + dx**2)
            
            focus_mask = distance <= self.focus_radius
            focus_energy = np.sum(np.abs(field.activation[focus_mask]))
            total_energy = np.sum(np.abs(field.activation))
            
            self.focus_strength = focus_energy / (total_energy + 1e-8)
            
            # Calculate background damping
            background_mask = distance > self.focus_radius
            background_energy = np.sum(np.abs(field.activation[background_mask]))
            self.background_damping = 1.0 - (background_energy / (total_energy + 1e-8))
    
    def _update_attention_map(self, field: NDAnalogField):
        """Update attention map based on current field state"""
        if self.focus_location is None:
            self.attention_map = np.zeros_like(field.activation)
            return
        
        # Create attention map based on distance from focus
        if len(field.shape) == 2:
            y_indices, x_indices = np.indices(field.shape)
            dy = y_indices - self.focus_location[0]
            dx = x_indices - self.focus_location[1]
            distance = np.sqrt(dy**2 + dx**2)
            
            # Attention strength inversely proportional to distance
            attention_strength = np.where(
                distance <= self.focus_radius,
                1.0 - (distance / self.focus_radius),
                0.0
            )
            
            self.attention_map = attention_strength
        else:
            self.attention_map = np.zeros_like(field.activation)
    
    def get_attention_map(self) -> np.ndarray:
        """Get current attention map"""
        return self.attention_map.copy() if self.attention_map is not None else None
    
    def get_focus_region(self) -> list:
        """Get coordinates within focus region"""
        if self.focus_location is None or self.attention_map is None:
            return []
        
        focus_coords = np.argwhere(self.attention_map > 0.1)
        return [tuple(coord) for coord in focus_coords]
    
    def get_focus_statistics(self) -> dict:
        """Get statistics about current focus"""
        return {
            'focus_location': self.focus_location,
            'focus_radius': self.focus_radius,
            'focus_strength': self.focus_strength,
            'background_damping': self.background_damping,
            'focus_count': self.focus_count,
            'total_processed': self.total_processed,
            'attraction_strength': self.attractor.strength,
            'damping_threshold': self.damper.threshold
        }
    
    def shift_focus(self, new_location: tuple, transition_steps: int = 5) -> NDAnalogField:
        """Gradually shift focus to new location
        
        Args:
            new_location: Target focus location
            transition_steps: Number of steps for transition
            
        Returns:
            Field with shifted focus
        """
        if self.focus_location is None:
            self.update_location(new_location)
            return None
        
        # Calculate transition path
        start_y, start_x = self.focus_location
        end_y, end_x = new_location
        
        step_y = (end_y - start_y) / transition_steps
        step_x = (end_x - start_x) / transition_steps
        
        # Create transition field
        transition_field = None
        
        for step in range(transition_steps + 1):
            intermediate_location = (
                int(start_y + step * step_y),
                int(start_x + step * step_x)
            )
            
            # Update focus location
            self.update_location(intermediate_location)
            
            # Apply focus (would need field input for full implementation)
            # This is a simplified version
            if step == transition_steps:
                # Final step
                break
        
        return transition_field
    
    def expand_focus(self, new_radius: float, transition_steps: int = 3):
        """Gradually expand focus radius
        
        Args:
            new_radius: Target focus radius
            transition_steps: Number of steps for expansion
        """
        if self.focus_radius is None:
            self.focus_radius = new_radius
            return
        
        step_size = (new_radius - self.focus_radius) / transition_steps
        
        for step in range(transition_steps + 1):
            intermediate_radius = self.focus_radius + step * step_size
            self.focus_radius = intermediate_radius
            self.attractor.radius = intermediate_radius
    
    def contract_focus(self, new_radius: float, transition_steps: int = 3):
        """Gradually contract focus radius
        
        Args:
            new_radius: Target focus radius
            transition_steps: Number of steps for contraction
        """
        if self.focus_radius is None:
            self.focus_radius = new_radius
            return
        
        step_size = (self.focus_radius - new_radius) / transition_steps
        
        for step in range(transition_steps + 1):
            intermediate_radius = self.focus_radius - step * step_size
            self.focus_radius = max(intermediate_radius, 0.1)  # Minimum radius
            self.attractor.radius = self.focus_radius
    
    def get_attraction_field(self, field: NDAnalogField) -> np.ndarray:
        """Get attraction strength at each point"""
        return self.attractor.get_attraction_field(field)
    
    def get_damping_regions(self, field: NDAnalogField) -> list:
        """Get regions where damping is applied"""
        return self.damper.get_overflow_regions(field)
    
    def is_focused_on(self, location: tuple, tolerance: float = 1.0) -> bool:
        """Check if focus is on specific location
        
        Args:
            location: Location to check
            tolerance: Distance tolerance
            
        Returns:
            True if location is within focus
        """
        if self.focus_location is None:
            return False
        
        if len(location) == 2 and len(self.focus_location) == 2:
            distance = np.sqrt(
                (location[0] - self.focus_location[0])**2 +
                (location[1] - self.focus_location[1])**2
            )
            return distance <= (self.focus_radius + tolerance)
        
        return False
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'focus_location': self.focus_location,
            'focus_radius': self.focus_radius,
            'focus_strength': self.focus_strength,
            'background_damping': self.background_damping,
            'focus_count': self.focus_count,
            'total_processed': self.total_processed,
            'attraction_strength': self.attractor.strength,
            'damping_threshold': self.damper.threshold,
            'attention_map_shape': self.attention_map.shape if self.attention_map is not None else None
        }
    
    def reset(self):
        """Reset focus state"""
        self.focus_location = None
        self.focus_radius = 2.0
        self.attention_map = None
        self.focus_strength = 0.0
        self.background_damping = 0.0
        self.focus_count = 0
        self.total_processed = 0
        self.attractor.location = None
        self.damper.total_damped = 0.0
    
    def __repr__(self):
        return f"Focus(loc={self.focus_location}, radius={self.focus_radius:.1f}, strength={self.focus_strength:.2f})"