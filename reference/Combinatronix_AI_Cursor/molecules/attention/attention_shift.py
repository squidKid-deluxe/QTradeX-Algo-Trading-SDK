# ============================================================================
# AttentionShift - Shift Attention Using Gradient and Vortex
# ============================================================================

"""
AttentionShift - Shift attention using gradient flow and vortex dynamics

Composition: Gradient + Vortex
Category: Attention
Complexity: Molecule (50-200 lines)

Shifts attention by using gradient flow to identify attention-worthy regions
and vortex dynamics to create smooth transitions between focus areas. This
enables dynamic attention shifting, smooth focus transitions, and adaptive
attention redirection.

Example:
    >>> shift = AttentionShift(gradient_strength=0.4, vortex_strength=0.2)
    >>> shifted_field = shift.shift_attention(field, target_location=(6, 6))
    >>> flow_field = shift.get_flow_field()
    >>> transition_path = shift.get_transition_path()
"""

import numpy as np
try:
    from ...atoms.pattern_primitives import GradientAtom
    from ...atoms.field_dynamics import VortexAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.pattern_primitives import GradientAtom
    from combinatronix.atoms.field_dynamics import VortexAtom
    from combinatronix.core import NDAnalogField


class AttentionShift:
    """Shift attention using gradient flow and vortex dynamics"""
    
    def __init__(self, gradient_strength: float = 0.4, vortex_strength: float = 0.2,
                 angular_velocity: float = 0.5, transition_smoothness: float = 0.3):
        """
        Args:
            gradient_strength: Strength of gradient-based attention flow
            vortex_strength: Strength of vortex dynamics
            angular_velocity: Speed of vortex rotation
            transition_smoothness: How smooth transitions should be
        """
        self.gradient = GradientAtom(
            strength=gradient_strength,
            direction='ascent'  # Flow toward higher activation
        )
        self.vortex = VortexAtom(
            center=None,  # Will be set dynamically
            angular_velocity=angular_velocity,
            strength=vortex_strength
        )
        
        # Attention shift state
        self.current_focus = None
        self.target_focus = None
        self.transition_path = []
        self.flow_field = None
        self.shift_count = 0
        self.total_processed = 0
        self.shift_history = []
        self.transition_smoothness = transition_smoothness
    
    def shift_attention(self, field: NDAnalogField, target_location: tuple = None,
                       transition_steps: int = 5, **kwargs) -> NDAnalogField:
        """Shift attention to target location
        
        Args:
            field: Field to shift attention in
            target_location: Target location for attention (None = auto-detect)
            transition_steps: Number of steps for smooth transition
            **kwargs: Additional parameters
            
        Returns:
            Field with shifted attention
        """
        self.total_processed += 1
        
        # Auto-detect target if not provided
        if target_location is None:
            target_location = self._detect_attention_target(field)
        
        if target_location is None:
            return field  # No target found
        
        # Update focus locations
        self.current_focus = self._get_current_focus(field)
        self.target_focus = target_location
        
        # Calculate transition path
        self._calculate_transition_path(transition_steps)
        
        # Apply attention shift
        shifted_field = self._apply_attention_shift(field, target_location)
        
        # Update flow field
        self._update_flow_field(field)
        
        # Record shift
        self._record_shift(target_location, transition_steps)
        
        return shifted_field
    
    def _detect_attention_target(self, field: NDAnalogField) -> tuple:
        """Auto-detect attention target using gradient analysis"""
        # Compute gradient magnitude
        gradient_magnitude = self.gradient.get_gradient_field(field)
        
        # Find peaks in gradient (high attention areas)
        peaks = self.gradient.find_peaks(field, threshold=0.5)
        
        if not peaks:
            return None
        
        # Select strongest peak
        strongest_peak = max(peaks, key=lambda p: p[2])
        return (strongest_peak[0], strongest_peak[1])
    
    def _get_current_focus(self, field: NDAnalogField) -> tuple:
        """Get current focus location from field"""
        if self.current_focus is not None:
            return self.current_focus
        
        # Find current highest activation
        max_idx = np.unravel_index(np.argmax(field.activation), field.activation.shape)
        return max_idx
    
    def _calculate_transition_path(self, transition_steps: int):
        """Calculate smooth transition path from current to target focus"""
        if self.current_focus is None or self.target_focus is None:
            self.transition_path = []
            return
        
        start_y, start_x = self.current_focus
        end_y, end_x = self.target_focus
        
        # Calculate step sizes
        step_y = (end_y - start_y) / transition_steps
        step_x = (end_x - start_x) / transition_steps
        
        # Generate path points
        path = []
        for step in range(transition_steps + 1):
            y = int(start_y + step * step_y)
            x = int(start_x + step * step_x)
            path.append((y, x))
        
        self.transition_path = path
    
    def _apply_attention_shift(self, field: NDAnalogField, target_location: tuple) -> NDAnalogField:
        """Apply attention shift using gradient and vortex dynamics"""
        # Set vortex center to target location
        self.vortex.center = target_location
        
        # Apply gradient flow to identify attention-worthy regions
        gradient_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        self.gradient.apply(gradient_field)
        
        # Apply vortex dynamics for smooth transition
        vortex_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        self.vortex.apply(vortex_field)
        
        # Combine gradient and vortex effects
        combined_activation = (
            gradient_field.activation * 0.6 +
            vortex_field.activation * 0.4
        )
        
        # Apply smooth transition
        if self.transition_smoothness > 0:
            # Blend with original based on transition smoothness
            field.activation = (
                field.activation * (1 - self.transition_smoothness) +
                combined_activation * self.transition_smoothness
            )
        else:
            field.activation = combined_activation
        
        return field
    
    def _update_flow_field(self, field: NDAnalogField):
        """Update flow field representation"""
        if self.vortex.center is not None:
            flow_y, flow_x = self.vortex.get_flow_field(field)
            if flow_y is not None and flow_x is not None:
                self.flow_field = {
                    'flow_y': flow_y,
                    'flow_x': flow_x,
                    'center': self.vortex.center,
                    'strength': self.vortex.strength
                }
        else:
            self.flow_field = None
    
    def _record_shift(self, target_location: tuple, transition_steps: int):
        """Record attention shift for history"""
        self.shift_count += 1
        
        shift_record = {
            'shift_id': self.shift_count,
            'from_location': self.current_focus,
            'to_location': target_location,
            'transition_steps': transition_steps,
            'timestamp': self.total_processed
        }
        
        self.shift_history.append(shift_record)
        
        # Keep only recent history
        if len(self.shift_history) > 50:
            self.shift_history = self.shift_history[-50:]
    
    def get_flow_field(self) -> dict:
        """Get current flow field representation"""
        return self.flow_field.copy() if self.flow_field is not None else None
    
    def get_transition_path(self) -> list:
        """Get current transition path"""
        return self.transition_path.copy()
    
    def get_attention_flow(self, field: NDAnalogField) -> np.ndarray:
        """Get attention flow magnitude at each point"""
        if self.flow_field is None:
            return np.zeros_like(field.activation)
        
        flow_y = self.flow_field['flow_y']
        flow_x = self.flow_field['flow_x']
        
        # Calculate flow magnitude
        flow_magnitude = np.sqrt(flow_y**2 + flow_x**2)
        return flow_magnitude
    
    def get_attention_direction(self, field: NDAnalogField) -> np.ndarray:
        """Get attention flow direction at each point"""
        if self.flow_field is None:
            return np.zeros_like(field.activation)
        
        flow_y = self.flow_field['flow_y']
        flow_x = self.flow_field['flow_x']
        
        # Calculate flow direction (angle)
        flow_direction = np.arctan2(flow_y, flow_x)
        return flow_direction
    
    def get_shift_statistics(self) -> dict:
        """Get statistics about attention shifts"""
        if not self.shift_history:
            return {
                'total_shifts': 0,
                'average_transition_steps': 0,
                'current_focus': self.current_focus,
                'target_focus': self.target_focus
            }
        
        transition_steps = [shift['transition_steps'] for shift in self.shift_history]
        
        return {
            'total_shifts': len(self.shift_history),
            'average_transition_steps': np.mean(transition_steps),
            'current_focus': self.current_focus,
            'target_focus': self.target_focus,
            'transition_path_length': len(self.transition_path),
            'flow_field_active': self.flow_field is not None
        }
    
    def get_shift_history(self, window_size: int = 10) -> list:
        """Get recent attention shift history
        
        Args:
            window_size: Number of recent shifts to return
            
        Returns:
            List of recent attention shifts
        """
        return self.shift_history[-window_size:]
    
    def detect_attention_conflicts(self, field: NDAnalogField) -> list:
        """Detect conflicts between different attention flows
        
        Args:
            field: Field to analyze
            
        Returns:
            List of attention conflict regions
        """
        if self.flow_field is None:
            return []
        
        flow_y = self.flow_field['flow_y']
        flow_x = self.flow_field['flow_x']
        
        # Calculate flow divergence (conflict indicator)
        if len(field.shape) == 2:
            # Compute divergence of flow field
            div_y = np.gradient(flow_y, axis=0)
            div_x = np.gradient(flow_x, axis=1)
            divergence = div_y + div_x
            
            # Find high divergence regions (conflicts)
            conflict_threshold = np.percentile(np.abs(divergence), 90)
            conflict_mask = np.abs(divergence) > conflict_threshold
            
            # Get conflict coordinates
            conflict_coords = np.argwhere(conflict_mask)
            conflicts = [{
                'location': tuple(coord),
                'divergence': divergence[tuple(coord)],
                'severity': abs(divergence[tuple(coord)])
            } for coord in conflict_coords]
            
            return conflicts
        
        return []
    
    def smooth_attention_transition(self, field: NDAnalogField, 
                                  target_location: tuple, steps: int = 10) -> list:
        """Create smooth attention transition over multiple steps
        
        Args:
            field: Field to transition
            target_location: Target location
            steps: Number of transition steps
            
        Returns:
            List of fields representing transition steps
        """
        transition_fields = []
        
        # Calculate transition path
        self._calculate_transition_path(steps)
        
        # Create intermediate fields
        for i, path_point in enumerate(self.transition_path):
            # Create field for this step
            step_field = type('Field', (), {
                'activation': field.activation.copy(),
                'shape': field.shape
            })()
            
            # Apply partial attention shift
            partial_strength = i / steps
            self.vortex.center = path_point
            self.vortex.strength = self.vortex.strength * partial_strength
            
            self.vortex.apply(step_field)
            transition_fields.append(step_field)
        
        return transition_fields
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'current_focus': self.current_focus,
            'target_focus': self.target_focus,
            'transition_path_length': len(self.transition_path),
            'shift_count': self.shift_count,
            'total_processed': self.total_processed,
            'gradient_strength': self.gradient.strength,
            'vortex_strength': self.vortex.strength,
            'vortex_center': self.vortex.center,
            'flow_field_active': self.flow_field is not None,
            'shift_history_length': len(self.shift_history)
        }
    
    def reset(self):
        """Reset attention shift state"""
        self.current_focus = None
        self.target_focus = None
        self.transition_path = []
        self.flow_field = None
        self.shift_count = 0
        self.total_processed = 0
        self.shift_history = []
        self.vortex.center = None
        self.vortex.rotation_angle = 0
    
    def __repr__(self):
        return f"AttentionShift(shifts={self.shift_count}, current={self.current_focus}, target={self.target_focus})"