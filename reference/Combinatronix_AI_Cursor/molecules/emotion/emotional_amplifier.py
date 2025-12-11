# ============================================================================
# EmotionalAmplifier - Amplify and Intensify Emotional Responses
# ============================================================================

"""
EmotionalAmplifier - Amplify and intensify emotional responses using amplification and vortex dynamics

Composition: Amplifier + Vortex
Category: Emotion
Complexity: Molecule (50-200 lines)

Amplifies and intensifies emotional responses by boosting weak emotional
signals and creating dynamic emotional patterns through vortex effects.
This enables emotional intensity control, mood enhancement, and dynamic
emotional expression.

Example:
    >>> emotional_amp = EmotionalAmplifier(amplification_gain=2.0, vortex_strength=0.3)
    >>> intensified_field = emotional_amp.amplify_emotion(emotional_field)
    >>> emotional_intensity = emotional_amp.get_emotional_intensity()
    >>> dynamic_pattern = emotional_amp.create_dynamic_emotion(base_field)
"""

import numpy as np
from collections import deque
try:
    from ...atoms.tension_resolvers import AmplifierAtom
    from ...atoms.field_dynamics import VortexAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.tension_resolvers import AmplifierAtom
    from combinatronix.core import NDAnalogField


class EmotionalAmplifier:
    """Amplify and intensify emotional responses using amplification and vortex dynamics"""
    
    def __init__(self, amplification_gain: float = 2.0, vortex_strength: float = 0.3,
                 amplification_threshold: float = 0.2, angular_velocity: float = 0.5):
        """
        Args:
            amplification_gain: How much to amplify emotional signals
            vortex_strength: Strength of vortex dynamics
            amplification_threshold: Threshold for amplification
            angular_velocity: Speed of vortex rotation
        """
        self.amplifier = AmplifierAtom(
            threshold=amplification_threshold,
            gain=amplification_gain,
            mode='adaptive'
        )
        self.vortex = VortexAtom(
            center=None,  # Will be set dynamically
            angular_velocity=angular_velocity,
            strength=vortex_strength
        )
        
        # Emotional amplification state
        self.emotional_intensity = 0.0
        self.amplification_history = deque(maxlen=50)
        self.vortex_centers = []  # List of vortex center locations
        self.amplification_count = 0
        self.total_processed = 0
        self.intensity_levels = []
        self.dynamic_patterns = {}
    
    def amplify_emotion(self, emotional_field: NDAnalogField, 
                       intensity_level: float = 1.0, context: str = None) -> NDAnalogField:
        """Amplify emotional field
        
        Args:
            emotional_field: Field containing emotional activation
            intensity_level: Desired intensity level (0.0-2.0)
            context: Context for amplification
            
        Returns:
            Amplified emotional field
        """
        self.total_processed += 1
        
        # Create working copy
        amplified_field = type('Field', (), {
            'activation': emotional_field.activation.copy(),
            'shape': emotional_field.shape
        })()
        
        # Step 1: Apply amplification
        self.amplifier.apply(amplified_field)
        
        # Step 2: Apply vortex dynamics for emotional flow
        self._apply_emotional_vortex(amplified_field, intensity_level)
        
        # Step 3: Scale by intensity level
        amplified_field.activation *= intensity_level
        
        # Step 4: Update emotional intensity
        self._update_emotional_intensity(amplified_field)
        
        # Step 5: Record amplification
        self._record_amplification(emotional_field, amplified_field, intensity_level, context)
        
        # Update field
        emotional_field.activation = amplified_field.activation
        
        return emotional_field
    
    def _apply_emotional_vortex(self, field: NDAnalogField, intensity_level: float):
        """Apply vortex dynamics for emotional flow
        
        Args:
            field: Field to apply vortex to
            intensity_level: Intensity level for vortex strength
        """
        # Set vortex center to highest activation point
        if len(field.shape) == 2:
            max_idx = np.unravel_index(np.argmax(field.activation), field.activation.shape)
            self.vortex.center = max_idx
            self.vortex_centers.append(max_idx)
        
        # Adjust vortex strength based on intensity
        original_strength = self.vortex.strength
        self.vortex.strength = original_strength * intensity_level
        
        # Apply vortex
        self.vortex.apply(field)
        
        # Restore original strength
        self.vortex.strength = original_strength
    
    def _update_emotional_intensity(self, field: NDAnalogField):
        """Update emotional intensity level
        
        Args:
            field: Field to analyze
        """
        # Calculate emotional intensity
        energy = np.sum(np.abs(field.activation))
        variance = np.var(field.activation)
        max_activation = np.max(field.activation)
        
        # Combine metrics for intensity
        self.emotional_intensity = (energy * 0.4 + variance * 0.3 + max_activation * 0.3)
        
        # Keep within bounds
        self.emotional_intensity = np.clip(self.emotional_intensity, 0.0, 1.0)
        
        # Add to history
        self.intensity_levels.append(self.emotional_intensity)
    
    def _record_amplification(self, original_field: NDAnalogField, amplified_field: NDAnalogField,
                            intensity_level: float, context: str):
        """Record amplification operation"""
        self.amplification_count += 1
        
        amplification_record = {
            'amplification_id': f"amp_{self.amplification_count}",
            'original_energy': np.sum(np.abs(original_field.activation)),
            'amplified_energy': np.sum(np.abs(amplified_field.activation)),
            'intensity_level': intensity_level,
            'context': context,
            'emotional_intensity': self.emotional_intensity,
            'vortex_center': self.vortex.center,
            'timestamp': self.total_processed
        }
        
        self.amplification_history.append(amplification_record)
    
    def create_dynamic_emotion(self, base_field: NDAnalogField, 
                             emotion_type: str = 'dynamic', duration: int = 10) -> list:
        """Create dynamic emotional pattern over time
        
        Args:
            base_field: Base emotional field
            emotion_type: Type of dynamic emotion
            duration: Number of time steps
            
        Returns:
            List of fields representing dynamic emotion over time
        """
        dynamic_fields = []
        
        for step in range(duration):
            # Create field for this step
            step_field = type('Field', (), {
                'activation': base_field.activation.copy(),
                'shape': base_field.shape
            })()
            
            # Apply time-varying amplification
            time_factor = 1.0 + 0.5 * np.sin(step * 0.5)
            self.amplifier.gain = self.amplifier.gain * time_factor
            
            # Apply amplification
            self.amplifier.apply(step_field)
            
            # Apply vortex with varying center
            if len(step_field.shape) == 2:
                # Move vortex center in circular pattern
                center_y, center_x = step_field.shape[0] // 2, step_field.shape[1] // 2
                radius = min(step_field.shape) // 4
                angle = step * 0.3
                new_center_y = int(center_y + radius * np.sin(angle))
                new_center_x = int(center_x + radius * np.cos(angle))
                
                # Ensure center is within bounds
                new_center_y = max(0, min(step_field.shape[0] - 1, new_center_y))
                new_center_x = max(0, min(step_field.shape[1] - 1, new_center_x))
                
                self.vortex.center = (new_center_y, new_center_x)
                self.vortex.apply(step_field)
            
            dynamic_fields.append(step_field)
        
        # Store dynamic pattern
        pattern_id = f"dynamic_{emotion_type}_{len(self.dynamic_patterns) + 1}"
        self.dynamic_patterns[pattern_id] = {
            'id': pattern_id,
            'emotion_type': emotion_type,
            'duration': duration,
            'fields': [f.activation.copy() for f in dynamic_fields],
            'created_time': self.total_processed
        }
        
        return dynamic_fields
    
    def get_emotional_intensity(self) -> float:
        """Get current emotional intensity
        
        Returns:
            Current emotional intensity (0.0-1.0)
        """
        return self.emotional_intensity
    
    def get_intensity_trend(self, window_size: int = 5) -> str:
        """Get intensity trend over recent history
        
        Args:
            window_size: Number of recent levels to analyze
            
        Returns:
            Trend description ('increasing', 'decreasing', 'stable', 'volatile')
        """
        if len(self.intensity_levels) < window_size:
            return 'insufficient_data'
        
        recent_levels = self.intensity_levels[-window_size:]
        
        # Calculate trend
        if len(recent_levels) < 2:
            return 'stable'
        
        trend_slope = np.polyfit(range(len(recent_levels)), recent_levels, 1)[0]
        level_variance = np.var(recent_levels)
        
        if level_variance > 0.1:
            return 'volatile'
        elif trend_slope > 0.05:
            return 'increasing'
        elif trend_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_amplification_statistics(self) -> dict:
        """Get amplification statistics
        
        Returns:
            Dictionary with amplification statistics
        """
        if not self.amplification_history:
            return {
                'total_amplifications': 0,
                'average_intensity': 0.0,
                'average_energy_increase': 0.0,
                'vortex_centers_count': 0
            }
        
        intensities = [record['emotional_intensity'] for record in self.amplification_history]
        energy_increases = [record['amplified_energy'] - record['original_energy'] 
                          for record in self.amplification_history]
        
        return {
            'total_amplifications': len(self.amplification_history),
            'average_intensity': np.mean(intensities),
            'current_intensity': self.emotional_intensity,
            'average_energy_increase': np.mean(energy_increases),
            'vortex_centers_count': len(self.vortex_centers),
            'dynamic_patterns_count': len(self.dynamic_patterns),
            'amplification_gain': self.amplifier.gain
        }
    
    def get_dynamic_pattern(self, pattern_id: str) -> dict:
        """Get dynamic pattern by ID
        
        Args:
            pattern_id: ID of pattern to retrieve
            
        Returns:
            Pattern data or None if not found
        """
        return self.dynamic_patterns.get(pattern_id)
    
    def get_vortex_flow_field(self, field: NDAnalogField) -> tuple:
        """Get vortex flow field
        
        Args:
            field: Field to analyze
            
        Returns:
            Tuple of (flow_y, flow_x) arrays
        """
        if self.vortex.center is not None:
            return self.vortex.get_flow_field(field)
        return None, None
    
    def create_emotional_cascade(self, trigger_field: NDAnalogField, 
                               cascade_strength: float = 0.8) -> NDAnalogField:
        """Create emotional cascade effect
        
        Args:
            trigger_field: Field that triggers the cascade
            cascade_strength: Strength of cascade effect
            
        Returns:
            Field with cascade effect
        """
        cascade_field = type('Field', (), {
            'activation': trigger_field.activation.copy(),
            'shape': trigger_field.shape
        })()
        
        # Apply multiple amplification steps
        for step in range(3):
            # Amplify
            self.amplifier.apply(cascade_field)
            
            # Apply vortex with moving center
            if len(cascade_field.shape) == 2:
                center_y, center_x = cascade_field.shape[0] // 2, cascade_field.shape[1] // 2
                offset = step * 2
                new_center = (center_y + offset, center_x + offset)
                
                # Ensure within bounds
                new_center = (max(0, min(cascade_field.shape[0] - 1, new_center[0])),
                             max(0, min(cascade_field.shape[1] - 1, new_center[1])))
                
                self.vortex.center = new_center
                self.vortex.apply(cascade_field)
        
        # Scale by cascade strength
        cascade_field.activation *= cascade_strength
        
        return cascade_field
    
    def detect_emotional_overload(self, threshold: float = 0.8) -> bool:
        """Detect if emotional intensity is too high (overload)
        
        Args:
            threshold: Threshold for overload detection
            
        Returns:
            True if experiencing emotional overload
        """
        return self.emotional_intensity > threshold
    
    def dampen_emotion(self, field: NDAnalogField, damping_factor: float = 0.5) -> NDAnalogField:
        """Dampen emotional intensity
        
        Args:
            field: Field to dampen
            damping_factor: How much to dampen (0.0-1.0)
            
        Returns:
            Dampened field
        """
        dampened_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape
        })()
        
        # Reduce activation
        dampened_field.activation *= damping_factor
        
        # Update intensity
        self._update_emotional_intensity(dampened_field)
        
        return dampened_field
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'emotional_intensity': self.emotional_intensity,
            'amplification_count': self.amplification_count,
            'total_processed': self.total_processed,
            'intensity_levels_count': len(self.intensity_levels),
            'vortex_centers_count': len(self.vortex_centers),
            'dynamic_patterns_count': len(self.dynamic_patterns),
            'amplification_gain': self.amplifier.gain,
            'vortex_strength': self.vortex.strength,
            'vortex_center': self.vortex.center
        }
    
    def reset(self):
        """Reset emotional amplifier"""
        self.emotional_intensity = 0.0
        self.amplification_history.clear()
        self.vortex_centers.clear()
        self.amplification_count = 0
        self.total_processed = 0
        self.intensity_levels.clear()
        self.dynamic_patterns.clear()
        self.vortex.center = None
        self.vortex.rotation_angle = 0
    
    def __repr__(self):
        return f"EmotionalAmplifier(intensity={self.emotional_intensity:.2f}, amplifications={self.amplification_count})"

