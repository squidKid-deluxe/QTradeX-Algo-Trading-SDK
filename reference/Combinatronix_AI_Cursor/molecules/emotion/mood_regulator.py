# ============================================================================
# MoodRegulator - Regulate Emotional States and Moods
# ============================================================================

"""
MoodRegulator - Regulate emotional states using balancing and damping

Composition: Balancer + Damper
Category: Emotion
Complexity: Molecule (50-200 lines)

Regulates emotional states by balancing opposing emotions and damping
extreme emotional responses. This enables emotional stability, mood
regulation, and adaptive emotional responses to changing circumstances.

Example:
    >>> mood_regulator = MoodRegulator(equilibrium_rate=0.3, damping_threshold=0.8)
    >>> regulated_field = mood_regulator.regulate_mood(emotional_field)
    >>> mood_state = mood_regulator.get_mood_state()
    >>> mood_history = mood_regulator.get_mood_history()
"""

import numpy as np
from collections import deque
try:
    from ...atoms.tension_resolvers import BalancerAtom, DamperAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.tension_resolvers import BalancerAtom, DamperAtom
    from combinatronix.core import NDAnalogField


class MoodRegulator:
    """Regulate emotional states using balancing and damping"""
    
    def __init__(self, equilibrium_rate: float = 0.3, damping_threshold: float = 0.8,
                 damping_rate: float = 0.5, mood_history_size: int = 50):
        """
        Args:
            equilibrium_rate: How fast to move toward emotional equilibrium
            damping_threshold: Activation level that triggers damping
            damping_rate: How much to reduce extreme emotional responses
            mood_history_size: Number of mood states to remember
        """
        self.balancer = BalancerAtom(
            equilibrium_rate=equilibrium_rate,
            min_tension=0.1
        )
        self.damper = DamperAtom(
            threshold=damping_threshold,
            damping_rate=damping_rate,
            mode='soft'
        )
        
        # Mood regulation state
        self.mood_history = deque(maxlen=mood_history_size)
        self.current_mood = None
        self.mood_stability = 0.0
        self.emotional_energy = 0.0
        self.regulation_count = 0
        self.total_processed = 0
        self.mood_transitions = []
    
    def regulate_mood(self, emotional_field: NDAnalogField, 
                     target_mood: str = None, context: str = None) -> NDAnalogField:
        """Regulate mood in emotional field
        
        Args:
            emotional_field: Field containing emotional activation
            target_mood: Desired mood state (optional)
            context: Context for mood regulation
            
        Returns:
            Field with regulated mood
        """
        self.total_processed += 1
        
        # Create working copy
        regulated_field = type('Field', (), {
            'activation': emotional_field.activation.copy(),
            'shape': emotional_field.shape
        })()
        
        # Step 1: Balance internal emotional tensions
        self.balancer.apply(regulated_field)
        
        # Step 2: Dampen extreme emotional responses
        self.damper.apply(regulated_field)
        
        # Step 3: Apply target mood if specified
        if target_mood:
            self._apply_target_mood(regulated_field, target_mood)
        
        # Step 4: Update mood state
        self._update_mood_state(regulated_field, context)
        
        # Step 5: Record regulation
        self._record_mood_regulation(emotional_field, regulated_field, target_mood, context)
        
        # Update field
        emotional_field.activation = regulated_field.activation
        
        return emotional_field
    
    def _apply_target_mood(self, field: NDAnalogField, target_mood: str):
        """Apply specific target mood to field
        
        Args:
            field: Field to modify
            target_mood: Target mood to apply
        """
        mood_patterns = {
            'calm': self._create_calm_pattern(field.shape),
            'excited': self._create_excited_pattern(field.shape),
            'focused': self._create_focused_pattern(field.shape),
            'relaxed': self._create_relaxed_pattern(field.shape),
            'energetic': self._create_energetic_pattern(field.shape),
            'serene': self._create_serene_pattern(field.shape)
        }
        
        if target_mood in mood_patterns:
            target_pattern = mood_patterns[target_mood]
            # Blend with current state
            field.activation = field.activation * 0.7 + target_pattern * 0.3
    
    def _create_calm_pattern(self, shape: tuple) -> np.ndarray:
        """Create calm mood pattern"""
        pattern = np.zeros(shape)
        if len(shape) == 2:
            # Soft, centered activation
            center_y, center_x = shape[0] // 2, shape[1] // 2
            for i in range(shape[0]):
                for j in range(shape[1]):
                    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    pattern[i, j] = np.exp(-distance / 3.0) * 0.3
        return pattern
    
    def _create_excited_pattern(self, shape: tuple) -> np.ndarray:
        """Create excited mood pattern"""
        pattern = np.zeros(shape)
        if len(shape) == 2:
            # High energy, scattered activation
            pattern = np.random.random(shape) * 0.8
        return pattern
    
    def _create_focused_pattern(self, shape: tuple) -> np.ndarray:
        """Create focused mood pattern"""
        pattern = np.zeros(shape)
        if len(shape) == 2:
            # Concentrated activation in center
            center_y, center_x = shape[0] // 2, shape[1] // 2
            pattern[center_y-1:center_y+2, center_x-1:center_x+2] = 0.7
        return pattern
    
    def _create_relaxed_pattern(self, shape: tuple) -> np.ndarray:
        """Create relaxed mood pattern"""
        pattern = np.zeros(shape)
        if len(shape) == 2:
            # Gentle, wave-like activation
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pattern[i, j] = 0.2 * np.sin(i * 0.5) * np.cos(j * 0.5) + 0.3
        return np.clip(pattern, 0, 1)
    
    def _create_energetic_pattern(self, shape: tuple) -> np.ndarray:
        """Create energetic mood pattern"""
        pattern = np.zeros(shape)
        if len(shape) == 2:
            # High activation with movement-like patterns
            for i in range(shape[0]):
                for j in range(shape[1]):
                    pattern[i, j] = 0.5 + 0.3 * np.sin(i + j) + 0.2 * np.random.random()
        return np.clip(pattern, 0, 1)
    
    def _create_serene_pattern(self, shape: tuple) -> np.ndarray:
        """Create serene mood pattern"""
        pattern = np.zeros(shape)
        if len(shape) == 2:
            # Very low, uniform activation
            pattern.fill(0.1)
        return pattern
    
    def _update_mood_state(self, field: NDAnalogField, context: str):
        """Update current mood state based on field
        
        Args:
            field: Current emotional field
            context: Context for mood
        """
        # Analyze emotional characteristics
        energy = np.sum(np.abs(field.activation))
        variance = np.var(field.activation)
        max_activation = np.max(field.activation)
        
        # Determine mood based on characteristics
        if energy < 0.1:
            mood = 'serene'
        elif energy < 0.3 and variance < 0.1:
            mood = 'calm'
        elif energy < 0.5 and variance < 0.2:
            mood = 'relaxed'
        elif energy < 0.7 and variance < 0.3:
            mood = 'focused'
        elif energy < 0.9 and variance > 0.3:
            mood = 'energetic'
        else:
            mood = 'excited'
        
        # Update mood state
        self.current_mood = mood
        self.emotional_energy = energy
        
        # Calculate mood stability
        if len(self.mood_history) > 1:
            recent_moods = [m['mood'] for m in list(self.mood_history)[-5:]]
            self.mood_stability = len(set(recent_moods)) / len(recent_moods)
        else:
            self.mood_stability = 1.0
        
        # Record mood transition
        if len(self.mood_history) > 0:
            previous_mood = self.mood_history[-1]['mood']
            if previous_mood != mood:
                self.mood_transitions.append({
                    'from': previous_mood,
                    'to': mood,
                    'timestamp': self.total_processed,
                    'context': context
                })
    
    def _record_mood_regulation(self, original_field: NDAnalogField, 
                               regulated_field: NDAnalogField, target_mood: str, context: str):
        """Record mood regulation operation"""
        self.regulation_count += 1
        
        mood_record = {
            'regulation_id': f"mood_reg_{self.regulation_count}",
            'mood': self.current_mood,
            'target_mood': target_mood,
            'context': context,
            'emotional_energy': self.emotional_energy,
            'mood_stability': self.mood_stability,
            'original_energy': np.sum(np.abs(original_field.activation)),
            'regulated_energy': np.sum(np.abs(regulated_field.activation)),
            'timestamp': self.total_processed
        }
        
        self.mood_history.append(mood_record)
    
    def get_mood_state(self) -> dict:
        """Get current mood state
        
        Returns:
            Dictionary with current mood information
        """
        return {
            'current_mood': self.current_mood,
            'emotional_energy': self.emotional_energy,
            'mood_stability': self.mood_stability,
            'regulation_count': self.regulation_count,
            'total_processed': self.total_processed
        }
    
    def get_mood_history(self, window_size: int = 10) -> list:
        """Get recent mood history
        
        Args:
            window_size: Number of recent mood states to return
            
        Returns:
            List of recent mood states
        """
        return list(self.mood_history)[-window_size:]
    
    def get_mood_transitions(self) -> list:
        """Get mood transition history
        
        Returns:
            List of mood transitions
        """
        return self.mood_transitions.copy()
    
    def get_mood_statistics(self) -> dict:
        """Get mood regulation statistics
        
        Returns:
            Dictionary with mood statistics
        """
        if not self.mood_history:
            return {
                'total_regulations': 0,
                'mood_diversity': 0.0,
                'average_stability': 0.0,
                'most_common_mood': None
            }
        
        moods = [m['mood'] for m in self.mood_history]
        mood_counts = {}
        for mood in moods:
            mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        most_common_mood = max(mood_counts, key=mood_counts.get) if mood_counts else None
        mood_diversity = len(set(moods)) / len(moods) if moods else 0.0
        
        stabilities = [m['mood_stability'] for m in self.mood_history]
        avg_stability = np.mean(stabilities) if stabilities else 0.0
        
        return {
            'total_regulations': len(self.mood_history),
            'mood_diversity': mood_diversity,
            'average_stability': avg_stability,
            'most_common_mood': most_common_mood,
            'transition_count': len(self.mood_transitions),
            'current_energy': self.emotional_energy
        }
    
    def detect_emotional_instability(self, threshold: float = 0.3) -> bool:
        """Detect if emotional state is unstable
        
        Args:
            threshold: Stability threshold for instability detection
            
        Returns:
            True if emotionally unstable
        """
        return self.mood_stability < threshold
    
    def stabilize_mood(self, field: NDAnalogField, stabilization_strength: float = 0.8) -> NDAnalogField:
        """Apply additional stabilization to unstable mood
        
        Args:
            field: Field to stabilize
            stabilization_strength: How much to stabilize
            
        Returns:
            Stabilized field
        """
        if not self.detect_emotional_instability():
            return field
        
        # Apply stronger damping
        original_damping_rate = self.damper.damping_rate
        self.damper.damping_rate = min(1.0, original_damping_rate * stabilization_strength)
        
        # Apply regulation
        stabilized_field = self.regulate_mood(field, context="stabilization")
        
        # Restore original damping rate
        self.damper.damping_rate = original_damping_rate
        
        return stabilized_field
    
    def get_emotional_energy(self) -> float:
        """Get current emotional energy level
        
        Returns:
            Current emotional energy
        """
        return self.emotional_energy
    
    def get_mood_trend(self, window_size: int = 5) -> str:
        """Get mood trend over recent history
        
        Args:
            window_size: Number of recent states to analyze
            
        Returns:
            Trend description ('stable', 'increasing', 'decreasing', 'volatile')
        """
        if len(self.mood_history) < window_size:
            return 'insufficient_data'
        
        recent_energies = [m['emotional_energy'] for m in list(self.mood_history)[-window_size:]]
        
        # Calculate trend
        if len(recent_energies) < 2:
            return 'stable'
        
        trend_slope = np.polyfit(range(len(recent_energies)), recent_energies, 1)[0]
        energy_variance = np.var(recent_energies)
        
        if energy_variance > 0.1:
            return 'volatile'
        elif trend_slope > 0.05:
            return 'increasing'
        elif trend_slope < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'current_mood': self.current_mood,
            'emotional_energy': self.emotional_energy,
            'mood_stability': self.mood_stability,
            'regulation_count': self.regulation_count,
            'total_processed': self.total_processed,
            'mood_history_size': len(self.mood_history),
            'transition_count': len(self.mood_transitions),
            'equilibrium_rate': self.balancer.equilibrium_rate,
            'damping_threshold': self.damper.threshold
        }
    
    def reset(self):
        """Reset mood regulator"""
        self.mood_history.clear()
        self.current_mood = None
        self.mood_stability = 0.0
        self.emotional_energy = 0.0
        self.regulation_count = 0
        self.total_processed = 0
        self.mood_transitions.clear()
        self.balancer.tension_history.clear()
    
    def __repr__(self):
        return f"MoodRegulator(mood={self.current_mood}, energy={self.emotional_energy:.2f}, stability={self.mood_stability:.2f})"

