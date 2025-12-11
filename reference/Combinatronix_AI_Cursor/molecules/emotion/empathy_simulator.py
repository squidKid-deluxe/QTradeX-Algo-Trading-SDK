# ============================================================================
# EmpathySimulator - Simulate Empathetic Responses
# ============================================================================

"""
EmpathySimulator - Simulate empathetic responses using resonance and translation

Composition: Resonator + Translator
Category: Emotion
Complexity: Molecule (50-200 lines)

Simulates empathetic responses by resonating with others' emotional states
and translating them into appropriate responses. This enables emotional
understanding, compassion, and social emotional intelligence.

Example:
    >>> empathy_sim = EmpathySimulator(resonance_threshold=0.7, translation_strength=0.8)
    >>> empathetic_response = empathy_sim.simulate_empathy(others_emotion_field)
    >>> empathy_level = empathy_sim.get_empathy_level()
    >>> emotional_mirror = empathy_sim.create_emotional_mirror(target_field)
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.multi_field import ResonatorAtom, TranslatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.multi_field import ResonatorAtom, TranslatorAtom
    from combinatronix.core import NDAnalogField


class EmpathySimulator:
    """Simulate empathetic responses using resonance and translation"""
    
    def __init__(self, resonance_threshold: float = 0.7, translation_strength: float = 0.8,
                 amplification: float = 1.5, empathy_depth: int = 3):
        """
        Args:
            resonance_threshold: Threshold for emotional resonance
            translation_strength: Strength of emotional translation
            amplification: How much to amplify resonant emotions
            empathy_depth: Depth of empathy processing
        """
        self.resonator = ResonatorAtom(
            amplification=amplification,
            threshold=resonance_threshold,
            mode='correlation'
        )
        self.translator = TranslatorAtom(
            scale_factor=1.0,
            rotation=0.0,
            transformation='linear'
        )
        
        # Empathy simulation state
        self.translation_strength = translation_strength
        self.empathy_depth = empathy_depth
        self.empathy_responses = []  # List of empathy response records
        self.emotional_mirrors = {}  # mirror_id -> mirror_data
        self.empathy_level = 0.0
        self.empathy_count = 0
        self.total_processed = 0
        self.empathy_history = []
    
    def simulate_empathy(self, others_emotion_field: NDAnalogField, 
                        context: str = None, empathy_type: str = 'cognitive') -> NDAnalogField:
        """Simulate empathetic response to others' emotions
        
        Args:
            others_emotion_field: Field containing others' emotional state
            context: Context of the emotional situation
            empathy_type: Type of empathy ('cognitive', 'emotional', 'compassionate')
            
        Returns:
            Field containing empathetic response
        """
        self.total_processed += 1
        
        # Create response field
        empathetic_field = type('Field', (), {
            'activation': np.zeros_like(others_emotion_field.activation),
            'shape': others_emotion_field.shape
        })()
        
        # Step 1: Resonate with others' emotions
        resonance_strength = self._resonate_with_emotion(others_emotion_field, empathetic_field)
        
        # Step 2: Translate emotional state based on empathy type
        translated_emotion = self._translate_emotion(others_emotion_field, empathetic_field, empathy_type)
        
        # Step 3: Apply empathy depth processing
        processed_response = self._apply_empathy_depth(translated_emotion, empathy_type)
        
        # Step 4: Update empathy level
        self._update_empathy_level(resonance_strength, empathy_type)
        
        # Step 5: Record empathy response
        self._record_empathy_response(others_emotion_field, processed_response, 
                                    resonance_strength, context, empathy_type)
        
        return processed_response
    
    def _resonate_with_emotion(self, others_field: NDAnalogField, 
                              response_field: NDAnalogField) -> float:
        """Resonate with others' emotional field
        
        Args:
            others_field: Others' emotional field
            response_field: Field to contain response
            
        Returns:
            Resonance strength
        """
        # Apply resonator to create emotional resonance
        self.resonator.apply(response_field, others_field)
        
        # Calculate resonance strength
        resonance_strength = self.resonator.get_resonance_strength(response_field, others_field)
        
        return resonance_strength
    
    def _translate_emotion(self, source_field: NDAnalogField, target_field: NDAnalogField, 
                          empathy_type: str) -> NDAnalogField:
        """Translate emotional state based on empathy type
        
        Args:
            source_field: Source emotional field
            target_field: Target field for translation
            empathy_type: Type of empathy to apply
            
        Returns:
            Translated emotional field
        """
        # Configure translator based on empathy type
        if empathy_type == 'cognitive':
            # Cognitive empathy: understand without feeling
            self.translator.scale_factor = 0.5
            self.translator.transformation = 'linear'
        elif empathy_type == 'emotional':
            # Emotional empathy: feel what others feel
            self.translator.scale_factor = 1.0
            self.translator.transformation = 'linear'
        elif empathy_type == 'compassionate':
            # Compassionate empathy: feel and want to help
            self.translator.scale_factor = 1.2
            self.translator.transformation = 'nonlinear'
        else:
            # Default: balanced empathy
            self.translator.scale_factor = 0.8
            self.translator.transformation = 'linear'
        
        # Apply translation
        self.translator.translate(source_field, target_field, self.translation_strength)
        
        return target_field
    
    def _apply_empathy_depth(self, emotion_field: NDAnalogField, empathy_type: str) -> NDAnalogField:
        """Apply empathy depth processing
        
        Args:
            emotion_field: Field to process
            empathy_type: Type of empathy
            
        Returns:
            Processed field with empathy depth
        """
        processed_field = type('Field', (), {
            'activation': emotion_field.activation.copy(),
            'shape': emotion_field.shape
        })()
        
        # Apply depth-based processing
        for depth in range(self.empathy_depth):
            if empathy_type == 'cognitive':
                # Cognitive: analyze and understand
                processed_field.activation = self._apply_cognitive_processing(processed_field.activation)
            elif empathy_type == 'emotional':
                # Emotional: feel deeply
                processed_field.activation = self._apply_emotional_processing(processed_field.activation)
            elif empathy_type == 'compassionate':
                # Compassionate: feel and act
                processed_field.activation = self._apply_compassionate_processing(processed_field.activation)
        
        return processed_field
    
    def _apply_cognitive_processing(self, activation: np.ndarray) -> np.ndarray:
        """Apply cognitive empathy processing"""
        # Analyze emotional patterns
        if len(activation.shape) == 2:
            # Smooth and analyze
            from scipy import ndimage
            smoothed = ndimage.gaussian_filter(activation, sigma=1.0)
            return smoothed * 0.7 + activation * 0.3
        return activation
    
    def _apply_emotional_processing(self, activation: np.ndarray) -> np.ndarray:
        """Apply emotional empathy processing"""
        # Amplify emotional response
        amplified = activation * 1.2
        return np.clip(amplified, 0, 1)
    
    def _apply_compassionate_processing(self, activation: np.ndarray) -> np.ndarray:
        """Apply compassionate empathy processing"""
        # Combine understanding with action tendency
        if len(activation.shape) == 2:
            # Add helping tendency (increased activation in center)
            center_y, center_x = activation.shape[0] // 2, activation.shape[1] // 2
            for i in range(activation.shape[0]):
                for j in range(activation.shape[1]):
                    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    helping_factor = np.exp(-distance / 2.0) * 0.3
                    activation[i, j] += helping_factor
        
        return np.clip(activation, 0, 1)
    
    def _update_empathy_level(self, resonance_strength: float, empathy_type: str):
        """Update current empathy level
        
        Args:
            resonance_strength: Strength of emotional resonance
            empathy_type: Type of empathy being used
        """
        # Calculate empathy level based on resonance and type
        type_weights = {
            'cognitive': 0.6,
            'emotional': 1.0,
            'compassionate': 1.2
        }
        
        weight = type_weights.get(empathy_type, 0.8)
        self.empathy_level = resonance_strength * weight
        
        # Keep running average
        if len(self.empathy_history) > 0:
            self.empathy_level = (self.empathy_level + np.mean(self.empathy_history[-5:])) / 2
        
        self.empathy_history.append(self.empathy_level)
    
    def _record_empathy_response(self, others_field: NDAnalogField, response_field: NDAnalogField,
                               resonance_strength: float, context: str, empathy_type: str):
        """Record empathy response"""
        self.empathy_count += 1
        
        response_record = {
            'response_id': f"empathy_{self.empathy_count}",
            'others_energy': np.sum(np.abs(others_field.activation)),
            'response_energy': np.sum(np.abs(response_field.activation)),
            'resonance_strength': resonance_strength,
            'empathy_type': empathy_type,
            'context': context,
            'empathy_level': self.empathy_level,
            'timestamp': self.total_processed
        }
        
        self.empathy_responses.append(response_record)
    
    def create_emotional_mirror(self, target_field: NDAnalogField, 
                               mirror_name: str = None) -> str:
        """Create emotional mirror for sustained empathy
        
        Args:
            target_field: Field to mirror
            mirror_name: Name for the mirror
            
        Returns:
            Mirror ID
        """
        if mirror_name is None:
            mirror_name = f"mirror_{len(self.emotional_mirrors) + 1}"
        
        mirror_id = f"mirror_{id(target_field)}"
        
        mirror_data = {
            'id': mirror_id,
            'name': mirror_name,
            'target_field': target_field.activation.copy(),
            'mirror_field': np.zeros_like(target_field.activation),
            'resonance_strength': 0.0,
            'created_time': self.total_processed,
            'usage_count': 0
        }
        
        self.emotional_mirrors[mirror_id] = mirror_data
        
        return mirror_id
    
    def update_emotional_mirror(self, mirror_id: str, current_field: NDAnalogField) -> bool:
        """Update emotional mirror with current state
        
        Args:
            mirror_id: ID of mirror to update
            current_field: Current emotional field
            
        Returns:
            True if mirror updated successfully
        """
        if mirror_id not in self.emotional_mirrors:
            return False
        
        mirror = self.emotional_mirrors[mirror_id]
        mirror['usage_count'] += 1
        
        # Calculate resonance with target
        target_field = type('Field', (), {
            'activation': mirror['target_field'],
            'shape': mirror['target_field'].shape
        })()
        
        resonance_strength = self.resonator.get_resonance_strength(current_field, target_field)
        mirror['resonance_strength'] = resonance_strength
        
        # Update mirror field
        if resonance_strength > self.resonator.threshold:
            self.resonator.apply(current_field, target_field)
            mirror['mirror_field'] = current_field.activation.copy()
        
        return True
    
    def get_empathy_level(self) -> float:
        """Get current empathy level
        
        Returns:
            Current empathy level (0.0-1.0)
        """
        return self.empathy_level
    
    def get_empathy_statistics(self) -> dict:
        """Get empathy simulation statistics
        
        Returns:
            Dictionary with empathy statistics
        """
        if not self.empathy_responses:
            return {
                'total_responses': 0,
                'average_empathy_level': 0.0,
                'empathy_types': {},
                'mirror_count': len(self.emotional_mirrors)
            }
        
        empathy_types = [r['empathy_type'] for r in self.empathy_responses]
        type_counts = {}
        for emp_type in empathy_types:
            type_counts[emp_type] = type_counts.get(emp_type, 0) + 1
        
        empathy_levels = [r['empathy_level'] for r in self.empathy_responses]
        avg_empathy_level = np.mean(empathy_levels)
        
        return {
            'total_responses': len(self.empathy_responses),
            'average_empathy_level': avg_empathy_level,
            'current_empathy_level': self.empathy_level,
            'empathy_types': type_counts,
            'mirror_count': len(self.emotional_mirrors),
            'average_resonance': np.mean([r['resonance_strength'] for r in self.empathy_responses])
        }
    
    def get_emotional_mirror(self, mirror_id: str) -> dict:
        """Get emotional mirror by ID
        
        Args:
            mirror_id: ID of mirror to retrieve
            
        Returns:
            Mirror data or None if not found
        """
        return self.emotional_mirrors.get(mirror_id)
    
    def get_empathy_history(self, window_size: int = 10) -> list:
        """Get recent empathy history
        
        Args:
            window_size: Number of recent responses to return
            
        Returns:
            List of recent empathy responses
        """
        return self.empathy_responses[-window_size:]
    
    def detect_empathy_fatigue(self, threshold: float = 0.3) -> bool:
        """Detect if empathy levels are too low (fatigue)
        
        Args:
            threshold: Threshold for fatigue detection
            
        Returns:
            True if experiencing empathy fatigue
        """
        if len(self.empathy_history) < 5:
            return False
        
        recent_empathy = np.mean(self.empathy_history[-5:])
        return recent_empathy < threshold
    
    def restore_empathy(self, restoration_strength: float = 0.5) -> float:
        """Restore empathy levels
        
        Args:
            restoration_strength: How much to restore
            
        Returns:
            New empathy level
        """
        # Increase empathy level
        self.empathy_level = min(1.0, self.empathy_level + restoration_strength)
        
        # Add to history
        self.empathy_history.append(self.empathy_level)
        
        return self.empathy_level
    
    def get_empathy_trend(self, window_size: int = 5) -> str:
        """Get empathy trend over recent history
        
        Args:
            window_size: Number of recent responses to analyze
            
        Returns:
            Trend description ('increasing', 'decreasing', 'stable', 'volatile')
        """
        if len(self.empathy_history) < window_size:
            return 'insufficient_data'
        
        recent_levels = self.empathy_history[-window_size:]
        
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
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'empathy_level': self.empathy_level,
            'empathy_count': self.empathy_count,
            'total_processed': self.total_processed,
            'response_count': len(self.empathy_responses),
            'mirror_count': len(self.emotional_mirrors),
            'resonance_threshold': self.resonator.threshold,
            'translation_strength': self.translation_strength,
            'empathy_depth': self.empathy_depth
        }
    
    def reset(self):
        """Reset empathy simulator"""
        self.empathy_responses.clear()
        self.emotional_mirrors.clear()
        self.empathy_level = 0.0
        self.empathy_count = 0
        self.total_processed = 0
        self.empathy_history.clear()
        self.resonator.resonance_history.clear()
    
    def __repr__(self):
        return f"EmpathySimulator(level={self.empathy_level:.2f}, responses={len(self.empathy_responses)})"

