# ============================================================================
# PatternRecognizer - Recognize and Classify Patterns
# ============================================================================

"""
PatternRecognizer - Recognize patterns using resonance and binding

Composition: Resonator + Binder
Category: Perception
Complexity: Molecule (50-200 lines)

Recognizes patterns by resonating with stored templates and creating
associative bindings. This enables pattern classification, template matching,
and recognition of familiar structures in visual or other sensory data.

Example:
    >>> recognizer = PatternRecognizer(recognition_threshold=0.7)
    >>> recognizer.add_template("circle", circle_pattern)
    >>> result = recognizer.recognize(field)
    >>> matches = recognizer.get_matches()
"""

import numpy as np
try:
    from ...atoms.multi_field import ResonatorAtom, BinderAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.multi_field import ResonatorAtom, BinderAtom
    from combinatronix.core import NDAnalogField


class PatternRecognizer:
    """Recognize patterns using resonance and associative binding"""
    
    def __init__(self, recognition_threshold: float = 0.7, 
                 amplification: float = 2.0, binding_strength: float = 0.8):
        """
        Args:
            recognition_threshold: Minimum resonance for recognition
            amplification: How much to amplify recognized patterns
            binding_strength: Strength of pattern bindings
        """
        self.resonator = ResonatorAtom(
            amplification=amplification,
            threshold=recognition_threshold,
            mode='correlation'
        )
        self.binder = BinderAtom(binding_strength=binding_strength)
        
        # Pattern storage
        self.templates = {}  # name -> pattern_data
        self.recognition_history = []
        self.current_matches = []
        self.recognition_count = 0
    
    def add_template(self, name: str, pattern: np.ndarray, 
                    description: str = "", metadata: dict = None):
        """Add a pattern template for recognition
        
        Args:
            name: Unique identifier for the pattern
            pattern: Pattern data (numpy array)
            description: Human-readable description
            metadata: Additional pattern information
        """
        self.templates[name] = {
            'pattern': pattern.copy(),
            'description': description,
            'metadata': metadata or {},
            'recognition_count': 0,
            'last_recognized': 0
        }
    
    def recognize(self, field: NDAnalogField, **kwargs) -> NDAnalogField:
        """Recognize patterns in the field
        
        Args:
            field: Input field to analyze
            **kwargs: Additional parameters
            
        Returns:
            Field with recognition information in activation layer
        """
        self.recognition_count += 1
        self.current_matches = []
        
        # Create a copy for processing
        result_field = type('Field', (), {
            'activation': field.activation.copy(),
            'shape': field.shape,
            'phase': getattr(field, 'phase', None)
        })()
        
        # Try to recognize each template
        for template_name, template_data in self.templates.items():
            # Create field from template
            template_field = type('Field', (), {
                'activation': template_data['pattern'],
                'shape': template_data['pattern'].shape,
                'phase': None
            })()
            
            # Compute resonance with template
            resonance_strength = self.resonator.get_resonance_strength(
                result_field, template_field
            )
            
            # Check if pattern is recognized
            if resonance_strength >= self.resonator.threshold:
                # Found a match!
                match_info = {
                    'template_name': template_name,
                    'resonance_strength': resonance_strength,
                    'template_data': template_data,
                    'recognition_time': self.recognition_count,
                    'location': self._find_match_location(result_field, template_field)
                }
                
                self.current_matches.append(match_info)
                
                # Update template statistics
                template_data['recognition_count'] += 1
                template_data['last_recognized'] = self.recognition_count
                
                # Apply resonance amplification
                self.resonator.apply(result_field, template_field)
                
                # Create binding for future recognition
                self.binder.bind(result_field, template_field)
        
        # Store recognition result
        self.recognition_history.append({
            'time': self.recognition_count,
            'matches': len(self.current_matches),
            'field_energy': np.sum(np.abs(result_field.activation))
        })
        
        # Update field with recognition results
        field.activation = result_field.activation
        
        return field
    
    def _find_match_location(self, field: NDAnalogField, template_field: NDAnalogField) -> tuple:
        """Find the location where template best matches field"""
        if field.shape != template_field.shape:
            # For different shapes, return center
            return (field.shape[0] // 2, field.shape[1] // 2)
        
        # Find location with highest correlation
        correlation_map = self._compute_correlation_map(field, template_field)
        max_idx = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)
        return max_idx
    
    def _compute_correlation_map(self, field: NDAnalogField, template_field: NDAnalogField) -> np.ndarray:
        """Compute local correlation map between field and template"""
        # Simple correlation computation
        field_norm = field.activation / (np.linalg.norm(field.activation) + 1e-8)
        template_norm = template_field.activation / (np.linalg.norm(template_field.activation) + 1e-8)
        
        # Pointwise correlation
        correlation = field_norm * template_norm
        return correlation
    
    def get_matches(self) -> list:
        """Get current recognition matches"""
        return self.current_matches.copy()
    
    def get_template_statistics(self) -> dict:
        """Get statistics about stored templates"""
        stats = {}
        for name, data in self.templates.items():
            stats[name] = {
                'recognition_count': data['recognition_count'],
                'last_recognized': data['last_recognized'],
                'description': data['description'],
                'pattern_shape': data['pattern'].shape,
                'pattern_energy': np.sum(np.abs(data['pattern']))
            }
        return stats
    
    def get_recognition_history(self, window_size: int = 10) -> list:
        """Get recent recognition history"""
        return self.recognition_history[-window_size:]
    
    def find_similar_patterns(self, pattern: np.ndarray, 
                             similarity_threshold: float = 0.8) -> list:
        """Find templates similar to given pattern"""
        similar = []
        
        for name, template_data in self.templates.items():
            # Compute similarity
            template_pattern = template_data['pattern']
            
            # Ensure same shape for comparison
            if pattern.shape == template_pattern.shape:
                similarity = self._compute_pattern_similarity(pattern, template_pattern)
                
                if similarity >= similarity_threshold:
                    similar.append({
                        'name': name,
                        'similarity': similarity,
                        'template_data': template_data
                    })
        
        # Sort by similarity
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar
    
    def _compute_pattern_similarity(self, pattern_a: np.ndarray, pattern_b: np.ndarray) -> float:
        """Compute similarity between two patterns"""
        # Normalize patterns
        norm_a = pattern_a / (np.linalg.norm(pattern_a) + 1e-8)
        norm_b = pattern_b / (np.linalg.norm(pattern_b) + 1e-8)
        
        # Compute correlation
        correlation = np.dot(norm_a.flatten(), norm_b.flatten())
        return correlation
    
    def update_template(self, name: str, new_pattern: np.ndarray, 
                       learning_rate: float = 0.1):
        """Update existing template with new pattern (learning)"""
        if name in self.templates:
            old_pattern = self.templates[name]['pattern']
            
            # Weighted average for learning
            updated_pattern = (1 - learning_rate) * old_pattern + learning_rate * new_pattern
            
            self.templates[name]['pattern'] = updated_pattern
            self.templates[name]['metadata']['last_updated'] = self.recognition_count
    
    def remove_template(self, name: str):
        """Remove a template from recognition"""
        if name in self.templates:
            del self.templates[name]
    
    def clear_templates(self):
        """Remove all templates"""
        self.templates.clear()
    
    def get_recognition_confidence(self) -> float:
        """Get overall confidence in current recognition"""
        if not self.current_matches:
            return 0.0
        
        # Average resonance strength of matches
        avg_resonance = np.mean([match['resonance_strength'] for match in self.current_matches])
        return avg_resonance
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'templates': {name: {
                'pattern_shape': data['pattern'].shape,
                'recognition_count': data['recognition_count'],
                'last_recognized': data['last_recognized']
            } for name, data in self.templates.items()},
            'current_matches': self.current_matches.copy(),
            'recognition_count': self.recognition_count,
            'recognition_confidence': self.get_recognition_confidence(),
            'binding_count': self.binder.get_binding_count()
        }
    
    def reset(self):
        """Reset recognizer state"""
        self.current_matches = []
        self.recognition_history = []
        self.recognition_count = 0
        self.binder.clear_bindings()
        self.resonator.resonance_history.clear()
    
    def __repr__(self):
        return f"PatternRecognizer(templates={len(self.templates)}, threshold={self.resonator.threshold:.2f})"