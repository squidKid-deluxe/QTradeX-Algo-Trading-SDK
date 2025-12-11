# ============================================================================
# PatternCompleter - Complete Partial Patterns
# ============================================================================

"""
PatternCompleter - Complete partial patterns using anticipation and resonance

Composition: Anticipator + Resonator
Category: Reasoning
Complexity: Molecule (50-200 lines)

Completes partial patterns by anticipating what should come next and using
resonance to amplify the most likely completions. This enables pattern
completion, sequence prediction, and filling in missing information.

Example:
    >>> completer = PatternCompleter(anticipation_strength=0.5, resonance_threshold=0.6)
    >>> completed = completer.complete_pattern(partial_field)
    >>> completions = completer.get_completion_candidates()
    >>> confidence = completer.get_completion_confidence()
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.temporal import AnticipatorAtom
    from ...atoms.multi_field import ResonatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.temporal import AnticipatorAtom
    from combinatronix.atoms.multi_field import ResonatorAtom
    from combinatronix.core import NDAnalogField


class PatternCompleter:
    """Complete partial patterns using anticipation and resonance"""
    
    def __init__(self, anticipation_strength: float = 0.5, resonance_threshold: float = 0.6,
                 history_depth: int = 3, amplification: float = 2.0):
        """
        Args:
            anticipation_strength: How much to use anticipation for completion
            resonance_threshold: Threshold for resonance-based completion
            history_depth: How many past states to use for anticipation
            amplification: How much to amplify resonant completions
        """
        self.anticipator = AnticipatorAtom(
            history_depth=history_depth,
            prediction_weight=anticipation_strength
        )
        self.resonator = ResonatorAtom(
            amplification=amplification,
            threshold=resonance_threshold,
            mode='correlation'
        )
        
        # Pattern completion state
        self.completions = []  # List of completion records
        self.completion_templates = {}  # template_id -> template_data
        self.completion_count = 0
        self.total_processed = 0
        self.anticipation_strength = anticipation_strength
        self.resonance_threshold = resonance_threshold
    
    def complete_pattern(self, partial_field: NDAnalogField, 
                        template_id: str = None, context: str = None) -> NDAnalogField:
        """Complete partial pattern using anticipation and resonance
        
        Args:
            partial_field: Field with partial pattern
            template_id: Optional template to use for completion
            context: Context for completion
            
        Returns:
            Field with completed pattern
        """
        self.total_processed += 1
        
        # Create working copy
        completed_field = type('Field', (), {
            'activation': partial_field.activation.copy(),
            'shape': partial_field.shape
        })()
        
        # Step 1: Use anticipation to predict completion
        prediction = self._anticipate_completion(completed_field)
        
        # Step 2: Use resonance to amplify likely completions
        if template_id and template_id in self.completion_templates:
            template = self.completion_templates[template_id]
            resonance_completion = self._resonate_with_template(completed_field, template)
        else:
            resonance_completion = self._resonate_with_history(completed_field)
        
        # Step 3: Combine anticipation and resonance
        final_completion = self._combine_completions(
            completed_field, prediction, resonance_completion
        )
        
        # Step 4: Record completion
        self._record_completion(partial_field, final_completion, template_id, context)
        
        # Update field
        completed_field.activation = final_completion
        return completed_field
    
    def _anticipate_completion(self, field: NDAnalogField) -> np.ndarray:
        """Use anticipation to predict completion
        
        Args:
            field: Field to complete
            
        Returns:
            Predicted completion
        """
        # Add current state to history
        self.anticipator.history.append(field.activation.copy())
        
        # Generate prediction
        prediction = self.anticipator.predict(field)
        
        return prediction
    
    def _resonate_with_template(self, field: NDAnalogField, template: dict) -> np.ndarray:
        """Use resonance with template for completion
        
        Args:
            field: Field to complete
            template: Template pattern to resonate with
            
        Returns:
            Resonance-based completion
        """
        # Create template field
        template_field = type('Field', (), {
            'activation': template['pattern'],
            'shape': template['pattern'].shape
        })()
        
        # Apply resonance
        self.resonator.apply(field, template_field)
        
        return field.activation
    
    def _resonate_with_history(self, field: NDAnalogField) -> np.ndarray:
        """Use resonance with historical patterns
        
        Args:
            field: Field to complete
            
        Returns:
            Resonance-based completion
        """
        if len(self.anticipator.history) < 2:
            return field.activation
        
        # Find most resonant historical pattern
        best_resonance = 0.0
        best_pattern = None
        
        for historical_pattern in self.anticipator.history:
            hist_field = type('Field', (), {
                'activation': historical_pattern,
                'shape': historical_pattern.shape
            })()
            
            resonance_strength = self.resonator.get_resonance_strength(field, hist_field)
            
            if resonance_strength > best_resonance:
                best_resonance = resonance_strength
                best_pattern = historical_pattern
        
        # Apply resonance with best pattern
        if best_pattern is not None:
            best_field = type('Field', (), {
                'activation': best_pattern,
                'shape': best_pattern.shape
            })()
            
            self.resonator.apply(field, best_field)
        
        return field.activation
    
    def _combine_completions(self, original_field: NDAnalogField, 
                           prediction: np.ndarray, resonance: np.ndarray) -> np.ndarray:
        """Combine anticipation and resonance completions
        
        Args:
            original_field: Original partial field
            prediction: Anticipated completion
            resonance: Resonance-based completion
            
        Returns:
            Combined completion
        """
        # Weight the different completion methods
        anticipation_weight = self.anticipation_strength
        resonance_weight = 1.0 - anticipation_weight
        
        # Ensure same shape
        if prediction.shape != original_field.shape:
            prediction = self._resize_to_match(prediction, original_field.shape)
        if resonance.shape != original_field.shape:
            resonance = self._resize_to_match(resonance, original_field.shape)
        
        # Combine completions
        combined = (original_field.activation * 0.3 + 
                   prediction * anticipation_weight * 0.4 + 
                   resonance * resonance_weight * 0.3)
        
        return combined
    
    def _resize_to_match(self, pattern: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize pattern to match target shape
        
        Args:
            pattern: Pattern to resize
            target_shape: Target shape
            
        Returns:
            Resized pattern
        """
        if pattern.shape == target_shape:
            return pattern
        
        # Simple resizing (could be improved with proper interpolation)
        result = np.zeros(target_shape)
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(pattern.shape, target_shape))
        
        if len(min_shape) == 2:
            result[:min_shape[0], :min_shape[1]] = pattern[:min_shape[0], :min_shape[1]]
        else:
            result = pattern
        
        return result
    
    def _record_completion(self, original_field: NDAnalogField, completed_field: NDAnalogField,
                          template_id: str, context: str):
        """Record completion operation"""
        self.completion_count += 1
        
        completion_record = {
            'completion_id': f"completion_{self.completion_count}",
            'original_energy': np.sum(np.abs(original_field.activation)),
            'completed_energy': np.sum(np.abs(completed_field.activation)),
            'template_id': template_id,
            'context': context,
            'anticipation_strength': self.anticipation_strength,
            'resonance_threshold': self.resonance_threshold,
            'timestamp': self.total_processed
        }
        
        self.completions.append(completion_record)
    
    def add_completion_template(self, template_id: str, pattern: np.ndarray, 
                              description: str = "", metadata: dict = None):
        """Add completion template
        
        Args:
            template_id: Unique identifier for template
            pattern: Template pattern
            description: Human-readable description
            metadata: Additional template information
        """
        self.completion_templates[template_id] = {
            'id': template_id,
            'pattern': pattern.copy(),
            'description': description,
            'metadata': metadata or {},
            'usage_count': 0,
            'created_time': self.total_processed
        }
    
    def get_completion_candidates(self, partial_field: NDAnalogField, 
                                 max_candidates: int = 5) -> list:
        """Get completion candidates for partial pattern
        
        Args:
            partial_field: Partial pattern to complete
            max_candidates: Maximum number of candidates
            
        Returns:
            List of completion candidates with scores
        """
        candidates = []
        
        # Generate anticipation-based candidate
        prediction = self._anticipate_completion(partial_field)
        anticipation_score = self._compute_completion_score(partial_field, prediction)
        candidates.append({
            'type': 'anticipation',
            'pattern': prediction,
            'score': anticipation_score,
            'method': 'temporal_prediction'
        })
        
        # Generate template-based candidates
        for template_id, template in self.completion_templates.items():
            template_field = type('Field', (), {
                'activation': partial_field.activation.copy(),
                'shape': partial_field.shape
            })()
            
            resonance_completion = self._resonate_with_template(template_field, template)
            resonance_score = self._compute_completion_score(partial_field, resonance_completion)
            
            candidates.append({
                'type': 'template',
                'template_id': template_id,
                'pattern': resonance_completion,
                'score': resonance_score,
                'method': 'template_resonance'
            })
        
        # Sort by score and return top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:max_candidates]
    
    def _compute_completion_score(self, partial_field: NDAnalogField, 
                                 completion: np.ndarray) -> float:
        """Compute score for completion quality
        
        Args:
            partial_field: Original partial field
            completion: Proposed completion
            
        Returns:
            Completion score (0.0-1.0)
        """
        # Ensure same shape
        if completion.shape != partial_field.shape:
            completion = self._resize_to_match(completion, partial_field.shape)
        
        # Score based on smoothness and coherence
        # Higher score = better completion
        
        # 1. Smoothness: low gradient magnitude
        if len(completion.shape) == 2:
            grad_y, grad_x = np.gradient(completion)
            gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            smoothness = 1.0 - np.mean(gradient_magnitude)
        else:
            smoothness = 0.5
        
        # 2. Coherence: similarity to partial pattern
        partial_mask = partial_field.activation > 0.1
        if np.any(partial_mask):
            coherence = np.corrcoef(
                partial_field.activation[partial_mask].flatten(),
                completion[partial_mask].flatten()
            )[0, 1]
            coherence = max(0, coherence) if not np.isnan(coherence) else 0
        else:
            coherence = 0.5
        
        # 3. Completeness: fills gaps without overfilling
        original_energy = np.sum(np.abs(partial_field.activation))
        completion_energy = np.sum(np.abs(completion))
        completeness = min(1.0, completion_energy / (original_energy + 1e-8))
        
        # Combine scores
        score = (smoothness * 0.4 + coherence * 0.4 + completeness * 0.2)
        return np.clip(score, 0.0, 1.0)
    
    def get_completion_confidence(self, partial_field: NDAnalogField) -> float:
        """Get confidence in completion quality
        
        Args:
            partial_field: Partial pattern
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Get best completion candidate
        candidates = self.get_completion_candidates(partial_field, max_candidates=1)
        
        if not candidates:
            return 0.0
        
        best_candidate = candidates[0]
        return best_candidate['score']
    
    def get_completion_statistics(self) -> dict:
        """Get statistics about pattern completion
        
        Returns:
            Dictionary with completion statistics
        """
        if not self.completions:
            return {
                'total_completions': 0,
                'average_energy_increase': 0.0,
                'template_count': len(self.completion_templates),
                'anticipation_accuracy': 0.0
            }
        
        energy_increases = [c['completed_energy'] - c['original_energy'] 
                          for c in self.completions]
        avg_energy_increase = np.mean(energy_increases)
        
        # Calculate anticipation accuracy
        prediction_errors = []
        for completion in self.completions[-10:]:  # Recent completions
            if hasattr(self.anticipator, 'last_prediction') and self.anticipator.last_prediction is not None:
                error = self.anticipator.get_prediction_error(
                    type('Field', (), {
                        'activation': np.zeros_like(self.anticipator.last_prediction),
                        'shape': self.anticipator.last_prediction.shape
                    })()
                )
                prediction_errors.append(error)
        
        anticipation_accuracy = 1.0 - np.mean(prediction_errors) if prediction_errors else 0.0
        
        return {
            'total_completions': len(self.completions),
            'average_energy_increase': avg_energy_increase,
            'template_count': len(self.completion_templates),
            'anticipation_accuracy': anticipation_accuracy,
            'resonance_threshold': self.resonance_threshold,
            'anticipation_strength': self.anticipation_strength
        }
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'completions': [{
                'completion_id': c['completion_id'],
                'template_id': c['template_id'],
                'energy_increase': c['completed_energy'] - c['original_energy']
            } for c in self.completions],
            'templates': [{
                'id': t['id'],
                'description': t['description'],
                'usage_count': t['usage_count']
            } for t in self.completion_templates.values()],
            'completion_count': self.completion_count,
            'total_processed': self.total_processed,
            'history_depth': self.anticipator.history_depth
        }
    
    def reset(self):
        """Reset pattern completer"""
        self.completions.clear()
        self.completion_templates.clear()
        self.completion_count = 0
        self.total_processed = 0
        self.anticipator.reset()
        self.resonator.resonance_history.clear()
    
    def __repr__(self):
        return f"PatternCompleter(completions={len(self.completions)}, templates={len(self.completion_templates)})"