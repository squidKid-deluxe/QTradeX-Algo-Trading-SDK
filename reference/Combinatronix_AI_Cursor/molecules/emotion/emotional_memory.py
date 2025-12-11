# ============================================================================
# EmotionalMemory - Store and Retrieve Emotional Experiences
# ============================================================================

"""
EmotionalMemory - Store and retrieve emotional experiences using memory traces and resonance

Composition: MemoryTrace + Resonator
Category: Emotion
Complexity: Molecule (50-200 lines)

Stores emotional experiences with their associated contexts and retrieves
them through resonance-based recall. This enables emotional learning,
trauma processing, and emotional pattern recognition.

Example:
    >>> emotional_memory = EmotionalMemory(accumulation_rate=0.2, resonance_threshold=0.6)
    >>> emotional_memory.store_experience(emotional_field, "joy", context="birthday")
    >>> recalled = emotional_memory.recall_by_emotion("joy")
    >>> similar_experiences = emotional_memory.find_similar_emotions(current_field)
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.temporal import MemoryTraceAtom
    from ...atoms.multi_field import ResonatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.temporal import MemoryTraceAtom
    from combinatronix.atoms.multi_field import ResonatorAtom
    from combinatronix.core import NDAnalogField


class EmotionalMemory:
    """Store and retrieve emotional experiences using memory traces and resonance"""
    
    def __init__(self, accumulation_rate: float = 0.2, resonance_threshold: float = 0.6,
                 decay_rate: float = 0.99, amplification: float = 1.5):
        """
        Args:
            accumulation_rate: Rate at which emotional experiences are stored
            resonance_threshold: Threshold for resonance-based recall
            decay_rate: Rate at which emotional memories decay
            amplification: How much to amplify resonant memories
        """
        self.memory_trace = MemoryTraceAtom(
            accumulation_rate=accumulation_rate,
            decay_rate=decay_rate,
            threshold=0.01
        )
        self.resonator = ResonatorAtom(
            amplification=amplification,
            threshold=resonance_threshold,
            mode='correlation'
        )
        
        # Emotional memory state
        self.emotional_experiences = {}  # experience_id -> experience_data
        self.emotion_index = defaultdict(list)  # emotion -> [experience_ids]
        self.context_index = defaultdict(list)  # context -> [experience_ids]
        self.experience_count = 0
        self.total_processed = 0
        self.recall_history = []
    
    def store_experience(self, emotional_field: NDAnalogField, emotion: str, 
                        context: str = None, intensity: float = 1.0, 
                        metadata: dict = None) -> str:
        """Store emotional experience
        
        Args:
            emotional_field: Field containing emotional activation
            emotion: Emotion label (e.g., 'joy', 'sadness', 'anger')
            context: Context of the emotional experience
            intensity: Intensity of the emotion (0.0-1.0)
            metadata: Additional metadata about the experience
            
        Returns:
            Experience ID
        """
        self.total_processed += 1
        self.experience_count += 1
        
        experience_id = f"emotion_{self.experience_count}"
        
        # Store experience data
        experience_data = {
            'id': experience_id,
            'emotion': emotion,
            'context': context,
            'intensity': intensity,
            'field': emotional_field.activation.copy(),
            'field_shape': emotional_field.shape,
            'metadata': metadata or {},
            'timestamp': self.total_processed,
            'access_count': 0,
            'resonance_strength': 0.0
        }
        
        self.emotional_experiences[experience_id] = experience_data
        
        # Update indices
        self.emotion_index[emotion].append(experience_id)
        if context:
            self.context_index[context].append(experience_id)
        
        # Apply memory trace
        self.memory_trace.apply(emotional_field)
        
        return experience_id
    
    def recall_by_emotion(self, emotion: str, max_results: int = 5) -> list:
        """Recall experiences by emotion type
        
        Args:
            emotion: Emotion to recall
            max_results: Maximum number of results
            
        Returns:
            List of recalled experiences
        """
        if emotion not in self.emotion_index:
            return []
        
        experience_ids = self.emotion_index[emotion]
        recalled_experiences = []
        
        for exp_id in experience_ids[-max_results:]:  # Get most recent
            if exp_id in self.emotional_experiences:
                exp = self.emotional_experiences[exp_id]
                exp['access_count'] += 1
                recalled_experiences.append(exp)
        
        return recalled_experiences
    
    def recall_by_context(self, context: str, max_results: int = 5) -> list:
        """Recall experiences by context
        
        Args:
            context: Context to recall
            max_results: Maximum number of results
            
        Returns:
            List of recalled experiences
        """
        if context not in self.context_index:
            return []
        
        experience_ids = self.context_index[context]
        recalled_experiences = []
        
        for exp_id in experience_ids[-max_results:]:
            if exp_id in self.emotional_experiences:
                exp = self.emotional_experiences[exp_id]
                exp['access_count'] += 1
                recalled_experiences.append(exp)
        
        return recalled_experiences
    
    def find_similar_emotions(self, query_field: NDAnalogField, 
                             max_results: int = 5) -> list:
        """Find similar emotional experiences using resonance
        
        Args:
            query_field: Field to find similar emotions for
            max_results: Maximum number of results
            
        Returns:
            List of similar experiences with resonance scores
        """
        similar_experiences = []
        
        for exp_id, exp_data in self.emotional_experiences.items():
            # Create field from stored experience
            stored_field = type('Field', (), {
                'activation': exp_data['field'],
                'shape': exp_data['field_shape']
            })()
            
            # Calculate resonance
            resonance_strength = self.resonator.get_resonance_strength(query_field, stored_field)
            
            if resonance_strength >= self.resonator.threshold:
                similar_experiences.append({
                    'experience': exp_data,
                    'resonance_strength': resonance_strength,
                    'similarity': resonance_strength
                })
        
        # Sort by resonance strength
        similar_experiences.sort(key=lambda x: x['resonance_strength'], reverse=True)
        
        # Update access counts
        for result in similar_experiences[:max_results]:
            result['experience']['access_count'] += 1
            result['experience']['resonance_strength'] = result['resonance_strength']
        
        return similar_experiences[:max_results]
    
    def recall_with_resonance(self, query_field: NDAnalogField, 
                             emotion_filter: str = None, max_results: int = 5) -> list:
        """Recall experiences using resonance with optional emotion filter
        
        Args:
            query_field: Field to find resonant experiences for
            emotion_filter: Optional emotion type to filter by
            max_results: Maximum number of results
            
        Returns:
            List of resonant experiences
        """
        candidates = []
        
        # Get candidate experiences
        if emotion_filter and emotion_filter in self.emotion_index:
            candidate_ids = self.emotion_index[emotion_filter]
        else:
            candidate_ids = list(self.emotional_experiences.keys())
        
        # Calculate resonance for each candidate
        for exp_id in candidate_ids:
            if exp_id not in self.emotional_experiences:
                continue
                
            exp_data = self.emotional_experiences[exp_id]
            stored_field = type('Field', (), {
                'activation': exp_data['field'],
                'shape': exp_data['field_shape']
            })()
            
            resonance_strength = self.resonator.get_resonance_strength(query_field, stored_field)
            
            if resonance_strength >= self.resonator.threshold:
                candidates.append({
                    'experience': exp_data,
                    'resonance_strength': resonance_strength
                })
        
        # Sort by resonance strength
        candidates.sort(key=lambda x: x['resonance_strength'], reverse=True)
        
        # Update access counts and record recall
        results = candidates[:max_results]
        for result in results:
            result['experience']['access_count'] += 1
            result['experience']['resonance_strength'] = result['resonance_strength']
        
        # Record recall operation
        self._record_recall(query_field, results, emotion_filter)
        
        return results
    
    def _record_recall(self, query_field: NDAnalogField, results: list, emotion_filter: str):
        """Record recall operation"""
        recall_record = {
            'recall_id': f"recall_{len(self.recall_history) + 1}",
            'query_energy': np.sum(np.abs(query_field.activation)),
            'results_count': len(results),
            'emotion_filter': emotion_filter,
            'timestamp': self.total_processed
        }
        
        self.recall_history.append(recall_record)
    
    def get_emotional_pattern(self, emotion: str) -> np.ndarray:
        """Get typical emotional pattern for emotion type
        
        Args:
            emotion: Emotion type
            
        Returns:
            Typical pattern for that emotion
        """
        if emotion not in self.emotion_index:
            return None
        
        experience_ids = self.emotion_index[emotion]
        if not experience_ids:
            return None
        
        # Average patterns for this emotion
        patterns = []
        for exp_id in experience_ids:
            if exp_id in self.emotional_experiences:
                exp = self.emotional_experiences[exp_id]
                patterns.append(exp['field'])
        
        if not patterns:
            return None
        
        # Find common shape
        common_shape = patterns[0].shape
        for pattern in patterns[1:]:
            if pattern.shape != common_shape:
                # Resize to common shape if needed
                common_shape = tuple(min(s1, s2) for s1, s2 in zip(common_shape, pattern.shape))
        
        # Average patterns
        averaged_pattern = np.zeros(common_shape)
        for pattern in patterns:
            if pattern.shape == common_shape:
                averaged_pattern += pattern
            else:
                # Simple resizing
                resized = pattern[:common_shape[0], :common_shape[1]] if len(common_shape) == 2 else pattern
                averaged_pattern += resized
        
        return averaged_pattern / len(patterns)
    
    def get_emotion_statistics(self) -> dict:
        """Get statistics about stored emotions
        
        Returns:
            Dictionary with emotion statistics
        """
        if not self.emotional_experiences:
            return {
                'total_experiences': 0,
                'emotion_types': 0,
                'context_types': 0,
                'average_intensity': 0.0
            }
        
        emotions = [exp['emotion'] for exp in self.emotional_experiences.values()]
        contexts = [exp['context'] for exp in self.emotional_experiences.values() if exp['context']]
        intensities = [exp['intensity'] for exp in self.emotional_experiences.values()]
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return {
            'total_experiences': len(self.emotional_experiences),
            'emotion_types': len(set(emotions)),
            'context_types': len(set(contexts)),
            'average_intensity': np.mean(intensities),
            'emotion_distribution': emotion_counts,
            'most_common_emotion': max(emotion_counts, key=emotion_counts.get) if emotion_counts else None,
            'total_recalls': len(self.recall_history)
        }
    
    def get_experience_by_id(self, experience_id: str) -> dict:
        """Get experience by ID
        
        Args:
            experience_id: ID of experience to retrieve
            
        Returns:
            Experience data or None if not found
        """
        return self.emotional_experiences.get(experience_id)
    
    def update_experience_intensity(self, experience_id: str, new_intensity: float):
        """Update intensity of stored experience
        
        Args:
            experience_id: ID of experience to update
            new_intensity: New intensity value
        """
        if experience_id in self.emotional_experiences:
            self.emotional_experiences[experience_id]['intensity'] = new_intensity
    
    def get_emotional_timeline(self, emotion: str = None, window_size: int = 10) -> list:
        """Get emotional timeline
        
        Args:
            emotion: Optional emotion filter
            window_size: Number of recent experiences to include
            
        Returns:
            List of recent emotional experiences
        """
        if emotion and emotion in self.emotion_index:
            experience_ids = self.emotion_index[emotion]
        else:
            experience_ids = list(self.emotional_experiences.keys())
        
        # Sort by timestamp
        experiences = [self.emotional_experiences[exp_id] for exp_id in experience_ids 
                      if exp_id in self.emotional_experiences]
        experiences.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return experiences[:window_size]
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'experiences': [{
                'id': exp['id'],
                'emotion': exp['emotion'],
                'intensity': exp['intensity'],
                'access_count': exp['access_count']
            } for exp in self.emotional_experiences.values()],
            'emotion_types': list(self.emotion_index.keys()),
            'context_types': list(self.context_index.keys()),
            'experience_count': self.experience_count,
            'total_processed': self.total_processed,
            'recall_count': len(self.recall_history)
        }
    
    def reset(self):
        """Reset emotional memory"""
        self.emotional_experiences.clear()
        self.emotion_index.clear()
        self.context_index.clear()
        self.experience_count = 0
        self.total_processed = 0
        self.recall_history.clear()
        self.memory_trace.clear()
        self.resonator.resonance_history.clear()
    
    def __repr__(self):
        return f"EmotionalMemory(experiences={len(self.emotional_experiences)}, emotions={len(self.emotion_index)})"

