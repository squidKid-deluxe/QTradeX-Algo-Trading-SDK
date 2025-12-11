# ============================================================================
# EpisodicMemory - Event Sequence Storage & Recall Using Molecular Architecture
# ============================================================================

"""
EpisodicMemory - Stores temporal sequences of events with context binding

Composition:
- Atoms: MemoryTrace, Echo, Binder, Anticipator, Comparator, Resonator
- Molecules: AssociativeMemory, WorkingMemory, PatternRecognizer
- Fields: timeline_field, context_field, replay_field, consolidation_field

This organism stores temporal sequences of events with context binding,
like human episodic memory - remembers "what happened when where".
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Generator
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import time

try:
    from ...core import NDAnalogField
    from ...atoms.temporal import MemoryTraceAtom, AnticipatorAtom
    from ...atoms.pattern_primitives import EchoAtom
    from ...atoms.multi_field import BinderAtom, ComparatorAtom, ResonatorAtom
    from ...molecules.memory import AssociativeMemoryMolecule, WorkingMemoryMolecule
    from ...molecules.perception import PatternRecognizerMolecule
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.atoms.temporal import MemoryTraceAtom, AnticipatorAtom
    from combinatronix.atoms.pattern_primitives import EchoAtom
    from combinatronix.atoms.multi_field import BinderAtom, ComparatorAtom, ResonatorAtom
    from combinatronix.molecules.memory import AssociativeMemoryMolecule, WorkingMemoryMolecule
    from combinatronix.molecules.perception import PatternRecognizerMolecule


@dataclass
class Episode:
    """Represents a single episodic memory event"""
    field_state: NDAnalogField
    timestamp: int
    context: Dict[str, Any]
    index: int
    access_count: int = 0
    last_accessed: int = 0
    strength: float = 1.0
    consolidation_level: float = 0.0


@dataclass
class RecallResult:
    """Result of episode recall"""
    episode: Episode
    similarity_score: float
    confidence: float
    context_match: float
    temporal_distance: int


class EpisodicMemory:
    """Event sequence storage and recall using molecular architecture"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the episodic memory
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'field_shape': (32, 32),
            'max_episodes': 1000,
            'consolidation_threshold': 0.7,
            'similarity_threshold': 0.3,
            'enable_visualization': True,
            'memory_decay_rate': 0.95,
            'consolidation_rate': 0.1,
            'context_weight': 0.3,
            'temporal_weight': 0.2,
            'similarity_weight': 0.5
        }
        
        if config:
            self.config.update(config)
        
        # Initialize atoms
        self._initialize_atoms()
        
        # Initialize molecules
        self._initialize_molecules()
        
        # Initialize fields
        self._initialize_fields()
        
        # State tracking
        self.state = {
            'episodes': [],
            'current_timestamp': 0,
            'next_index': 0,
            'consolidation_history': [],
            'recall_history': [],
            'total_episodes_stored': 0,
            'total_recalls': 0
        }
        
        print(f"🧠 EpisodicMemory initialized ({self.config['field_shape'][0]}×{self.config['field_shape'][1]})")
    
    def _initialize_atoms(self):
        """Initialize atomic operations"""
        self.atoms = {
            'memory_trace': MemoryTraceAtom(
                accumulation_rate=0.15,
                decay_rate=self.config['memory_decay_rate']
            ),
            'echo': EchoAtom(
                decay_rate=0.85,
                depth=10
            ),
            'binder': BinderAtom(
                binding_strength=0.8,
                temporal_window=5
            ),
            'anticipator': AnticipatorAtom(
                history_depth=5,
                prediction_strength=0.7
            ),
            'comparator': ComparatorAtom(
                metric='cosine',
                normalize=True
            ),
            'resonator': ResonatorAtom(
                amplification=1.5,
                threshold=0.5
            )
        }
    
    def _initialize_molecules(self):
        """Initialize molecular operations"""
        self.molecules = {
            'associative_memory': AssociativeMemoryMolecule(
                amplification=1.2,
                resonance_threshold=0.5
            ),
            'working_memory': WorkingMemoryMolecule(
                capacity=50,
                decay_rate=0.95
            ),
            'pattern_recognizer': PatternRecognizerMolecule(
                amplification=1.5,
                resonance_threshold=0.5
            )
        }
    
    def _initialize_fields(self):
        """Initialize field structures"""
        self.fields = {
            'timeline_field': NDAnalogField(self.config['field_shape']),
            'context_field': NDAnalogField(self.config['field_shape']),
            'replay_field': NDAnalogField(self.config['field_shape']),
            'consolidation_field': NDAnalogField(self.config['field_shape']),
            'similarity_field': NDAnalogField(self.config['field_shape']),
            'anticipation_field': NDAnalogField(self.config['field_shape'])
        }
    
    def encode_episode(self, event_field: NDAnalogField, context: Dict[str, Any] = None) -> Episode:
        """
        Store new episode using molecular operations
        
        Args:
            event_field: Field state representing the event
            context: Context information (location, emotion, etc.)
            
        Returns:
            Created Episode object
        """
        if context is None:
            context = {}
        
        # Create episode snapshot
        episode = Episode(
            field_state=event_field.copy(),
            timestamp=self.state['current_timestamp'],
            context=context.copy(),
            index=self.state['next_index'],
            access_count=0,
            last_accessed=self.state['current_timestamp'],
            strength=1.0,
            consolidation_level=0.0
        )
        
        # Store episode
        self.state['episodes'].append(episode)
        self.state['next_index'] += 1
        self.state['current_timestamp'] += 1
        self.state['total_episodes_stored'] += 1
        
        # Bind to timeline using molecular operations
        self._bind_to_timeline_molecular(episode)
        
        # Accumulate into memory trace
        self.atoms['memory_trace'].apply(event_field)
        
        # Update context field
        self._update_context_field(context)
        
        # Store in working memory
        self.molecules['working_memory'].store(episode)
        
        print(f"📝 Encoded episode {episode.index} at timestamp {episode.timestamp}")
        
        return episode
    
    def _bind_to_timeline_molecular(self, episode: Episode):
        """Bind episode to timeline using molecular operations"""
        # Use binder atom to create temporal connections
        binding_strength = self.atoms['binder'].bind(
            episode.field_state,
            self.fields['timeline_field'],
            strength=episode.strength
        )
        
        # Update timeline field with temporal information
        temporal_position = episode.timestamp / max(1, self.state['current_timestamp'])
        timeline_update = episode.field_state.activation * temporal_position * binding_strength
        
        self.fields['timeline_field'].activation += timeline_update * 0.1
        self.fields['timeline_field'].activation = np.clip(self.fields['timeline_field'].activation, 0, 1)
    
    def _update_context_field(self, context: Dict[str, Any]):
        """Update context field with context information"""
        if not context:
            return
        
        # Create context pattern based on context information
        context_pattern = np.zeros(self.config['field_shape'])
        
        # Map context to field positions
        if 'location' in context:
            loc = context['location']
            if isinstance(loc, (tuple, list)) and len(loc) >= 2:
                x, y = int(loc[0] * self.config['field_shape'][1]), int(loc[1] * self.config['field_shape'][0])
                if 0 <= x < self.config['field_shape'][1] and 0 <= y < self.config['field_shape'][0]:
                    context_pattern[y, x] = 1.0
        
        if 'emotion' in context:
            emotion_val = context['emotion']
            if isinstance(emotion_val, (int, float)):
                # Map emotion to field intensity
                intensity = (emotion_val + 1) / 2  # Normalize to [0, 1]
                context_pattern += intensity * 0.3
        
        # Update context field
        self.fields['context_field'].activation += context_pattern
        self.fields['context_field'].activation = np.clip(self.fields['context_field'].activation, 0, 1)
    
    def recall_episode(self, cue_field: NDAnalogField, k: int = 3, 
                      context_cue: Dict[str, Any] = None) -> List[RecallResult]:
        """
        Retrieve k most similar episodes using molecular operations
        
        Args:
            cue_field: Field cue for recall
            k: Number of episodes to return
            context_cue: Context information for recall
            
        Returns:
            List of RecallResult objects
        """
        self.state['total_recalls'] += 1
        
        # Compute similarities using molecular operations
        similarities = []
        
        for episode in self.state['episodes']:
            # Compute field similarity using comparator atom
            field_similarity = self._compute_field_similarity_molecular(cue_field, episode.field_state)
            
            # Compute context similarity
            context_similarity = self._compute_context_similarity(context_cue, episode.context)
            
            # Compute temporal distance
            temporal_distance = abs(self.state['current_timestamp'] - episode.timestamp)
            temporal_similarity = 1.0 / (1.0 + temporal_distance * 0.1)
            
            # Weighted combination
            overall_similarity = (
                self.config['similarity_weight'] * field_similarity +
                self.config['context_weight'] * context_similarity +
                self.config['temporal_weight'] * temporal_similarity
            )
            
            # Calculate confidence
            confidence = min(1.0, overall_similarity * episode.strength)
            
            # Create recall result
            recall_result = RecallResult(
                episode=episode,
                similarity_score=overall_similarity,
                confidence=confidence,
                context_match=context_similarity,
                temporal_distance=temporal_distance
            )
            
            similarities.append(recall_result)
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        top_k = similarities[:k]
        
        # Update access counts
        for result in top_k:
            result.episode.access_count += 1
            result.episode.last_accessed = self.state['current_timestamp']
        
        # Record recall
        self.state['recall_history'].append({
            'timestamp': self.state['current_timestamp'],
            'cue_energy': np.sum(cue_field.activation),
            'context_cue': context_cue,
            'results_count': len(top_k),
            'top_similarity': top_k[0].similarity_score if top_k else 0.0
        })
        
        print(f"🔍 Recalled {len(top_k)} episodes (top similarity: {top_k[0].similarity_score:.3f})")
        
        return top_k
    
    def _compute_field_similarity_molecular(self, field1: NDAnalogField, field2: NDAnalogField) -> float:
        """Compute field similarity using molecular operations"""
        # Use comparator atom for field comparison
        comparison = self.atoms['comparator'].apply(field1, field2)
        
        # Use resonator atom for enhanced similarity
        resonance = self.atoms['resonator'].apply(field1, field2)
        
        # Use pattern recognizer molecule for pattern matching
        recognition = self.molecules['pattern_recognizer'].recognize(field1, field2)
        
        # Combine similarities
        similarity = (
            np.mean(comparison.activation) * 0.4 +
            np.mean(resonance.activation) * 0.3 +
            np.mean(recognition.activation) * 0.3
        )
        
        return similarity
    
    def _compute_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute context similarity"""
        if not context1 or not context2:
            return 0.0
        
        similarities = []
        
        for key in set(context1.keys()) & set(context2.keys()):
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numeric similarity
                similarity = 1.0 - abs(val1 - val2) / (abs(val1) + abs(val2) + 1e-8)
            elif isinstance(val1, (tuple, list)) and isinstance(val2, (tuple, list)):
                # Tuple/list similarity
                if len(val1) == len(val2):
                    similarity = 1.0 - sum(abs(a - b) for a, b in zip(val1, val2)) / len(val1)
                else:
                    similarity = 0.0
            else:
                # String similarity
                similarity = 1.0 if val1 == val2 else 0.0
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def replay_sequence(self, start_index: int, length: int = 5) -> Generator[NDAnalogField, None, None]:
        """
        Replay episode sequence using molecular operations
        
        Args:
            start_index: Starting episode index
            length: Number of episodes to replay
            
        Yields:
            NDAnalogField objects representing the replay
        """
        sequence = self.state['episodes'][start_index:start_index + length]
        
        if not sequence:
            return
        
        # Initialize replay field
        self.fields['replay_field'].activation.fill(0)
        
        for i, episode in enumerate(sequence):
            # Create replay field
            replay_field = episode.field_state.copy()
            
            # Apply echo atom for decaying replay
            self.atoms['echo'].apply(replay_field)
            
            # Apply memory trace for consolidation
            self.atoms['memory_trace'].apply(replay_field)
            
            # Update replay field
            self.fields['replay_field'].activation = replay_field.activation.copy()
            
            yield replay_field.copy()
    
    def find_similar_episodes(self, query_episode: Episode, k: int = 5) -> List[RecallResult]:
        """
        Find episodes similar to query episode using molecular operations
        
        Args:
            query_episode: Episode to find similarities for
            k: Number of similar episodes to return
            
        Returns:
            List of RecallResult objects
        """
        # Use associative memory molecule for pattern completion
        associations = self.molecules['associative_memory'].associate(query_episode.field_state)
        
        # Find similar episodes
        similarities = []
        
        for episode in self.state['episodes']:
            if episode.index == query_episode.index:
                continue  # Skip self
            
            # Compute similarity using molecular operations
            field_similarity = self._compute_field_similarity_molecular(query_episode.field_state, episode.field_state)
            context_similarity = self._compute_context_similarity(query_episode.context, episode.context)
            
            # Temporal similarity
            temporal_distance = abs(query_episode.timestamp - episode.timestamp)
            temporal_similarity = 1.0 / (1.0 + temporal_distance * 0.1)
            
            # Overall similarity
            overall_similarity = (
                field_similarity * 0.5 +
                context_similarity * 0.3 +
                temporal_similarity * 0.2
            )
            
            if overall_similarity > self.config['similarity_threshold']:
                recall_result = RecallResult(
                    episode=episode,
                    similarity_score=overall_similarity,
                    confidence=overall_similarity * episode.strength,
                    context_match=context_similarity,
                    temporal_distance=temporal_distance
                )
                similarities.append(recall_result)
        
        # Sort and return top k
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        return similarities[:k]
    
    def consolidate(self):
        """
        Sleep-like consolidation using molecular operations
        
        Strengthens important episodes and merges similar ones
        """
        print("🔄 Starting memory consolidation...")
        
        # Apply memory trace consolidation
        self.atoms['memory_trace'].consolidate()
        
        # Strengthen frequently accessed episodes
        for episode in self.state['episodes']:
            if episode.access_count > 0:
                # Strengthen based on access count
                strength_increase = min(0.1, episode.access_count * 0.01)
                episode.strength = min(1.0, episode.strength + strength_increase)
                
                # Increase consolidation level
                episode.consolidation_level = min(1.0, episode.consolidation_level + 0.1)
        
        # Merge similar episodes
        self._merge_similar_episodes()
        
        # Update consolidation field
        self._update_consolidation_field()
        
        # Record consolidation
        self.state['consolidation_history'].append({
            'timestamp': self.state['current_timestamp'],
            'episodes_count': len(self.state['episodes']),
            'consolidated_episodes': sum(1 for ep in self.state['episodes'] if ep.consolidation_level > 0.5)
        })
        
        print(f"✅ Consolidation complete: {len(self.state['episodes'])} episodes")
    
    def _merge_similar_episodes(self):
        """Merge similar episodes to reduce redundancy"""
        episodes_to_remove = []
        
        for i, episode1 in enumerate(self.state['episodes']):
            if episode1 in episodes_to_remove:
                continue
            
            for j, episode2 in enumerate(self.state['episodes'][i+1:], i+1):
                if episode2 in episodes_to_remove:
                    continue
                
                # Check if episodes are similar enough to merge
                similarity = self._compute_field_similarity_molecular(episode1.field_state, episode2.field_state)
                
                if similarity > self.config['consolidation_threshold']:
                    # Merge episodes (keep the stronger one)
                    if episode1.strength >= episode2.strength:
                        # Merge episode2 into episode1
                        episode1.strength = min(1.0, episode1.strength + episode2.strength * 0.5)
                        episode1.access_count += episode2.access_count
                        episodes_to_remove.append(episode2)
                    else:
                        # Merge episode1 into episode2
                        episode2.strength = min(1.0, episode2.strength + episode1.strength * 0.5)
                        episode2.access_count += episode1.access_count
                        episodes_to_remove.append(episode1)
                        break
        
        # Remove merged episodes
        for episode in episodes_to_remove:
            if episode in self.state['episodes']:
                self.state['episodes'].remove(episode)
    
    def _update_consolidation_field(self):
        """Update consolidation field with consolidation information"""
        self.fields['consolidation_field'].activation.fill(0)
        
        for episode in self.state['episodes']:
            if episode.consolidation_level > 0.5:
                # Add consolidated episode to field
                self.fields['consolidation_field'].activation += episode.field_state.activation * episode.consolidation_level * 0.1
        
        self.fields['consolidation_field'].activation = np.clip(self.fields['consolidation_field'].activation, 0, 1)
    
    def anticipate_next(self, current_episode: Episode) -> NDAnalogField:
        """
        Anticipate next episode using molecular operations
        
        Args:
            current_episode: Current episode to base anticipation on
            
        Returns:
            Anticipated next episode field
        """
        # Use anticipator atom to predict next episode
        anticipation = self.atoms['anticipator'].anticipate(current_episode.field_state)
        
        # Update anticipation field
        self.fields['anticipation_field'].activation = anticipation.activation.copy()
        
        return anticipation
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory state"""
        episodes = self.state['episodes']
        
        return {
            "total_episodes": len(episodes),
            "current_timestamp": self.state['current_timestamp'],
            "total_stored": self.state['total_episodes_stored'],
            "total_recalls": self.state['total_recalls'],
            "consolidated_episodes": sum(1 for ep in episodes if ep.consolidation_level > 0.5),
            "average_strength": np.mean([ep.strength for ep in episodes]) if episodes else 0.0,
            "average_access_count": np.mean([ep.access_count for ep in episodes]) if episodes else 0.0,
            "field_energies": {
                name: np.sum(field.activation) for name, field in self.fields.items()
            }
        }
    
    def get_episodes(self) -> List[Episode]:
        """Get all episodes"""
        return self.state['episodes'].copy()
    
    def get_state(self) -> Dict:
        """Get current internal state"""
        return {
            'config': self.config.copy(),
            'state': self.state.copy(),
            'field_shapes': {name: field.shape for name, field in self.fields.items()},
            'atom_states': {name: atom.__repr__() for name, atom in self.atoms.items()},
            'molecule_states': {name: molecule.__repr__() for name, molecule in self.molecules.items()}
        }
    
    def reset(self):
        """Reset the episodic memory"""
        self.state = {
            'episodes': [],
            'current_timestamp': 0,
            'next_index': 0,
            'consolidation_history': [],
            'recall_history': [],
            'total_episodes_stored': 0,
            'total_recalls': 0
        }
        
        # Reset fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Reset molecules
        for molecule in self.molecules.values():
            if hasattr(molecule, 'reset'):
                molecule.reset()
    
    def visualize_memory_state(self, save_path: Optional[str] = None):
        """Visualize current memory state"""
        if not self.config['enable_visualization']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"EpisodicMemory - Molecular Memory State", fontsize=16)
        
        # Timeline field
        im1 = axes[0, 0].imshow(self.fields['timeline_field'].activation, cmap='Blues')
        axes[0, 0].set_title("Timeline Field")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Context field
        im2 = axes[0, 1].imshow(self.fields['context_field'].activation, cmap='Greens')
        axes[0, 1].set_title("Context Field")
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Consolidation field
        im3 = axes[0, 2].imshow(self.fields['consolidation_field'].activation, cmap='Reds')
        axes[0, 2].set_title("Consolidation Field")
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Episode strengths over time
        if self.state['episodes']:
            timestamps = [ep.timestamp for ep in self.state['episodes']]
            strengths = [ep.strength for ep in self.state['episodes']]
            axes[1, 0].scatter(timestamps, strengths, alpha=0.6)
            axes[1, 0].set_title("Episode Strengths Over Time")
            axes[1, 0].set_xlabel("Timestamp")
            axes[1, 0].set_ylabel("Strength")
        
        # Access counts
        if self.state['episodes']:
            access_counts = [ep.access_count for ep in self.state['episodes']]
            axes[1, 1].hist(access_counts, bins=min(20, len(access_counts)))
            axes[1, 1].set_title("Episode Access Counts")
            axes[1, 1].set_xlabel("Access Count")
            axes[1, 1].set_ylabel("Frequency")
        
        # Memory summary
        summary = self.get_memory_summary()
        axes[1, 2].text(0.1, 0.8, f"Total Episodes: {summary['total_episodes']}", fontsize=12)
        axes[1, 2].text(0.1, 0.7, f"Consolidated: {summary['consolidated_episodes']}", fontsize=12)
        axes[1, 2].text(0.1, 0.6, f"Avg Strength: {summary['average_strength']:.3f}", fontsize=12)
        axes[1, 2].text(0.1, 0.5, f"Total Recalls: {summary['total_recalls']}", fontsize=12)
        axes[1, 2].text(0.1, 0.4, f"Current Time: {summary['current_timestamp']}", fontsize=12)
        axes[1, 2].set_title("Memory Summary")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def __repr__(self):
        return f"EpisodicMemory(episodes={len(self.state['episodes'])}, " \
               f"timestamp={self.state['current_timestamp']}, " \
               f"recalls={self.state['total_recalls']})"


# === Demo Functions ===

def demo_basic_episodic_memory():
    """Demo basic episodic memory operations"""
    print("=== Basic Episodic Memory Demo ===")
    
    memory = EpisodicMemory({'field_shape': (16, 16), 'enable_visualization': False})
    
    # Create some test episodes
    episodes = []
    
    for i in range(5):
        # Create random event field
        event_field = NDAnalogField((16, 16))
        event_field.activation = np.random.random((16, 16)) * 0.8
        
        # Create context
        context = {
            'location': (i * 0.2, i * 0.3),
            'emotion': np.random.uniform(-1, 1),
            'activity': f'event_{i}'
        }
        
        # Encode episode
        episode = memory.encode_episode(event_field, context)
        episodes.append(episode)
        
        print(f"Encoded episode {i+1}: {context['activity']}")
    
    # Test recall
    print("\nTesting recall...")
    cue_field = episodes[2].field_state.copy()
    cue_field.activation *= 0.8  # Slightly different cue
    
    results = memory.recall_episode(cue_field, k=3)
    
    print(f"Recall results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. Episode {result.episode.index}: similarity={result.similarity_score:.3f}, "
              f"confidence={result.confidence:.3f}")
    
    # Test sequence replay
    print("\nTesting sequence replay...")
    replay_sequence = list(memory.replay_sequence(0, length=3))
    print(f"Replayed {len(replay_sequence)} episodes")
    
    # Test consolidation
    print("\nTesting consolidation...")
    memory.consolidate()
    
    # Show summary
    summary = memory.get_memory_summary()
    print(f"\nMemory summary:")
    print(f"  Total episodes: {summary['total_episodes']}")
    print(f"  Consolidated: {summary['consolidated_episodes']}")
    print(f"  Average strength: {summary['average_strength']:.3f}")
    
    return memory


def demo_complex_episodic_memory():
    """Demo complex episodic memory with visualization"""
    print("\n=== Complex Episodic Memory Demo ===")
    
    memory = EpisodicMemory({'field_shape': (20, 20), 'enable_visualization': True})
    
    # Create a story sequence
    story_events = [
        {"name": "wake_up", "field": np.eye(20) * 0.5, "context": {"location": (0.1, 0.1), "emotion": 0.8}},
        {"name": "eat_breakfast", "field": np.ones((20, 20)) * 0.3, "context": {"location": (0.3, 0.2), "emotion": 0.6}},
        {"name": "go_to_work", "field": np.random.random((20, 20)) * 0.7, "context": {"location": (0.8, 0.9), "emotion": 0.2}},
        {"name": "meeting", "field": np.zeros((20, 20)), "context": {"location": (0.5, 0.5), "emotion": -0.1}},
        {"name": "lunch", "field": np.ones((20, 20)) * 0.4, "context": {"location": (0.4, 0.3), "emotion": 0.5}},
        {"name": "work_project", "field": np.random.random((20, 20)) * 0.6, "context": {"location": (0.7, 0.8), "emotion": 0.3}},
        {"name": "go_home", "field": np.random.random((20, 20)) * 0.5, "context": {"location": (0.2, 0.1), "emotion": 0.7}},
        {"name": "dinner", "field": np.ones((20, 20)) * 0.5, "context": {"location": (0.3, 0.2), "emotion": 0.8}},
        {"name": "relax", "field": np.eye(20) * 0.3, "context": {"location": (0.1, 0.1), "emotion": 0.9}},
        {"name": "sleep", "field": np.zeros((20, 20)), "context": {"location": (0.1, 0.1), "emotion": 0.1}}
    ]
    
    # Encode story episodes
    print("Encoding story episodes...")
    for event in story_events:
        event_field = NDAnalogField((20, 20), activation=event["field"])
        episode = memory.encode_episode(event_field, event["context"])
        print(f"  Encoded: {event['name']}")
    
    # Test various recall scenarios
    print("\nTesting recall scenarios...")
    
    # Recall by field similarity
    cue_field = NDAnalogField((20, 20), activation=story_events[1]["field"] * 0.8)
    results = memory.recall_episode(cue_field, k=3)
    print(f"Field-based recall: {len(results)} results")
    
    # Recall by context
    context_cue = {"location": (0.3, 0.2), "emotion": 0.6}
    results = memory.recall_episode(cue_field, k=3, context_cue=context_cue)
    print(f"Context-based recall: {len(results)} results")
    
    # Find similar episodes
    query_episode = memory.get_episodes()[2]  # go_to_work
    similar = memory.find_similar_episodes(query_episode, k=3)
    print(f"Similar episodes to 'go_to_work': {len(similar)} found")
    
    # Test anticipation
    current_episode = memory.get_episodes()[5]  # work_project
    anticipated = memory.anticipate_next(current_episode)
    print(f"Anticipated next episode energy: {np.sum(anticipated.activation):.3f}")
    
    # Test consolidation
    print("\nTesting consolidation...")
    memory.consolidate()
    
    # Show visualization
    memory.visualize_memory_state()
    
    return memory


# === Main Demo ===

if __name__ == '__main__':
    print("🧠 EPISODIC MEMORY - Event Sequence Storage & Recall 🧠")
    print("Stores temporal sequences with context binding using molecular operations!")
    print("Like human episodic memory - remembers 'what happened when where'\n")
    
    # Run demos
    basic_memory = demo_basic_episodic_memory()
    complex_memory = demo_complex_episodic_memory()
    
    # System capabilities summary
    print("\n" + "="*60)
    print("🎯 EPISODIC MEMORY CAPABILITIES DEMONSTRATED")
    print("="*60)
    
    all_memories = [basic_memory, complex_memory]
    total_episodes = sum(len(memory.get_episodes()) for memory in all_memories)
    total_recalls = sum(memory.state['total_recalls'] for memory in all_memories)
    
    print(f"✅ Event sequence storage and encoding")
    print(f"✅ Context-aware episode recall")
    print(f"✅ Temporal sequence replay")
    print(f"✅ Similar episode finding")
    print(f"✅ Memory consolidation")
    print(f"✅ Episode anticipation")
    print(f"✅ Molecular pattern binding")
    print(f"✅ Multi-dimensional context support")
    
    print(f"\n📊 DEMO STATISTICS:")
    print(f"Total episodes stored: {total_episodes}")
    print(f"Total recalls performed: {total_recalls}")
    print(f"Average episodes per memory: {total_episodes / len(all_memories):.1f}")
    
    print(f"\n💡 KEY INNOVATIONS:")
    print(f"• Molecular event encoding and storage")
    print(f"• Context-aware recall using multiple similarity measures")
    print(f"• Temporal sequence replay with echo decay")
    print(f"• Memory consolidation and episode merging")
    print(f"• Episode anticipation using molecular operations")
    
    print(f"\n🌟 This demonstrates true episodic memory!")
    print("The system stores and recalls events with context and temporal information.")
    print("No training data, no neural networks - pure molecular intelligence!")
    
    print("\n🚀 EpisodicMemory Demo Complete! 🚀")