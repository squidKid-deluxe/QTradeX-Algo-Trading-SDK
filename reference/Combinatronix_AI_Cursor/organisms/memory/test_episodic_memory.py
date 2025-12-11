# ============================================================================
# EpisodicMemory Test Suite
# ============================================================================

"""
Test suite for EpisodicMemory organism
"""

import numpy as np
import unittest

try:
    from .episodic_memory import EpisodicMemory, Episode, RecallResult
except ImportError:
    from combinatronix.organisms.memory.episodic_memory import EpisodicMemory, Episode, RecallResult


class TestEpisodicMemory(unittest.TestCase):
    """Test EpisodicMemory organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory = EpisodicMemory({'enable_visualization': False})
    
    def test_initialization(self):
        """Test memory initialization"""
        self.assertIsNotNone(self.memory.atoms)
        self.assertIsNotNone(self.memory.molecules)
        self.assertIsNotNone(self.memory.fields)
        self.assertEqual(len(self.memory.atoms), 6)
        self.assertEqual(len(self.memory.molecules), 3)
        self.assertEqual(self.memory.state['current_timestamp'], 0)
        self.assertEqual(len(self.memory.state['episodes']), 0)
    
    def test_encode_episode(self):
        """Test episode encoding"""
        # Create test event field
        event_field = NDAnalogField((16, 16))
        event_field.activation = np.random.random((16, 16)) * 0.8
        
        # Create context
        context = {
            'location': (0.5, 0.5),
            'emotion': 0.7,
            'activity': 'test_event'
        }
        
        # Encode episode
        episode = self.memory.encode_episode(event_field, context)
        
        self.assertIsInstance(episode, Episode)
        self.assertEqual(episode.timestamp, 0)
        self.assertEqual(episode.index, 0)
        self.assertEqual(episode.context, context)
        self.assertEqual(episode.access_count, 0)
        self.assertEqual(episode.strength, 1.0)
        
        # Check that episode was stored
        self.assertEqual(len(self.memory.state['episodes']), 1)
        self.assertEqual(self.memory.state['current_timestamp'], 1)
        self.assertEqual(self.memory.state['next_index'], 1)
    
    def test_encode_episode_no_context(self):
        """Test episode encoding without context"""
        event_field = NDAnalogField((16, 16))
        event_field.activation = np.random.random((16, 16)) * 0.8
        
        episode = self.memory.encode_episode(event_field)
        
        self.assertIsInstance(episode, Episode)
        self.assertEqual(episode.context, {})
    
    def test_recall_episode(self):
        """Test episode recall"""
        # Encode some episodes
        episodes = []
        for i in range(3):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            context = {'activity': f'event_{i}'}
            episode = self.memory.encode_episode(event_field, context)
            episodes.append(episode)
        
        # Create cue field
        cue_field = episodes[1].field_state.copy()
        cue_field.activation *= 0.9  # Slightly different
        
        # Recall episodes
        results = self.memory.recall_episode(cue_field, k=2)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
        
        for result in results:
            self.assertIsInstance(result, RecallResult)
            self.assertIsInstance(result.episode, Episode)
            self.assertGreaterEqual(result.similarity_score, 0)
            self.assertLessEqual(result.similarity_score, 1)
            self.assertGreaterEqual(result.confidence, 0)
            self.assertLessEqual(result.confidence, 1)
            self.assertGreaterEqual(result.temporal_distance, 0)
    
    def test_recall_episode_with_context(self):
        """Test episode recall with context cue"""
        # Encode episodes with different contexts
        episodes = []
        contexts = [
            {'location': (0.1, 0.1), 'emotion': 0.8},
            {'location': (0.5, 0.5), 'emotion': 0.2},
            {'location': (0.9, 0.9), 'emotion': -0.5}
        ]
        
        for i, context in enumerate(contexts):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            episode = self.memory.encode_episode(event_field, context)
            episodes.append(episode)
        
        # Create cue with context
        cue_field = episodes[1].field_state.copy()
        context_cue = {'location': (0.5, 0.5), 'emotion': 0.3}
        
        # Recall with context
        results = self.memory.recall_episode(cue_field, k=2, context_cue=context_cue)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)
    
    def test_replay_sequence(self):
        """Test sequence replay"""
        # Encode some episodes
        episodes = []
        for i in range(5):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            episode = self.memory.encode_episode(event_field)
            episodes.append(episode)
        
        # Replay sequence
        replay_sequence = list(self.memory.replay_sequence(0, length=3))
        
        self.assertIsInstance(replay_sequence, list)
        self.assertEqual(len(replay_sequence), 3)
        
        for replay_field in replay_sequence:
            self.assertIsInstance(replay_field, NDAnalogField)
            self.assertEqual(replay_field.shape, self.memory.config['field_shape'])
    
    def test_find_similar_episodes(self):
        """Test finding similar episodes"""
        # Encode episodes
        episodes = []
        for i in range(4):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            episode = self.memory.encode_episode(event_field)
            episodes.append(episode)
        
        # Find similar episodes
        query_episode = episodes[1]
        similar = self.memory.find_similar_episodes(query_episode, k=2)
        
        self.assertIsInstance(similar, list)
        self.assertLessEqual(len(similar), 2)
        
        for result in similar:
            self.assertIsInstance(result, RecallResult)
            self.assertNotEqual(result.episode.index, query_episode.index)
    
    def test_consolidate(self):
        """Test memory consolidation"""
        # Encode some episodes
        episodes = []
        for i in range(5):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            episode = self.memory.encode_episode(event_field)
            episodes.append(episode)
        
        # Access some episodes to increase their strength
        for i in range(3):
            episodes[i].access_count = 5
            episodes[i].strength = 0.8
        
        # Consolidate
        initial_count = len(self.memory.state['episodes'])
        self.memory.consolidate()
        
        # Check that consolidation was recorded
        self.assertGreater(len(self.memory.state['consolidation_history']), 0)
        
        # Check that some episodes may have been merged
        final_count = len(self.memory.state['episodes'])
        self.assertLessEqual(final_count, initial_count)
    
    def test_anticipate_next(self):
        """Test episode anticipation"""
        # Encode some episodes
        episodes = []
        for i in range(3):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            episode = self.memory.encode_episode(event_field)
            episodes.append(episode)
        
        # Anticipate next episode
        current_episode = episodes[1]
        anticipated = self.memory.anticipate_next(current_episode)
        
        self.assertIsInstance(anticipated, NDAnalogField)
        self.assertEqual(anticipated.shape, self.memory.config['field_shape'])
    
    def test_compute_field_similarity_molecular(self):
        """Test molecular field similarity computation"""
        field1 = NDAnalogField((16, 16))
        field1.activation = np.random.random((16, 16)) * 0.8
        
        field2 = NDAnalogField((16, 16))
        field2.activation = np.random.random((16, 16)) * 0.8
        
        similarity = self.memory._compute_field_similarity_molecular(field1, field2)
        
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
    
    def test_compute_context_similarity(self):
        """Test context similarity computation"""
        context1 = {'location': (0.5, 0.5), 'emotion': 0.7}
        context2 = {'location': (0.6, 0.6), 'emotion': 0.8}
        
        similarity = self.memory._compute_context_similarity(context1, context2)
        
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
        self.assertGreater(similarity, 0.5)  # Should be fairly similar
    
    def test_compute_context_similarity_empty(self):
        """Test context similarity with empty contexts"""
        similarity = self.memory._compute_context_similarity({}, {})
        self.assertEqual(similarity, 0.0)
        
        similarity = self.memory._compute_context_similarity({'a': 1}, {})
        self.assertEqual(similarity, 0.0)
    
    def test_get_memory_summary(self):
        """Test memory summary generation"""
        # Add some episodes
        for i in range(3):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            self.memory.encode_episode(event_field)
        
        summary = self.memory.get_memory_summary()
        
        self.assertIn('total_episodes', summary)
        self.assertIn('current_timestamp', summary)
        self.assertIn('total_stored', summary)
        self.assertIn('total_recalls', summary)
        self.assertIn('consolidated_episodes', summary)
        self.assertIn('average_strength', summary)
        self.assertIn('average_access_count', summary)
        self.assertIn('field_energies', summary)
        
        self.assertEqual(summary['total_episodes'], 3)
        self.assertEqual(summary['current_timestamp'], 3)
        self.assertEqual(summary['total_stored'], 3)
    
    def test_get_episodes(self):
        """Test episodes retrieval"""
        # Add some episodes
        episodes = []
        for i in range(3):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            episode = self.memory.encode_episode(event_field)
            episodes.append(episode)
        
        retrieved_episodes = self.memory.get_episodes()
        
        self.assertEqual(len(retrieved_episodes), 3)
        self.assertIsInstance(retrieved_episodes, list)
        self.assertIsInstance(retrieved_episodes[0], Episode)
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.memory.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test memory reset"""
        # Add some episodes
        for i in range(3):
            event_field = NDAnalogField((16, 16))
            event_field.activation = np.random.random((16, 16)) * 0.8
            self.memory.encode_episode(event_field)
        
        # Reset
        self.memory.reset()
        
        # Check that state is reset
        self.assertEqual(len(self.memory.state['episodes']), 0)
        self.assertEqual(self.memory.state['current_timestamp'], 0)
        self.assertEqual(self.memory.state['next_index'], 0)
        self.assertEqual(len(self.memory.state['consolidation_history']), 0)
        self.assertEqual(len(self.memory.state['recall_history']), 0)
    
    def test_episode_creation(self):
        """Test Episode dataclass"""
        field_state = NDAnalogField((16, 16))
        context = {'location': (0.5, 0.5), 'emotion': 0.7}
        
        episode = Episode(
            field_state=field_state,
            timestamp=10,
            context=context,
            index=5,
            access_count=3,
            last_accessed=8,
            strength=0.8,
            consolidation_level=0.6
        )
        
        self.assertEqual(episode.timestamp, 10)
        self.assertEqual(episode.context, context)
        self.assertEqual(episode.index, 5)
        self.assertEqual(episode.access_count, 3)
        self.assertEqual(episode.last_accessed, 8)
        self.assertEqual(episode.strength, 0.8)
        self.assertEqual(episode.consolidation_level, 0.6)
    
    def test_recall_result_creation(self):
        """Test RecallResult dataclass"""
        episode = Episode(
            field_state=NDAnalogField((16, 16)),
            timestamp=5,
            context={},
            index=1
        )
        
        result = RecallResult(
            episode=episode,
            similarity_score=0.8,
            confidence=0.7,
            context_match=0.6,
            temporal_distance=2
        )
        
        self.assertEqual(result.episode, episode)
        self.assertEqual(result.similarity_score, 0.8)
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.context_match, 0.6)
        self.assertEqual(result.temporal_distance, 2)
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'field_shape': (8, 8), 'max_episodes': 50},
            {'field_shape': (32, 32), 'similarity_threshold': 0.5},
            {'consolidation_threshold': 0.8, 'memory_decay_rate': 0.9}
        ]
        
        for config in configs:
            memory = EpisodicMemory(config)
            
            # Should work with different configs
            event_field = NDAnalogField(memory.config['field_shape'])
            event_field.activation = np.random.random(memory.config['field_shape']) * 0.8
            
            episode = memory.encode_episode(event_field)
            self.assertIsInstance(episode, Episode)
            
            results = memory.recall_episode(event_field, k=1)
            self.assertIsInstance(results, list)
    
    def test_timeline_binding(self):
        """Test timeline binding"""
        # Encode episode
        event_field = NDAnalogField((16, 16))
        event_field.activation = np.random.random((16, 16)) * 0.8
        episode = self.memory.encode_episode(event_field)
        
        # Check that timeline field was updated
        timeline_energy = np.sum(self.memory.fields['timeline_field'].activation)
        self.assertGreater(timeline_energy, 0)
    
    def test_context_field_update(self):
        """Test context field update"""
        # Encode episode with context
        event_field = NDAnalogField((16, 16))
        event_field.activation = np.random.random((16, 16)) * 0.8
        context = {'location': (0.5, 0.5), 'emotion': 0.7}
        
        self.memory.encode_episode(event_field, context)
        
        # Check that context field was updated
        context_energy = np.sum(self.memory.fields['context_field'].activation)
        self.assertGreater(context_energy, 0)
    
    def test_access_count_update(self):
        """Test access count update during recall"""
        # Encode episode
        event_field = NDAnalogField((16, 16))
        event_field.activation = np.random.random((16, 16)) * 0.8
        episode = self.memory.encode_episode(event_field)
        
        initial_access_count = episode.access_count
        
        # Recall episode
        self.memory.recall_episode(event_field, k=1)
        
        # Check that access count was updated
        self.assertGreater(episode.access_count, initial_access_count)


if __name__ == '__main__':
    unittest.main(verbosity=2)

