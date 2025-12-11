# ============================================================================
# SemanticNetwork Test Suite
# ============================================================================

"""
Test suite for SemanticNetwork organism
"""

import numpy as np
import unittest

try:
    from .semantic_network import SemanticNetwork, ConceptNode, Relation, ActivationPath
except ImportError:
    from combinatronix.organisms.reasoning.semantic_network import SemanticNetwork, ConceptNode, Relation, ActivationPath


class TestSemanticNetwork(unittest.TestCase):
    """Test SemanticNetwork organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.network = SemanticNetwork({'enable_visualization': False})
    
    def test_initialization(self):
        """Test network initialization"""
        self.assertIsNotNone(self.network.atoms)
        self.assertIsNotNone(self.network.molecules)
        self.assertIsNotNone(self.network.fields)
        self.assertEqual(len(self.network.atoms), 6)
        self.assertEqual(len(self.network.molecules), 3)
        self.assertEqual(len(self.network.state['concepts']), 0)
        self.assertEqual(len(self.network.state['relations']), 0)
    
    def test_add_concept(self):
        """Test concept addition"""
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        concept = self.network.add_concept("test_concept", pattern)
        
        self.assertIsInstance(concept, ConceptNode)
        self.assertEqual(concept.name, "test_concept")
        self.assertEqual(concept.activation, 0.0)
        self.assertEqual(concept.strength, 1.0)
        
        # Check that concept was stored
        self.assertIn("test_concept", self.network.state['concepts'])
        self.assertIn("test_concept", self.network.state['concept_fields'])
    
    def test_add_concept_with_metadata(self):
        """Test concept addition with metadata"""
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        metadata = {"category": "test", "importance": 0.8}
        
        concept = self.network.add_concept("test_concept", pattern, metadata)
        
        self.assertEqual(concept.metadata, metadata)
    
    def test_add_duplicate_concept(self):
        """Test adding duplicate concept"""
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        
        self.network.add_concept("test_concept", pattern)
        
        with self.assertRaises(ValueError):
            self.network.add_concept("test_concept", pattern)
    
    def test_link_concepts(self):
        """Test concept linking"""
        # Add concepts first
        pattern_a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        pattern_b = np.array([[0, 1], [1, 0]], dtype=np.float32)
        
        self.network.add_concept("concept_a", pattern_a)
        self.network.add_concept("concept_b", pattern_b)
        
        # Link concepts
        relation = self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        
        self.assertIsInstance(relation, Relation)
        self.assertEqual(relation.concept_a, "concept_a")
        self.assertEqual(relation.concept_b, "concept_b")
        self.assertEqual(relation.relation_type, "is-a")
        self.assertEqual(relation.strength, 0.8)
        
        # Check that relation was stored
        self.assertEqual(len(self.network.state['relations']), 1)
    
    def test_link_nonexistent_concepts(self):
        """Test linking nonexistent concepts"""
        with self.assertRaises(ValueError):
            self.network.link_concepts("nonexistent_a", "nonexistent_b", "is-a")
    
    def test_link_invalid_relation_type(self):
        """Test linking with invalid relation type"""
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.network.add_concept("concept_a", pattern)
        self.network.add_concept("concept_b", pattern)
        
        with self.assertRaises(ValueError):
            self.network.link_concepts("concept_a", "concept_b", "invalid_relation")
    
    def test_activate_concept(self):
        """Test concept activation"""
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.network.add_concept("test_concept", pattern)
        
        # Activate concept
        result = self.network.activate_concept("test_concept", 0.8)
        
        self.assertTrue(result)
        
        # Check that concept was activated
        concept = self.network.state['concepts']["test_concept"]
        self.assertGreater(concept.activation, 0)
        self.assertEqual(concept.last_activated, self.network.state['tick_counter'])
    
    def test_activate_nonexistent_concept(self):
        """Test activating nonexistent concept"""
        result = self.network.activate_concept("nonexistent", 0.8)
        self.assertFalse(result)
    
    def test_spreading_activation(self):
        """Test spreading activation"""
        # Add concepts and relations
        patterns = {
            "concept_a": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "concept_b": np.array([[0, 1], [1, 0]], dtype=np.float32),
            "concept_c": np.array([[1, 1], [1, 1]], dtype=np.float32)
        }
        
        for name, pattern in patterns.items():
            self.network.add_concept(name, pattern)
        
        # Create relations
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        self.network.link_concepts("concept_b", "concept_c", "similar-to", 0.6)
        
        # Test spreading activation
        activation_map = self.network.spreading_activation(["concept_a"], steps=3)
        
        self.assertIsInstance(activation_map, dict)
        self.assertIn("concept_a", activation_map)
        self.assertIn("concept_b", activation_map)
        self.assertIn("concept_c", activation_map)
        
        # Check that initial concept is activated
        self.assertGreater(activation_map["concept_a"], 0)
    
    def test_find_path(self):
        """Test path finding"""
        # Add concepts and relations
        patterns = {
            "concept_a": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "concept_b": np.array([[0, 1], [1, 0]], dtype=np.float32),
            "concept_c": np.array([[1, 1], [1, 1]], dtype=np.float32)
        }
        
        for name, pattern in patterns.items():
            self.network.add_concept(name, pattern)
        
        # Create relations
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        self.network.link_concepts("concept_b", "concept_c", "similar-to", 0.6)
        
        # Find path
        path = self.network.find_path("concept_a", "concept_c")
        
        self.assertIsInstance(path, ActivationPath)
        self.assertEqual(path.path[0], "concept_a")
        self.assertEqual(path.path[-1], "concept_c")
        self.assertGreater(path.total_strength, 0)
        self.assertGreater(path.path_length, 0)
        self.assertEqual(len(path.relation_types), path.path_length)
    
    def test_find_path_nonexistent(self):
        """Test path finding with nonexistent concepts"""
        path = self.network.find_path("nonexistent_a", "nonexistent_b")
        self.assertIsNone(path)
    
    def test_find_path_no_connection(self):
        """Test path finding with no connection"""
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.network.add_concept("concept_a", pattern)
        self.network.add_concept("concept_b", pattern)
        
        path = self.network.find_path("concept_a", "concept_b")
        self.assertIsNone(path)
    
    def test_infer_new_relation(self):
        """Test relation inference"""
        # Add concepts
        pattern_a = np.array([[1, 0], [0, 1]], dtype=np.float32)
        pattern_b = np.array([[1, 0], [0, 1]], dtype=np.float32)  # Similar pattern
        
        self.network.add_concept("concept_a", pattern_a)
        self.network.add_concept("concept_b", pattern_b)
        
        # Infer relation
        relation = self.network.infer_new_relation("concept_a", "concept_b")
        
        if relation:  # May or may not infer depending on similarity
            self.assertIsInstance(relation, Relation)
            self.assertEqual(relation.concept_a, "concept_a")
            self.assertEqual(relation.concept_b, "concept_b")
            self.assertIn(relation.relation_type, self.network.config['relation_types'])
            self.assertGreaterEqual(relation.strength, 0)
            self.assertLessEqual(relation.strength, 1)
    
    def test_infer_relation_nonexistent_concepts(self):
        """Test relation inference with nonexistent concepts"""
        relation = self.network.infer_new_relation("nonexistent_a", "nonexistent_b")
        self.assertIsNone(relation)
    
    def test_consolidate_network(self):
        """Test network consolidation"""
        # Add concepts and relations
        patterns = {
            "concept_a": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "concept_b": np.array([[0, 1], [1, 0]], dtype=np.float32)
        }
        
        for name, pattern in patterns.items():
            self.network.add_concept(name, pattern)
        
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        
        # Consolidate network
        self.network.consolidate_network()
        
        # Check that consolidation was performed
        self.assertGreater(len(self.network.state['relations']), 0)
    
    def test_get_network_summary(self):
        """Test network summary generation"""
        # Add some concepts and relations
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.network.add_concept("concept_a", pattern)
        self.network.add_concept("concept_b", pattern)
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        
        summary = self.network.get_network_summary()
        
        self.assertIn('concepts', summary)
        self.assertIn('relations', summary)
        self.assertIn('total_activations', summary)
        self.assertIn('total_inferences', summary)
        self.assertIn('current_tick', summary)
        self.assertIn('relation_types', summary)
        self.assertIn('field_energies', summary)
        self.assertIn('concept_activations', summary)
        
        self.assertEqual(summary['concepts'], 2)
        self.assertEqual(summary['relations'], 1)
    
    def test_get_concepts(self):
        """Test concepts retrieval"""
        # Add some concepts
        patterns = {
            "concept_a": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "concept_b": np.array([[0, 1], [1, 0]], dtype=np.float32)
        }
        
        for name, pattern in patterns.items():
            self.network.add_concept(name, pattern)
        
        concepts = self.network.get_concepts()
        
        self.assertEqual(len(concepts), 2)
        self.assertIsInstance(concepts[0], ConceptNode)
    
    def test_get_relations(self):
        """Test relations retrieval"""
        # Add concepts and relations
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.network.add_concept("concept_a", pattern)
        self.network.add_concept("concept_b", pattern)
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        
        relations = self.network.get_relations()
        
        self.assertEqual(len(relations), 1)
        self.assertIsInstance(relations[0], Relation)
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.network.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test network reset"""
        # Add some concepts and relations
        pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.network.add_concept("concept_a", pattern)
        self.network.add_concept("concept_b", pattern)
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        
        # Reset
        self.network.reset()
        
        # Check that state is reset
        self.assertEqual(len(self.network.state['concepts']), 0)
        self.assertEqual(len(self.network.state['relations']), 0)
        self.assertEqual(self.network.state['tick_counter'], 0)
        self.assertEqual(self.network.state['total_activations'], 0)
        self.assertEqual(self.network.state['total_inferences'], 0)
    
    def test_resize_pattern(self):
        """Test pattern resizing"""
        # Test pattern smaller than field size
        small_pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        resized = self.network._resize_pattern(small_pattern)
        self.assertEqual(resized.shape, self.network.config['field_shape'])
        
        # Test pattern larger than field size
        large_pattern = np.ones((20, 20), dtype=np.float32)
        resized = self.network._resize_pattern(large_pattern)
        self.assertEqual(resized.shape, self.network.config['field_shape'])
        
        # Test pattern same size as field
        same_pattern = np.ones(self.network.config['field_shape'], dtype=np.float32)
        resized = self.network._resize_pattern(same_pattern)
        self.assertEqual(resized.shape, self.network.config['field_shape'])
        np.testing.assert_array_equal(resized, same_pattern)
    
    def test_relation_to_transfer_rate(self):
        """Test relation type to transfer rate conversion"""
        transfer_rates = {
            'is-a': 0.9,
            'part-of': 0.8,
            'causes': 0.7,
            'similar-to': 0.6,
            'opposite-of': 0.3,
            'related-to': 0.5,
            'contains': 0.7,
            'made-of': 0.8,
            'used-for': 0.6,
            'located-in': 0.5
        }
        
        for relation_type, expected_rate in transfer_rates.items():
            actual_rate = self.network._relation_to_transfer_rate(relation_type)
            self.assertEqual(actual_rate, expected_rate)
        
        # Test unknown relation type
        unknown_rate = self.network._relation_to_transfer_rate("unknown")
        self.assertEqual(unknown_rate, 0.5)
    
    def test_get_related_concepts(self):
        """Test getting related concepts"""
        # Add concepts and relations
        patterns = {
            "concept_a": np.array([[1, 0], [0, 1]], dtype=np.float32),
            "concept_b": np.array([[0, 1], [1, 0]], dtype=np.float32),
            "concept_c": np.array([[1, 1], [1, 1]], dtype=np.float32)
        }
        
        for name, pattern in patterns.items():
            self.network.add_concept(name, pattern)
        
        self.network.link_concepts("concept_a", "concept_b", "is-a", 0.8)
        self.network.link_concepts("concept_a", "concept_c", "similar-to", 0.6)
        
        # Get related concepts
        related = self.network._get_related_concepts("concept_a")
        
        self.assertEqual(len(related), 2)
        related_names = [name for name, _ in related]
        self.assertIn("concept_b", related_names)
        self.assertIn("concept_c", related_names)
    
    def test_compute_concept_similarity_molecular(self):
        """Test molecular concept similarity computation"""
        field_a = NDAnalogField((16, 16))
        field_a.activation = np.random.random((16, 16)) * 0.8
        
        field_b = NDAnalogField((16, 16))
        field_b.activation = np.random.random((16, 16)) * 0.8
        
        similarity = self.network._compute_concept_similarity_molecular(field_a, field_b)
        
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
    
    def test_infer_relation_type(self):
        """Test relation type inference"""
        # Test high similarity
        relation_type = self.network._infer_relation_type(0.9, NDAnalogField((16, 16)), NDAnalogField((16, 16)))
        self.assertEqual(relation_type, "similar-to")
        
        # Test low similarity
        relation_type = self.network._infer_relation_type(0.3, NDAnalogField((16, 16)), NDAnalogField((16, 16)))
        self.assertIsNone(relation_type)
    
    def test_concept_node_creation(self):
        """Test ConceptNode dataclass"""
        field_pattern = NDAnalogField((16, 16))
        metadata = {"category": "test"}
        
        concept = ConceptNode(
            name="test_concept",
            field_pattern=field_pattern,
            activation=0.8,
            metadata=metadata,
            strength=0.9,
            last_activated=5
        )
        
        self.assertEqual(concept.name, "test_concept")
        self.assertEqual(concept.field_pattern, field_pattern)
        self.assertEqual(concept.activation, 0.8)
        self.assertEqual(concept.metadata, metadata)
        self.assertEqual(concept.strength, 0.9)
        self.assertEqual(concept.last_activated, 5)
    
    def test_relation_creation(self):
        """Test Relation dataclass"""
        relation = Relation(
            concept_a="concept_a",
            concept_b="concept_b",
            relation_type="is-a",
            strength=0.8,
            transfer_rate=0.9,
            created_at=10,
            access_count=3,
            confidence=0.7
        )
        
        self.assertEqual(relation.concept_a, "concept_a")
        self.assertEqual(relation.concept_b, "concept_b")
        self.assertEqual(relation.relation_type, "is-a")
        self.assertEqual(relation.strength, 0.8)
        self.assertEqual(relation.transfer_rate, 0.9)
        self.assertEqual(relation.created_at, 10)
        self.assertEqual(relation.access_count, 3)
        self.assertEqual(relation.confidence, 0.7)
    
    def test_activation_path_creation(self):
        """Test ActivationPath dataclass"""
        path = ActivationPath(
            path=["concept_a", "concept_b", "concept_c"],
            total_strength=1.5,
            path_length=2,
            relation_types=["is-a", "similar-to"]
        )
        
        self.assertEqual(path.path, ["concept_a", "concept_b", "concept_c"])
        self.assertEqual(path.total_strength, 1.5)
        self.assertEqual(path.path_length, 2)
        self.assertEqual(path.relation_types, ["is-a", "similar-to"])
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'field_shape': (8, 8), 'max_concepts': 50},
            {'field_shape': (32, 32), 'spreading_steps': 3},
            {'inference_threshold': 0.5, 'path_search_depth': 5}
        ]
        
        for config in configs:
            network = SemanticNetwork(config)
            
            # Should work with different configs
            pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
            concept = network.add_concept("test", pattern)
            self.assertIsInstance(concept, ConceptNode)
            
            relation = network.link_concepts("test", "test", "is-a", 0.8)
            self.assertIsInstance(relation, Relation)


if __name__ == '__main__':
    unittest.main(verbosity=2)

