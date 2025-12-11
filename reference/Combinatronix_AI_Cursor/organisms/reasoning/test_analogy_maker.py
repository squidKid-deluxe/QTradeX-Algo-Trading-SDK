# ============================================================================
# AnalogyMaker Test Suite
# ============================================================================

"""
Test suite for AnalogyMaker organism
"""

import numpy as np
import unittest

try:
    from .analogy_maker import AnalogyMaker, AnalogyMapping, DomainPattern
except ImportError:
    from combinatronix.organisms.reasoning.analogy_maker import AnalogyMaker, AnalogyMapping, DomainPattern


class TestAnalogyMaker(unittest.TestCase):
    """Test AnalogyMaker organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maker = AnalogyMaker({'enable_visualization': False})
    
    def test_initialization(self):
        """Test maker initialization"""
        self.assertIsNotNone(self.maker.atoms)
        self.assertIsNotNone(self.maker.molecules)
        self.assertIsNotNone(self.maker.fields)
        self.assertEqual(len(self.maker.atoms), 4)
        self.assertEqual(len(self.maker.molecules), 2)
        self.assertEqual(self.maker.state['tick_counter'], 0)
        self.assertEqual(len(self.maker.state['analogy_mappings']), 0)
    
    def test_inject_source_domain(self):
        """Test source domain injection"""
        patterns = {
            "circle": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "square": np.array([[1, 0], [0, 1]], dtype=np.float32)
        }
        
        self.maker.inject_source_domain("geometry", patterns)
        
        self.assertIn("geometry", self.maker.state['source_patterns'])
        self.assertEqual(len(self.maker.state['source_patterns']["geometry"]), 2)
        self.assertIn("circle", self.maker.state['source_patterns']["geometry"])
        self.assertIn("square", self.maker.state['source_patterns']["geometry"])
    
    def test_inject_target_domain(self):
        """Test target domain injection"""
        patterns = {
            "sun": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "moon": np.array([[1, 0], [0, 1]], dtype=np.float32)
        }
        
        self.maker.inject_target_domain("nature", patterns)
        
        self.assertIn("nature", self.maker.state['target_patterns'])
        self.assertEqual(len(self.maker.state['target_patterns']["nature"]), 2)
        self.assertIn("sun", self.maker.state['target_patterns']["nature"])
        self.assertIn("moon", self.maker.state['target_patterns']["nature"])
    
    def test_resize_pattern(self):
        """Test pattern resizing"""
        # Test pattern smaller than field size
        small_pattern = np.array([[1, 0], [0, 1]], dtype=np.float32)
        resized = self.maker._resize_pattern(small_pattern)
        self.assertEqual(resized.shape, self.maker.config['field_size'])
        
        # Test pattern larger than field size
        large_pattern = np.ones((20, 20), dtype=np.float32)
        resized = self.maker._resize_pattern(large_pattern)
        self.assertEqual(resized.shape, self.maker.config['field_size'])
        
        # Test pattern same size as field
        same_pattern = np.ones(self.maker.config['field_size'], dtype=np.float32)
        resized = self.maker._resize_pattern(same_pattern)
        self.assertEqual(resized.shape, self.maker.config['field_size'])
        np.testing.assert_array_equal(resized, same_pattern)
    
    def test_extract_pattern_properties(self):
        """Test pattern property extraction"""
        pattern = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=np.float32)
        
        properties = self.maker._extract_pattern_properties(pattern)
        
        self.assertIn('energy', properties)
        self.assertIn('complexity', properties)
        self.assertIn('symmetry', properties)
        self.assertIn('sparsity', properties)
        self.assertIn('center_of_mass', properties)
        
        self.assertEqual(properties['energy'], 5.0)
        self.assertGreater(properties['complexity'], 0)
        self.assertGreaterEqual(properties['symmetry'], 0)
        self.assertLessEqual(properties['symmetry'], 1)
        self.assertGreater(properties['sparsity'], 0)
        self.assertLessEqual(properties['sparsity'], 1)
        self.assertIsInstance(properties['center_of_mass'], tuple)
        self.assertEqual(len(properties['center_of_mass']), 2)
    
    def test_calculate_symmetry(self):
        """Test symmetry calculation"""
        # Test symmetric pattern
        symmetric = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=np.float32)
        symmetry = self.maker._calculate_symmetry(symmetric)
        self.assertGreater(symmetry, 0.5)  # Should be fairly symmetric
        
        # Test asymmetric pattern
        asymmetric = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        symmetry = self.maker._calculate_symmetry(asymmetric)
        self.assertLess(symmetry, 0.5)  # Should be less symmetric
    
    def test_calculate_center_of_mass(self):
        """Test center of mass calculation"""
        # Test pattern with center of mass at (1, 1)
        pattern = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        com = self.maker._calculate_center_of_mass(pattern)
        self.assertAlmostEqual(com[0], 1.0, places=5)
        self.assertAlmostEqual(com[1], 1.0, places=5)
        
        # Test empty pattern
        empty = np.zeros((3, 3), dtype=np.float32)
        com = self.maker._calculate_center_of_mass(empty)
        self.assertEqual(com, (0.0, 0.0))
    
    def test_find_analogy(self):
        """Test basic analogy finding"""
        # Set up domains
        source_patterns = {
            "circle": np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32)
        }
        
        target_patterns = {
            "sun": np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32)
        }
        
        self.maker.inject_source_domain("geometry", source_patterns)
        self.maker.inject_target_domain("nature", target_patterns)
        
        # Find analogy
        mapping, similarity = self.maker.find_analogy("geometry", "nature", "circle", "sun")
        
        self.assertIsInstance(mapping, AnalogyMapping)
        self.assertEqual(mapping.source_concept, "circle")
        self.assertEqual(mapping.target_concept, "sun")
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
        self.assertGreaterEqual(mapping.confidence, 0)
        self.assertLessEqual(mapping.confidence, 1)
    
    def test_find_analogy_auto_target(self):
        """Test analogy finding with automatic target selection"""
        # Set up domains
        source_patterns = {
            "circle": np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32)
        }
        
        target_patterns = {
            "sun": np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32),
            "moon": np.array([
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1]
            ], dtype=np.float32)
        }
        
        self.maker.inject_source_domain("geometry", source_patterns)
        self.maker.inject_target_domain("nature", target_patterns)
        
        # Find analogy without specifying target
        mapping, similarity = self.maker.find_analogy("geometry", "nature", "circle")
        
        self.assertIsInstance(mapping, AnalogyMapping)
        self.assertEqual(mapping.source_concept, "circle")
        self.assertIn(mapping.target_concept, ["sun", "moon"])
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
    
    def test_find_best_target_match(self):
        """Test best target match finding"""
        # Set up source pattern
        source_pattern = DomainPattern(
            name="circle",
            pattern=NDAnalogField((4, 4), activation=np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32)),
            domain="geometry",
            properties={},
            relationships=[]
        )
        
        # Set up target patterns
        target_patterns = {
            "sun": DomainPattern(
                name="sun",
                pattern=NDAnalogField((4, 4), activation=np.array([
                    [0, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 1, 1, 0]
                ], dtype=np.float32)),
                domain="nature",
                properties={},
                relationships=[]
            ),
            "moon": DomainPattern(
                name="moon",
                pattern=NDAnalogField((4, 4), activation=np.array([
                    [1, 0, 0, 1],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [1, 0, 0, 1]
                ], dtype=np.float32)),
                domain="nature",
                properties={},
                relationships=[]
            )
        }
        
        self.maker.state['target_patterns']["nature"] = target_patterns
        
        # Find best match
        best_concept, best_pattern = self.maker._find_best_target_match(source_pattern, "nature")
        
        self.assertIn(best_concept, ["sun", "moon"])
        self.assertIsInstance(best_pattern, DomainPattern)
        self.assertEqual(best_pattern.name, best_concept)
    
    def test_compute_structural_similarity(self):
        """Test structural similarity computation"""
        # Create test patterns
        source_pattern = DomainPattern(
            name="circle",
            pattern=NDAnalogField((4, 4), activation=np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32)),
            domain="geometry",
            properties={},
            relationships=[]
        )
        
        target_pattern = DomainPattern(
            name="sun",
            pattern=NDAnalogField((4, 4), activation=np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32)),
            domain="nature",
            properties={},
            relationships=[]
        )
        
        # Compute similarity
        similarity = self.maker._compute_structural_similarity(source_pattern, target_pattern)
        
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
        self.assertGreater(similarity, 0.5)  # Should be high for identical patterns
    
    def test_determine_mapping_type(self):
        """Test mapping type determination"""
        # Create test patterns with different properties
        source_pattern = DomainPattern(
            name="circle",
            pattern=NDAnalogField((4, 4)),
            domain="geometry",
            properties={
                'symmetry': 0.9,
                'energy': 10.0,
                'complexity': 0.5,
                'sparsity': 0.8,
                'center_of_mass': (2.0, 2.0)
            },
            relationships=[]
        )
        
        target_pattern = DomainPattern(
            name="sun",
            pattern=NDAnalogField((4, 4)),
            domain="nature",
            properties={
                'symmetry': 0.85,  # Similar symmetry
                'energy': 10.5,    # Similar energy
                'complexity': 0.6, # Similar complexity
                'sparsity': 0.75,  # Similar sparsity
                'center_of_mass': (2.1, 2.1)  # Similar center of mass
            },
            relationships=[]
        )
        
        # Determine mapping type
        mapping_type = self.maker._determine_mapping_type(source_pattern, target_pattern)
        
        self.assertIn(mapping_type, ["structural", "functional", "relational", "causal"])
    
    def test_find_analogies_batch(self):
        """Test batch analogy finding"""
        # Set up domains
        source_patterns = {
            "circle": np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32),
            "square": np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ], dtype=np.float32)
        }
        
        target_patterns = {
            "sun": np.array([
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0]
            ], dtype=np.float32),
            "house": np.array([
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ], dtype=np.float32)
        }
        
        self.maker.inject_source_domain("geometry", source_patterns)
        self.maker.inject_target_domain("nature", target_patterns)
        
        # Find analogies in batch
        analogies = self.maker.find_analogies_batch("geometry", "nature", max_analogies=2)
        
        self.assertIsInstance(analogies, list)
        self.assertEqual(len(analogies), 2)
        
        for mapping, similarity in analogies:
            self.assertIsInstance(mapping, AnalogyMapping)
            self.assertGreaterEqual(similarity, 0)
            self.assertLessEqual(similarity, 1)
    
    def test_get_analogy_summary(self):
        """Test analogy summary generation"""
        # Add some state
        self.maker.state['tick_counter'] = 5
        self.maker.state['source_patterns'] = {"geometry": {}}
        self.maker.state['target_patterns'] = {"nature": {}}
        self.maker.state['analogy_mappings'] = {"test": None}
        
        summary = self.maker.get_analogy_summary()
        
        self.assertIn('tick', summary)
        self.assertIn('source_domains', summary)
        self.assertIn('target_domains', summary)
        self.assertIn('total_mappings', summary)
        self.assertIn('mapping_history_length', summary)
        self.assertIn('total_mappings_created', summary)
        self.assertIn('field_energies', summary)
        
        self.assertEqual(summary['tick'], 5)
        self.assertEqual(summary['source_domains'], 1)
        self.assertEqual(summary['target_domains'], 1)
        self.assertEqual(summary['total_mappings'], 1)
    
    def test_get_analogy_mappings(self):
        """Test analogy mappings retrieval"""
        # Add some mappings
        mapping1 = AnalogyMapping(
            source_concept="circle",
            target_concept="sun",
            similarity_score=0.8,
            mapping_strength=0.7,
            field_location=(2, 2),
            mapping_type="structural",
            confidence=0.9
        )
        
        mapping2 = AnalogyMapping(
            source_concept="square",
            target_concept="house",
            similarity_score=0.6,
            mapping_strength=0.5,
            field_location=(3, 3),
            mapping_type="functional",
            confidence=0.7
        )
        
        self.maker.state['analogy_mappings'] = {
            "mapping1": mapping1,
            "mapping2": mapping2
        }
        
        mappings = self.maker.get_analogy_mappings()
        
        self.assertEqual(len(mappings), 2)
        self.assertIn(mapping1, mappings)
        self.assertIn(mapping2, mappings)
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.maker.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test maker reset"""
        # Add some state
        self.maker.state['tick_counter'] = 5
        self.maker.state['source_patterns'] = {"geometry": {}}
        self.maker.state['target_patterns'] = {"nature": {}}
        self.maker.state['analogy_mappings'] = {"test": None}
        
        # Reset
        self.maker.reset()
        
        # Check that state is reset
        self.assertEqual(self.maker.state['tick_counter'], 0)
        self.assertEqual(len(self.maker.state['source_patterns']), 0)
        self.assertEqual(len(self.maker.state['target_patterns']), 0)
        self.assertEqual(len(self.maker.state['analogy_mappings']), 0)
    
    def test_analogy_mapping_creation(self):
        """Test AnalogyMapping dataclass"""
        mapping = AnalogyMapping(
            source_concept="circle",
            target_concept="sun",
            similarity_score=0.8,
            mapping_strength=0.7,
            field_location=(2, 2),
            mapping_type="structural",
            confidence=0.9
        )
        
        self.assertEqual(mapping.source_concept, "circle")
        self.assertEqual(mapping.target_concept, "sun")
        self.assertEqual(mapping.similarity_score, 0.8)
        self.assertEqual(mapping.mapping_strength, 0.7)
        self.assertEqual(mapping.field_location, (2, 2))
        self.assertEqual(mapping.mapping_type, "structural")
        self.assertEqual(mapping.confidence, 0.9)
    
    def test_domain_pattern_creation(self):
        """Test DomainPattern dataclass"""
        pattern = DomainPattern(
            name="circle",
            pattern=NDAnalogField((4, 4)),
            domain="geometry",
            properties={'energy': 10.0},
            relationships=['round', 'symmetric']
        )
        
        self.assertEqual(pattern.name, "circle")
        self.assertEqual(pattern.domain, "geometry")
        self.assertEqual(pattern.properties['energy'], 10.0)
        self.assertEqual(pattern.relationships, ['round', 'symmetric'])
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'field_size': (8, 8), 'similarity_threshold': 0.3},
            {'field_size': (12, 12), 'mapping_strength_threshold': 0.2},
            {'max_mappings': 20, 'analogy_depth': 2}
        ]
        
        for config in configs:
            maker = AnalogyMaker(config)
            
            # Should work with different configs
            source_patterns = {"test": np.array([[1, 0], [0, 1]], dtype=np.float32)}
            maker.inject_source_domain("test", source_patterns)
            
            target_patterns = {"test": np.array([[1, 0], [0, 1]], dtype=np.float32)}
            maker.inject_target_domain("test", target_patterns)
            
            mapping, similarity = maker.find_analogy("test", "test", "test", "test")
            self.assertIsInstance(mapping, AnalogyMapping)
            self.assertGreaterEqual(similarity, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

