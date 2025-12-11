# ============================================================================
# Reasoning Molecules Tests
# ============================================================================

"""
Comprehensive tests for reasoning molecules:
- Analogizer
- ContradictionResolver
- GapFiller
- PatternCompleter
"""

import numpy as np
import unittest
from unittest.mock import Mock

try:
    from ...core import NDAnalogField
    from .analogizer import Analogizer
    from .contradiction_resolver import ContradictionResolver
    from .gap_filler import GapFiller
    from .pattern_completer import PatternCompleter
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.molecules.reasoning.analogizer import Analogizer
    from combinatronix.molecules.reasoning.contradiction_resolver import ContradictionResolver
    from combinatronix.molecules.reasoning.gap_filler import GapFiller
    from combinatronix.molecules.reasoning.pattern_completer import PatternCompleter


class TestAnalogizer(unittest.TestCase):
    """Test Analogizer molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analogizer = Analogizer(similarity_threshold=0.7, bridge_strength=0.8)
        self.pattern_a = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.pattern_b = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
    
    def test_initialization(self):
        """Test analogizer initialization"""
        self.assertEqual(self.analogizer.similarity_threshold, 0.7)
        self.assertEqual(self.analogizer.bridge.transfer_rate, 0.8)
        self.assertEqual(len(self.analogizer.analogies), 0)
    
    def test_find_analogy(self):
        """Test finding analogy between patterns"""
        analogy = self.analogizer.find_analogy(self.pattern_a, self.pattern_b)
        
        self.assertIsNotNone(analogy)
        self.assertGreater(analogy['similarity'], 0.5)
        self.assertEqual(len(self.analogizer.analogies), 1)
    
    def test_create_metaphor(self):
        """Test creating metaphor"""
        field_a = NDAnalogField((5, 5))
        field_a.activation[1:4, 1:4] = 1.0
        
        field_b = NDAnalogField((5, 5))
        
        metaphor_id = self.analogizer.create_metaphor(field_a, field_b, "test_metaphor")
        
        self.assertIsNotNone(metaphor_id)
        self.assertIn(metaphor_id, self.analogizer.metaphor_mappings)
        self.assertEqual(self.analogizer.metaphor_mappings[metaphor_id]['name'], "test_metaphor")
    
    def test_apply_metaphor(self):
        """Test applying existing metaphor"""
        field_a = NDAnalogField((5, 5))
        field_a.activation[1:4, 1:4] = 1.0
        
        field_b = NDAnalogField((5, 5))
        
        metaphor_id = self.analogizer.create_metaphor(field_a, field_b, "test_metaphor")
        
        result = self.analogizer.apply_metaphor(metaphor_id)
        
        self.assertTrue(result)
        self.assertGreater(self.analogizer.metaphor_mappings[metaphor_id]['usage_count'], 0)
    
    def test_find_structural_analogy(self):
        """Test finding structural analogy"""
        structural_analogy = self.analogizer.find_structural_analogy(self.pattern_a, self.pattern_b)
        
        self.assertIn('similarity', structural_analogy)
        self.assertIn('matching_regions', structural_analogy)
        self.assertIn('is_structural_analogy', structural_analogy)
    
    def test_create_analogical_bridge(self):
        """Test creating analogical bridge"""
        field_a = NDAnalogField((5, 5))
        field_a.activation[1:3, 1:3] = 1.0
        
        field_b = NDAnalogField((5, 5))
        field_b.activation[3:5, 3:5] = 1.0
        
        result = self.analogizer.create_analogical_bridge(field_a, field_b, (1, 1), (3, 3))
        
        self.assertTrue(result)
        self.assertGreater(len(self.analogizer.bridge.bridges), 0)
    
    def test_get_analogy_network(self):
        """Test getting analogy network"""
        # Create some analogies
        self.analogizer.find_analogy(self.pattern_a, self.pattern_b)
        
        network = self.analogizer.get_analogy_network()
        
        self.assertIn('analogies', network)
        self.assertIn('network_connections', network)
        self.assertIn('metaphors', network)
        self.assertIn('statistics', network)
    
    def test_get_analogy_statistics(self):
        """Test getting analogy statistics"""
        # Create some analogies
        self.analogizer.find_analogy(self.pattern_a, self.pattern_b)
        
        stats = self.analogizer.get_analogy_statistics()
        
        self.assertIn('analogy_count', stats)
        self.assertIn('metaphor_count', stats)
        self.assertIn('average_similarity', stats)
        self.assertGreater(stats['analogy_count'], 0)
    
    def test_find_analogous_patterns(self):
        """Test finding analogous patterns"""
        # Create some analogies
        self.analogizer.find_analogy(self.pattern_a, self.pattern_b)
        
        query_pattern = np.array([[0.8, 0.2, 0.8], [0.2, 0.8, 0.2], [0.8, 0.2, 0.8]])
        analogous = self.analogizer.find_analogous_patterns(query_pattern)
        
        self.assertIsInstance(analogous, list)
    
    def test_strengthen_analogy(self):
        """Test strengthening analogy"""
        analogy = self.analogizer.find_analogy(self.pattern_a, self.pattern_b)
        original_strength = analogy['strength']
        
        self.analogizer.strengthen_analogy(analogy['id'], 1.2)
        
        # Check if strength increased
        updated_analogy = next(a for a in self.analogizer.analogies if a['id'] == analogy['id'])
        self.assertGreater(updated_analogy['strength'], original_strength)
    
    def test_reset(self):
        """Test analogizer reset"""
        # Create some data
        self.analogizer.find_analogy(self.pattern_a, self.pattern_b)
        
        self.analogizer.reset()
        
        self.assertEqual(len(self.analogizer.analogies), 0)
        self.assertEqual(len(self.analogizer.metaphor_mappings), 0)


class TestContradictionResolver(unittest.TestCase):
    """Test ContradictionResolver molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.resolver = ContradictionResolver(equilibrium_rate=0.3, composition_strength=0.8)
        self.claim_a = NDAnalogField((5, 5))
        self.claim_a.activation[1:4, 1:4] = 1.0
        
        self.claim_b = NDAnalogField((5, 5))
        self.claim_b.activation[2:5, 2:5] = 1.0
    
    def test_initialization(self):
        """Test resolver initialization"""
        self.assertEqual(self.resolver.balancer.equilibrium_rate, 0.3)
        self.assertEqual(self.resolver.composer.preserve_intermediate, True)
        self.assertEqual(len(self.resolver.contradictions), 0)
    
    def test_resolve_contradiction(self):
        """Test resolving contradiction"""
        resolution = self.resolver.resolve_contradiction(self.claim_a, self.claim_b)
        
        self.assertIsNotNone(resolution)
        self.assertIn('resolved', resolution)
        self.assertIn('tension_level', resolution)
        self.assertGreater(len(self.resolver.contradictions), 0)
        self.assertGreater(len(self.resolver.resolutions), 0)
    
    def test_create_synthesis(self):
        """Test creating synthesis from opposing views"""
        opposing_views = [self.claim_a, self.claim_b]
        
        synthesis_id = self.resolver.create_synthesis(opposing_views, "test_synthesis")
        
        self.assertIsNotNone(synthesis_id)
        self.assertIn(synthesis_id, self.resolver.syntheses)
        self.assertEqual(self.resolver.syntheses[synthesis_id]['name'], "test_synthesis")
    
    def test_apply_synthesis(self):
        """Test applying existing synthesis"""
        opposing_views = [self.claim_a, self.claim_b]
        synthesis_id = self.resolver.create_synthesis(opposing_views, "test_synthesis")
        
        target_field = NDAnalogField((5, 5))
        result = self.resolver.apply_synthesis(synthesis_id, target_field)
        
        self.assertTrue(result)
        self.assertGreater(self.resolver.syntheses[synthesis_id]['usage_count'], 0)
    
    def test_detect_contradictions(self):
        """Test detecting contradictions"""
        contradictions = self.resolver.detect_contradictions(self.claim_a, self.claim_b)
        
        self.assertIsInstance(contradictions, list)
    
    def test_get_contradiction_strength(self):
        """Test getting contradiction strength"""
        strength = self.resolver.get_contradiction_strength(self.claim_a, self.claim_b)
        
        self.assertIsInstance(strength, float)
        self.assertGreaterEqual(strength, 0.0)
    
    def test_get_resolution_statistics(self):
        """Test getting resolution statistics"""
        # Create some resolutions
        self.resolver.resolve_contradiction(self.claim_a, self.claim_b)
        
        stats = self.resolver.get_resolution_statistics()
        
        self.assertIn('total_contradictions', stats)
        self.assertIn('total_resolutions', stats)
        self.assertIn('resolution_rate', stats)
        self.assertGreater(stats['total_contradictions'], 0)
    
    def test_get_synthesis_by_name(self):
        """Test getting synthesis by name"""
        opposing_views = [self.claim_a, self.claim_b]
        synthesis_id = self.resolver.create_synthesis(opposing_views, "test_synthesis")
        
        synthesis = self.resolver.get_synthesis_by_name("test_synthesis")
        
        self.assertIsNotNone(synthesis)
        self.assertEqual(synthesis['name'], "test_synthesis")
    
    def test_reset(self):
        """Test resolver reset"""
        # Create some data
        self.resolver.resolve_contradiction(self.claim_a, self.claim_b)
        
        self.resolver.reset()
        
        self.assertEqual(len(self.resolver.contradictions), 0)
        self.assertEqual(len(self.resolver.resolutions), 0)
        self.assertEqual(len(self.resolver.syntheses), 0)


class TestGapFiller(unittest.TestCase):
    """Test GapFiller molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.gap_filler = GapFiller(creativity=0.6, anticipation_strength=0.4)
        self.field = NDAnalogField((8, 8))
        self.field.activation[2:4, 2:4] = 1.0
        self.field.activation[6:8, 6:8] = 1.0
        # Create a gap in the middle
        self.field.activation[4:6, 4:6] = 0.0
    
    def test_initialization(self):
        """Test gap filler initialization"""
        self.assertEqual(self.gap_filler.creativity, 0.6)
        self.assertEqual(self.gap_filler.anticipation_strength, 0.4)
        self.assertEqual(len(self.gap_filler.fill_history), 0)
    
    def test_fill_gaps(self):
        """Test filling gaps in field"""
        original_energy = np.sum(np.abs(self.field.activation))
        
        filled_field = self.gap_filler.fill_gaps(self.field)
        
        self.assertIsNotNone(filled_field)
        self.assertGreater(len(self.gap_filler.fill_history), 0)
        # Energy should increase after filling gaps
        self.assertGreater(np.sum(np.abs(filled_field.activation)), original_energy)
    
    def test_analyze_gaps(self):
        """Test analyzing gaps in field"""
        gap_analysis = self.gap_filler.analyze_gaps(self.field)
        
        self.assertIn('gap_count', gap_analysis)
        self.assertIn('gap_locations', gap_analysis)
        self.assertIn('gap_density', gap_analysis)
        self.assertIn('gap_details', gap_analysis)
        self.assertGreater(gap_analysis['gap_count'], 0)
    
    def test_get_predictions(self):
        """Test getting predictions"""
        self.gap_filler.fill_gaps(self.field)
        
        predictions = self.gap_filler.get_predictions()
        
        self.assertIsInstance(predictions, list)
    
    def test_get_prediction_accuracy(self):
        """Test getting prediction accuracy"""
        self.gap_filler.fill_gaps(self.field)
        
        accuracy = self.gap_filler.get_prediction_accuracy()
        
        self.assertIn('total_predictions', accuracy)
        self.assertIn('average_error', accuracy)
        self.assertIn('accuracy', accuracy)
    
    def test_create_bridging_concept(self):
        """Test creating bridging concept"""
        field_a = NDAnalogField((5, 5))
        field_a.activation[1:3, 1:3] = 1.0
        
        field_b = NDAnalogField((5, 5))
        field_b.activation[3:5, 3:5] = 1.0
        
        bridge_pattern = self.gap_filler.create_bridging_concept(field_a, field_b)
        
        self.assertIsNotNone(bridge_pattern)
        self.assertEqual(bridge_pattern.shape, field_a.shape)
    
    def test_fill_gaps_with_anticipation(self):
        """Test filling gaps using anticipation"""
        filled_field = self.gap_filler.fill_gaps_with_anticipation(self.field)
        
        self.assertIsNotNone(filled_field)
        self.assertEqual(filled_field.shape, self.field.shape)
    
    def test_get_gap_statistics(self):
        """Test getting gap statistics"""
        self.gap_filler.fill_gaps(self.field)
        
        stats = self.gap_filler.get_gap_statistics()
        
        self.assertIn('total_operations', stats)
        self.assertIn('total_gaps_filled', stats)
        self.assertIn('average_gaps_per_operation', stats)
        self.assertGreater(stats['total_operations'], 0)
    
    def test_get_surprise_map(self):
        """Test getting surprise map"""
        surprise_map = self.gap_filler.get_surprise_map(self.field)
        
        self.assertIsNotNone(surprise_map)
        self.assertEqual(surprise_map.shape, self.field.shape)
    
    def test_adapt_creativity(self):
        """Test adapting creativity"""
        original_creativity = self.gap_filler.creativity
        
        self.gap_filler.adapt_creativity(0.1)
        
        # Creativity should be within bounds
        self.assertGreaterEqual(self.gap_filler.creativity, 0.1)
        self.assertLessEqual(self.gap_filler.creativity, 0.9)
    
    def test_get_filled_gaps(self):
        """Test getting filled gaps"""
        self.gap_filler.fill_gaps(self.field)
        
        filled_gaps = self.gap_filler.get_filled_gaps()
        
        self.assertIsInstance(filled_gaps, list)
    
    def test_reset(self):
        """Test gap filler reset"""
        # Create some data
        self.gap_filler.fill_gaps(self.field)
        
        self.gap_filler.reset()
        
        self.assertEqual(len(self.gap_filler.fill_history), 0)
        self.assertEqual(len(self.gap_filler.predictions), 0)


class TestPatternCompleter(unittest.TestCase):
    """Test PatternCompleter molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.completer = PatternCompleter(anticipation_strength=0.5, resonance_threshold=0.6)
        self.partial_field = NDAnalogField((5, 5))
        self.partial_field.activation[1:3, 1:3] = 1.0
        # Leave some areas empty for completion
    
    def test_initialization(self):
        """Test completer initialization"""
        self.assertEqual(self.completer.anticipation_strength, 0.5)
        self.assertEqual(self.completer.resonance_threshold, 0.6)
        self.assertEqual(len(self.completer.completions), 0)
    
    def test_complete_pattern(self):
        """Test completing partial pattern"""
        completed = self.completer.complete_pattern(self.partial_field)
        
        self.assertIsNotNone(completed)
        self.assertEqual(completed.shape, self.partial_field.shape)
        self.assertGreater(len(self.completer.completions), 0)
    
    def test_add_completion_template(self):
        """Test adding completion template"""
        template_pattern = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        
        self.completer.add_completion_template("test_template", template_pattern, "Test template")
        
        self.assertIn("test_template", self.completer.completion_templates)
        self.assertEqual(self.completer.completion_templates["test_template"]['description'], "Test template")
    
    def test_get_completion_candidates(self):
        """Test getting completion candidates"""
        candidates = self.completer.get_completion_candidates(self.partial_field)
        
        self.assertIsInstance(candidates, list)
        self.assertGreater(len(candidates), 0)
        
        for candidate in candidates:
            self.assertIn('type', candidate)
            self.assertIn('pattern', candidate)
            self.assertIn('score', candidate)
    
    def test_get_completion_confidence(self):
        """Test getting completion confidence"""
        confidence = self.completer.get_completion_confidence(self.partial_field)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_get_completion_statistics(self):
        """Test getting completion statistics"""
        # Create some completions
        self.completer.complete_pattern(self.partial_field)
        
        stats = self.completer.get_completion_statistics()
        
        self.assertIn('total_completions', stats)
        self.assertIn('average_energy_increase', stats)
        self.assertIn('template_count', stats)
        self.assertGreater(stats['total_completions'], 0)
    
    def test_complete_with_template(self):
        """Test completing pattern with template"""
        template_pattern = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.completer.add_completion_template("test_template", template_pattern)
        
        completed = self.completer.complete_pattern(self.partial_field, template_id="test_template")
        
        self.assertIsNotNone(completed)
        self.assertGreater(self.completer.completion_templates["test_template"]['usage_count'], 0)
    
    def test_reset(self):
        """Test completer reset"""
        # Create some data
        self.completer.complete_pattern(self.partial_field)
        
        self.completer.reset()
        
        self.assertEqual(len(self.completer.completions), 0)
        self.assertEqual(len(self.completer.completion_templates), 0)


class TestReasoningIntegration(unittest.TestCase):
    """Test integration between reasoning molecules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analogizer = Analogizer()
        self.resolver = ContradictionResolver()
        self.gap_filler = GapFiller()
        self.completer = PatternCompleter()
        
        self.field_a = NDAnalogField((8, 8))
        self.field_a.activation[2:4, 2:4] = 1.0
        
        self.field_b = NDAnalogField((8, 8))
        self.field_b.activation[4:6, 4:6] = 1.0
    
    def test_reasoning_pipeline(self):
        """Test complete reasoning pipeline"""
        # Step 1: Find analogy
        analogy = self.analogizer.find_analogy(self.field_a.activation, self.field_b.activation)
        
        # Step 2: Resolve any contradictions
        resolution = self.resolver.resolve_contradiction(self.field_a, self.field_b)
        
        # Step 3: Fill gaps
        filled_field = self.gap_filler.fill_gaps(self.field_a)
        
        # Step 4: Complete patterns
        completed_field = self.completer.complete_pattern(filled_field)
        
        # All should complete without error
        self.assertIsNotNone(analogy)
        self.assertIsNotNone(resolution)
        self.assertIsNotNone(filled_field)
        self.assertIsNotNone(completed_field)
    
    def test_reasoning_coordination(self):
        """Test coordination between reasoning mechanisms"""
        # Create analogy
        analogy = self.analogizer.find_analogy(self.field_a.activation, self.field_b.activation)
        
        # Use analogy to guide gap filling
        if analogy.get('is_analogy', False):
            # Fill gaps using analogous pattern
            self.gap_filler.fill_gaps(self.field_a)
            
            # Complete using analogy as template
            template_pattern = analogy['pattern_b']
            self.completer.add_completion_template("analogy_template", template_pattern)
            completed = self.completer.complete_pattern(self.field_a, template_id="analogy_template")
            
            self.assertIsNotNone(completed)
    
    def test_reasoning_adaptation(self):
        """Test adaptive reasoning based on performance"""
        # Create initial reasoning
        analogy = self.analogizer.find_analogy(self.field_a.activation, self.field_b.activation)
        resolution = self.resolver.resolve_contradiction(self.field_a, self.field_b)
        
        # Adapt based on results
        if resolution.get('is_resolved', False):
            # If contradiction resolved, strengthen analogy
            if analogy.get('is_analogy', False):
                self.analogizer.strengthen_analogy(analogy['id'], 1.1)
        
        # Adapt gap filler creativity
        self.gap_filler.adapt_creativity(0.1)
        
        # All adaptations should complete without error
        self.assertTrue(True)


def create_test_pattern():
    """Helper function to create test pattern"""
    pattern = np.zeros((5, 5))
    pattern[1:4, 1:4] = 1.0
    return pattern


def create_test_field_with_gaps():
    """Helper function to create test field with gaps"""
    field = NDAnalogField((8, 8))
    field.activation[1:3, 1:3] = 1.0
    field.activation[5:7, 5:7] = 1.0
    # Gap in the middle
    field.activation[3:5, 3:5] = 0.0
    return field


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

