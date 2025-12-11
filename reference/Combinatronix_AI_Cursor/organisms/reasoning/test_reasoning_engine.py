# ============================================================================
# ReasoningEngine Test Suite
# ============================================================================

"""
Test suite for ReasoningEngine organism
"""

import numpy as np
import unittest

try:
    from .reasoning_engine import ReasoningEngine, ConceptTension, InventedSymbol
except ImportError:
    from combinatronix.organisms.reasoning.reasoning_engine import ReasoningEngine, ConceptTension, InventedSymbol


class TestReasoningEngine(unittest.TestCase):
    """Test ReasoningEngine organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = ReasoningEngine({'enable_visualization': False})
    
    def test_initialization(self):
        """Test engine initialization"""
        self.assertIsNotNone(self.engine.atoms)
        self.assertIsNotNone(self.engine.molecules)
        self.assertIsNotNone(self.engine.fields)
        self.assertEqual(len(self.engine.atoms), 6)
        self.assertEqual(len(self.engine.molecules), 4)
        self.assertEqual(self.engine.state['tick_counter'], 0)
        self.assertEqual(len(self.engine.state['invented_symbols']), 0)
    
    def test_inject_concepts(self):
        """Test concept injection"""
        concepts = {
            "big": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "small": np.array([[0, 0], [0, 1]], dtype=np.float32)
        }
        
        self.engine.inject_concepts(concepts)
        
        self.assertEqual(len(self.engine.state['concept_echoes']), 2)
        self.assertIn("big", self.engine.state['concept_echoes'])
        self.assertIn("small", self.engine.state['concept_echoes'])
    
    def test_inject_input_field(self):
        """Test input field injection"""
        input_field = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.engine.inject_input_field(input_field)
        
        # Should be resized to field size
        self.assertEqual(self.engine.fields['input_field'].activation.shape, self.engine.config['field_size'])
    
    def test_detect_contradictions_molecular(self):
        """Test contradiction detection using molecular operations"""
        # Create contradictory concepts
        concepts = {
            "big": np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.float32),
            "small": np.array([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.float32)
        }
        
        self.engine.inject_concepts(concepts)
        
        # Create input field with both concepts active
        input_field = concepts["big"] * 0.8 + concepts["small"] * 0.7
        self.engine.inject_input_field(input_field)
        
        # Detect contradictions
        all_patterns = self.engine.state['concept_echoes']
        contradictions = self.engine._detect_contradictions_molecular(all_patterns)
        
        self.assertGreater(len(contradictions), 0)
        self.assertEqual(contradictions[0].tension_type, "contradiction")
        self.assertEqual(contradictions[0].concept_a, "big")
        self.assertEqual(contradictions[0].concept_b, "small")
    
    def test_detect_gaps_molecular(self):
        """Test gap detection using molecular operations"""
        # Create minimal concepts
        concepts = {
            "cat": np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.float32)
        }
        
        self.engine.inject_concepts(concepts)
        
        # Create input field with gaps
        gap_field = np.zeros((8, 8))
        gap_field[3:5, 3:5] = 0.9  # Gap in center
        self.engine.inject_input_field(gap_field)
        
        # Detect gaps
        all_patterns = self.engine.state['concept_echoes']
        gaps = self.engine._detect_gaps_molecular(all_patterns)
        
        self.assertGreater(len(gaps), 0)
        self.assertEqual(gaps[0].tension_type, "gap")
    
    def test_detect_ambiguities_molecular(self):
        """Test ambiguity detection using molecular operations"""
        # Create similar concepts
        concepts = {
            "cat": np.array([
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.float32),
            "dog": np.array([
                [1, 1, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=np.float32)
        }
        
        self.engine.inject_concepts(concepts)
        
        # Create ambiguous input
        ambiguous_field = concepts["cat"] * 0.6 + concepts["dog"] * 0.7
        self.engine.inject_input_field(ambiguous_field)
        
        # Detect ambiguities
        all_patterns = self.engine.state['concept_echoes']
        ambiguities = self.engine._detect_ambiguities_molecular(all_patterns)
        
        self.assertGreater(len(ambiguities), 0)
        self.assertEqual(ambiguities[0].tension_type, "ambiguity")
    
    def test_detect_overflow_molecular(self):
        """Test overflow detection using molecular operations"""
        # Create overflow field
        overflow_field = np.ones((8, 8)) * 0.5
        overflow_field[3:5, 3:5] = 1.2  # Overflow region
        overflow_field = np.clip(overflow_field, 0, 1)
        
        self.engine.inject_input_field(overflow_field)
        
        # Detect overflow
        overflows = self.engine._detect_overflow_molecular()
        
        self.assertGreater(len(overflows), 0)
        self.assertEqual(overflows[0].tension_type, "overflow")
    
    def test_reason_step(self):
        """Test single reasoning step"""
        # Set up concepts and input
        concepts = {
            "big": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "small": np.array([[0, 0], [0, 1]], dtype=np.float32)
        }
        self.engine.inject_concepts(concepts)
        
        input_field = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.engine.inject_input_field(input_field)
        
        # Run reasoning step
        result = self.engine.reason_step()
        
        self.assertIn('tick', result)
        self.assertIn('tensions_detected', result)
        self.assertIn('symbols_invented', result)
        self.assertIn('total_symbols', result)
        self.assertIn('field_energy', result)
        self.assertIn('tension_energy', result)
        self.assertIn('new_symbols', result)
        self.assertIn('reasoning_operations', result)
        
        self.assertEqual(result['tick'], 1)
        self.assertGreaterEqual(result['tensions_detected'], 0)
        self.assertGreaterEqual(result['symbols_invented'], 0)
    
    def test_invent_symbol_for_tension_molecular(self):
        """Test symbol invention for different tension types"""
        # Test contradiction resolver
        tension = ConceptTension(
            concept_a="big",
            concept_b="small",
            tension_strength=0.8,
            field_location=(2, 2),
            tension_type="contradiction"
        )
        
        symbol = self.engine._invent_symbol_for_tension_molecular(tension)
        
        self.assertIsInstance(symbol, InventedSymbol)
        self.assertEqual(symbol.resolves_tension, tension)
        self.assertEqual(symbol.tension_strength, 0.8)
        self.assertIn("big", symbol.symbol)
        self.assertIn("small", symbol.symbol)
    
    def test_has_similar_symbol(self):
        """Test similar symbol detection"""
        # Create a tension
        tension1 = ConceptTension(
            concept_a="big",
            concept_b="small",
            tension_strength=0.8,
            field_location=(2, 2),
            tension_type="contradiction"
        )
        
        # Create another tension at nearby location
        tension2 = ConceptTension(
            concept_a="big",
            concept_b="small",
            tension_strength=0.7,
            field_location=(2, 3),  # Close to tension1
            tension_type="contradiction"
        )
        
        # Add first tension as invented symbol
        symbol = InventedSymbol(
            symbol="test_symbol",
            pattern=NDAnalogField(self.engine.config['field_size']),
            resolves_tension=tension1,
            strength=0.8,
            creation_tick=1,
            meaning_description="test"
        )
        self.engine.state['invented_symbols']['test_symbol'] = symbol
        
        # Check if similar symbol exists
        has_similar = self.engine._has_similar_symbol(tension2)
        self.assertTrue(has_similar)
    
    def test_update_tension_field(self):
        """Test tension field update"""
        # Add some tensions
        tension1 = ConceptTension(
            concept_a="big",
            concept_b="small",
            tension_strength=0.8,
            field_location=(2, 2),
            tension_type="contradiction"
        )
        tension2 = ConceptTension(
            concept_a="cat",
            concept_b="dog",
            tension_strength=0.6,
            field_location=(4, 4),
            tension_type="ambiguity"
        )
        
        self.engine.state['active_tensions'] = [tension1, tension2]
        
        # Update tension field
        self.engine._update_tension_field()
        
        # Check that tensions are recorded in field
        self.assertGreater(self.engine.fields['tension_field'].activation[2, 2], 0)
        self.assertGreater(self.engine.fields['tension_field'].activation[4, 4], 0)
    
    def test_apply_reasoning_kernels(self):
        """Test reasoning kernel application"""
        # Add different types of tensions
        tensions = [
            ConceptTension("big", "small", 0.8, (2, 2), "contradiction"),
            ConceptTension("unknown", "field", 0.6, (4, 4), "gap"),
            ConceptTension("cat", "dog", 0.7, (6, 6), "ambiguity")
        ]
        self.engine.state['active_tensions'] = tensions
        
        # Apply reasoning kernels
        operations = self.engine._apply_reasoning_kernels()
        
        self.assertIsInstance(operations, dict)
        self.assertIn('synthesize', operations)  # Should have synthesis for contradictions
        self.assertIn('analyze', operations)     # Should have analysis for gaps
        self.assertIn('abstract', operations)    # Should have abstraction for ambiguities
    
    def test_reasoning_kernels(self):
        """Test individual reasoning kernels"""
        # Test analyze kernel
        self.engine.state['active_tensions'] = [
            ConceptTension("unknown", "field", 0.6, (2, 2), "gap")
        ]
        result = self.engine._analyze_kernel()
        self.assertEqual(result, "analytical_spreading")
        
        # Test synthesize kernel
        self.engine.state['active_tensions'] = [
            ConceptTension("big", "small", 0.8, (2, 2), "contradiction")
        ]
        result = self.engine._synthesize_kernel()
        self.assertEqual(result, "synthetic_focusing")
        
        # Test abstract kernel
        self.engine.state['active_tensions'] = [
            ConceptTension("cat", "dog", 0.7, (2, 2), "ambiguity")
        ]
        result = self.engine._abstract_kernel()
        self.assertEqual(result, "identity_abstraction")
    
    def test_get_reasoning_summary(self):
        """Test reasoning summary generation"""
        # Add some state
        self.engine.state['tick_counter'] = 5
        self.engine.state['concept_echoes'] = {"big": None, "small": None}
        self.engine.state['invented_symbols'] = {"symbol1": None}
        self.engine.state['active_tensions'] = [None, None]
        
        summary = self.engine.get_reasoning_summary()
        
        self.assertIn('tick', summary)
        self.assertIn('known_concepts', summary)
        self.assertIn('invented_symbols', summary)
        self.assertIn('active_tensions', summary)
        self.assertIn('field_energy', summary)
        self.assertIn('tension_energy', summary)
        self.assertIn('strongest_tension', summary)
        self.assertIn('reasoning_history_length', summary)
        self.assertIn('total_symbols_invented', summary)
        
        self.assertEqual(summary['tick'], 5)
        self.assertEqual(summary['known_concepts'], 2)
        self.assertEqual(summary['invented_symbols'], 1)
        self.assertEqual(summary['active_tensions'], 2)
    
    def test_get_invented_symbols(self):
        """Test invented symbols retrieval"""
        # Add some invented symbols
        symbol1 = InventedSymbol(
            symbol="symbol1",
            pattern=NDAnalogField(self.engine.config['field_size']),
            resolves_tension=None,
            strength=0.8,
            creation_tick=1,
            meaning_description="test1"
        )
        symbol2 = InventedSymbol(
            symbol="symbol2",
            pattern=NDAnalogField(self.engine.config['field_size']),
            resolves_tension=None,
            strength=0.6,
            creation_tick=2,
            meaning_description="test2"
        )
        
        self.engine.state['invented_symbols'] = {
            "symbol1": symbol1,
            "symbol2": symbol2
        }
        
        symbols = self.engine.get_invented_symbols()
        
        self.assertEqual(len(symbols), 2)
        self.assertIn(symbol1, symbols)
        self.assertIn(symbol2, symbols)
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.engine.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test engine reset"""
        # Add some state
        self.engine.state['tick_counter'] = 5
        self.engine.state['concept_echoes'] = {"big": None}
        self.engine.state['invented_symbols'] = {"symbol1": None}
        self.engine.state['active_tensions'] = [None]
        
        # Reset
        self.engine.reset()
        
        # Check that state is reset
        self.assertEqual(self.engine.state['tick_counter'], 0)
        self.assertEqual(len(self.engine.state['concept_echoes']), 0)
        self.assertEqual(len(self.engine.state['invented_symbols']), 0)
        self.assertEqual(len(self.engine.state['active_tensions']), 0)
    
    def test_multiple_reasoning_steps(self):
        """Test multiple reasoning steps"""
        # Set up concepts and input
        concepts = {
            "big": np.array([[1, 1], [1, 1]], dtype=np.float32),
            "small": np.array([[0, 0], [0, 1]], dtype=np.float32)
        }
        self.engine.inject_concepts(concepts)
        
        input_field = np.array([[1, 0], [0, 1]], dtype=np.float32)
        self.engine.inject_input_field(input_field)
        
        # Run multiple reasoning steps
        for step in range(3):
            result = self.engine.reason_step()
            self.assertEqual(result['tick'], step + 1)
        
        # Check that history is recorded
        self.assertEqual(len(self.engine.state['reasoning_history']), 3)
        self.assertEqual(self.engine.state['tick_counter'], 3)
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'field_size': (6, 6), 'tension_sensitivity': 0.1},
            {'field_size': (10, 10), 'invention_threshold': 0.3},
            {'max_symbols': 20, 'reasoning_depth': 3}
        ]
        
        for config in configs:
            engine = ReasoningEngine(config)
            
            # Should work with different configs
            concepts = {"test": np.array([[1, 0], [0, 1]], dtype=np.float32)}
            engine.inject_concepts(concepts)
            
            input_field = np.array([[1, 0], [0, 1]], dtype=np.float32)
            engine.inject_input_field(input_field)
            
            result = engine.reason_step()
            self.assertIsInstance(result, dict)
    
    def test_concept_tension_creation(self):
        """Test ConceptTension dataclass"""
        tension = ConceptTension(
            concept_a="big",
            concept_b="small",
            tension_strength=0.8,
            field_location=(2, 3),
            tension_type="contradiction"
        )
        
        self.assertEqual(tension.concept_a, "big")
        self.assertEqual(tension.concept_b, "small")
        self.assertEqual(tension.tension_strength, 0.8)
        self.assertEqual(tension.field_location, (2, 3))
        self.assertEqual(tension.tension_type, "contradiction")
    
    def test_invented_symbol_creation(self):
        """Test InventedSymbol dataclass"""
        tension = ConceptTension("big", "small", 0.8, (2, 3), "contradiction")
        pattern = NDAnalogField((8, 8))
        
        symbol = InventedSymbol(
            symbol="test_symbol",
            pattern=pattern,
            resolves_tension=tension,
            strength=0.8,
            creation_tick=1,
            meaning_description="test description"
        )
        
        self.assertEqual(symbol.symbol, "test_symbol")
        self.assertEqual(symbol.pattern, pattern)
        self.assertEqual(symbol.resolves_tension, tension)
        self.assertEqual(symbol.strength, 0.8)
        self.assertEqual(symbol.creation_tick, 1)
        self.assertEqual(symbol.meaning_description, "test description")


if __name__ == '__main__':
    unittest.main(verbosity=2)

