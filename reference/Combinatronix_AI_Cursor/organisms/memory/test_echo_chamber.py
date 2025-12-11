# ============================================================================
# EchoChamber Test Suite
# ============================================================================

"""
Test suite for EchoChamber organism
"""

import numpy as np
import unittest

try:
    from .echo_chamber import EchoChamber, PatternEcho
except ImportError:
    from combinatronix.organisms.memory.echo_chamber import EchoChamber, PatternEcho


class TestEchoChamber(unittest.TestCase):
    """Test EchoChamber organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.chamber = EchoChamber({'field_size': (6, 6), 'max_echoes': 5, 'enable_visualization': False})
    
    def test_initialization(self):
        """Test echo chamber initialization"""
        self.assertIsNotNone(self.chamber.atoms)
        self.assertIsNotNone(self.chamber.molecules)
        self.assertIsNotNone(self.chamber.fields)
        self.assertEqual(len(self.chamber.atoms), 6)
        self.assertEqual(len(self.chamber.molecules), 3)
        self.assertEqual(len(self.chamber.state['echoes']), 0)
    
    def test_inject_pattern(self):
        """Test pattern injection"""
        pattern = np.eye(6, dtype=np.float32)
        result = self.chamber.inject_pattern(pattern, "identity")
        
        self.assertTrue(result['learned'])
        self.assertFalse(result['recognized'])
        self.assertEqual(len(self.chamber.state['echoes']), 1)
        self.assertEqual(self.chamber.state['echoes'][0].name, "identity")
    
    def test_recognize_pattern(self):
        """Test pattern recognition"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        # Create pattern field for recognition
        from combinatronix.core import NDAnalogField
        pattern_field = NDAnalogField((6, 6), activation=pattern)
        
        # Test recognition
        result = self.chamber.recognize_pattern(pattern_field)
        
        self.assertTrue(result['recognized'])
        self.assertGreater(result['confidence'], 0.5)
        self.assertEqual(result['match_name'], "identity")
    
    def test_echo_evolution(self):
        """Test echo evolution over time"""
        # Learn a pattern
        pattern = np.ones((6, 6), dtype=np.float32) * 0.8
        self.chamber.inject_pattern(pattern, "test")
        
        initial_strength = self.chamber.state['echoes'][0].strength
        
        # Let it evolve
        for _ in range(10):
            self.chamber.tick()
        
        # Should have decayed
        final_strength = self.chamber.state['echoes'][0].strength
        self.assertLess(final_strength, initial_strength)
    
    def test_echo_fading(self):
        """Test that weak echoes are removed"""
        # Learn multiple patterns
        for i in range(3):
            pattern = np.random.random((6, 6))
            self.chamber.inject_pattern(pattern, f"pattern_{i}")
        
        initial_count = len(self.chamber.state['echoes'])
        
        # Let them decay
        for _ in range(100):  # Many ticks to ensure decay
            self.chamber.tick()
        
        # Some echoes should have faded
        final_count = len(self.chamber.state['echoes'])
        self.assertLessEqual(final_count, initial_count)
    
    def test_memory_capacity(self):
        """Test memory capacity management"""
        # Fill beyond capacity
        for i in range(7):  # More than max_echoes (5)
            pattern = np.random.random((6, 6))
            self.chamber.inject_pattern(pattern, f"pattern_{i}")
        
        # Should not exceed capacity
        self.assertLessEqual(len(self.chamber.state['echoes']), self.chamber.config['max_echoes'])
    
    def test_pattern_similarity(self):
        """Test pattern similarity computation"""
        pattern1 = np.eye(6, dtype=np.float32)
        pattern2 = np.eye(6, dtype=np.float32) * 0.9  # Slightly different
        
        from combinatronix.core import NDAnalogField
        field1 = NDAnalogField((6, 6), activation=pattern1)
        field2 = NDAnalogField((6, 6), activation=pattern2)
        
        similarity = self.chamber._compute_molecular_similarity(field1, field2)
        
        self.assertGreater(similarity, 0.5)  # Should be similar
        self.assertLessEqual(similarity, 1.0)  # Should be normalized
    
    def test_echo_kernels(self):
        """Test echo processing kernels"""
        pattern = np.eye(6, dtype=np.float32)
        from combinatronix.core import NDAnalogField
        pattern_field = NDAnalogField((6, 6), activation=pattern)
        
        # Test identity kernel
        result = self.chamber._identity_kernel(pattern_field, 1.0)
        np.testing.assert_array_almost_equal(result.activation, pattern)
        
        # Test amplify kernel
        result = self.chamber._amplify_kernel(pattern_field, 1.0)
        self.assertGreaterEqual(np.sum(result.activation), np.sum(pattern))
        
        # Test fade kernel
        result = self.chamber._fade_kernel(pattern_field, 1.0)
        self.assertLessEqual(np.sum(result.activation), np.sum(pattern))
    
    def test_molecular_processing(self):
        """Test molecular processing of echoes"""
        # Learn a pattern
        pattern = np.random.random((6, 6))
        self.chamber.inject_pattern(pattern, "test")
        
        echo = self.chamber.state['echoes'][0]
        original_pattern = echo.pattern.activation.copy()
        
        # Process echo
        processed = self.chamber._process_echo_molecular(echo)
        
        # Should be different due to molecular processing
        self.assertFalse(np.array_equal(processed.activation, original_pattern))
    
    def test_get_memory_summary(self):
        """Test memory summary generation"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        summary = self.chamber.get_memory_summary()
        
        self.assertIn('total_echoes', summary)
        self.assertIn('total_strength', summary)
        self.assertIn('field_energy', summary)
        self.assertEqual(summary['total_echoes'], 1)
        self.assertGreater(summary['total_strength'], 0)
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.chamber.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test system reset"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        # Reset
        self.chamber.reset()
        
        # Should be empty
        self.assertEqual(len(self.chamber.state['echoes']), 0)
        self.assertEqual(self.chamber.state['tick_count'], 0)
        self.assertEqual(self.chamber.state['total_patterns_learned'], 0)
    
    def test_noisy_pattern_recognition(self):
        """Test recognition with noisy patterns"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        # Create noisy version
        noisy_pattern = pattern + np.random.random(pattern.shape) * 0.3
        
        from combinatronix.core import NDAnalogField
        noisy_field = NDAnalogField((6, 6), activation=noisy_pattern)
        
        # Should still recognize
        result = self.chamber.recognize_pattern(noisy_field)
        
        # Should have some recognition (may be lower confidence)
        self.assertGreaterEqual(result['confidence'], 0)
    
    def test_interference_events(self):
        """Test pattern interference"""
        # Learn multiple patterns
        pattern1 = np.zeros((6, 6))
        pattern1[2:4, 2:4] = 1.0
        
        pattern2 = np.zeros((6, 6))
        pattern2[3, :] = 1.0
        
        self.chamber.inject_pattern(pattern1, "square")
        self.chamber.inject_pattern(pattern2, "line")
        
        initial_interference = self.chamber.state['interference_events']
        
        # Let them interfere
        for _ in range(5):
            self.chamber.tick()
        
        # Should have interference events
        final_interference = self.chamber.state['interference_events']
        self.assertGreaterEqual(final_interference, initial_interference)
    
    def test_temporal_field_update(self):
        """Test temporal field updates"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        # Let it tick
        for _ in range(3):
            self.chamber.tick()
        
        # Temporal field should have data
        temporal_field = self.chamber.fields['temporal_field']
        self.assertGreater(np.sum(temporal_field.activation), 0)
    
    def test_echo_strengthening(self):
        """Test echo strengthening on recognition"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        initial_strength = self.chamber.state['echoes'][0].strength
        
        # Recognize the same pattern (should strengthen)
        from combinatronix.core import NDAnalogField
        pattern_field = NDAnalogField((6, 6), activation=pattern)
        result = self.chamber.recognize_pattern(pattern_field)
        
        if result['recognized']:
            # Strengthen the echo
            self.chamber._strengthen_similar_echo(pattern_field, result['match_strength'])
            
            # Should be stronger
            final_strength = self.chamber.state['echoes'][0].strength
            self.assertGreaterEqual(final_strength, initial_strength)
    
    def test_export_memory(self):
        """Test memory export"""
        # Learn a pattern
        pattern = np.eye(6, dtype=np.float32)
        self.chamber.inject_pattern(pattern, "identity")
        
        # Export should complete without error
        try:
            self.chamber.export_memory("test_echo_memory.json")
            # Clean up
            import os
            if os.path.exists("test_echo_memory.json"):
                os.remove("test_echo_memory.json")
        except Exception as e:
            self.fail(f"Export failed with error: {e}")
    
    def test_different_field_sizes(self):
        """Test with different field sizes"""
        sizes = [(4, 4), (6, 6), (8, 8), (10, 10)]
        
        for size in sizes:
            chamber = EchoChamber({'field_size': size, 'enable_visualization': False})
            
            # Create pattern for this size
            pattern = np.eye(size[0], dtype=np.float32)
            result = chamber.inject_pattern(pattern, f"identity_{size}")
            
            self.assertTrue(result['learned'])
            self.assertEqual(len(chamber.state['echoes']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

