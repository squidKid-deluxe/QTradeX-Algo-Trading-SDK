# ============================================================================
# FieldCorrelator Test Suite
# ============================================================================

"""
Test suite for FieldCorrelator organism
"""

import numpy as np
import unittest

try:
    from .field_correlator import FieldCorrelator, SubsystemType, CategoryDimension, SubsystemField
except ImportError:
    from combinatronix.organisms.integration.field_correlator import FieldCorrelator, SubsystemType, CategoryDimension, SubsystemField


class TestFieldCorrelator(unittest.TestCase):
    """Test FieldCorrelator organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.correlator = FieldCorrelator({'enable_visualization': False})
    
    def test_initialization(self):
        """Test correlator initialization"""
        self.assertIsNotNone(self.correlator.atoms)
        self.assertIsNotNone(self.correlator.molecules)
        self.assertIsNotNone(self.correlator.fields)
        self.assertEqual(len(self.correlator.atoms), 6)
        self.assertEqual(len(self.correlator.molecules), 4)
    
    def test_register_subsystem(self):
        """Test subsystem registration"""
        self.correlator.register_subsystem("test_system", SubsystemType.VISUAL, (16, 16))
        
        self.assertIn("test_system", self.correlator.state['subsystems'])
        self.assertEqual(len(self.correlator.state['subsystems']), 1)
        
        subsystem = self.correlator.state['subsystems']["test_system"]
        self.assertEqual(subsystem.name, "test_system")
        self.assertEqual(subsystem.subsystem, SubsystemType.VISUAL)
        self.assertEqual(subsystem.shape, (16, 16))
    
    def test_update_subsystem(self):
        """Test subsystem field updates"""
        self.correlator.register_subsystem("test_system", SubsystemType.VISUAL, (8, 8))
        
        # Update with activation pattern
        activation = np.random.random((8, 8))
        self.correlator.update_subsystem("test_system", activation)
        
        subsystem = self.correlator.state['subsystems']["test_system"]
        np.testing.assert_array_almost_equal(subsystem.field.activation, activation)
        self.assertEqual(subsystem.total_activation, np.sum(activation))
        self.assertEqual(subsystem.peak_activation, np.max(activation))
    
    def test_compute_correlations_single_subsystem(self):
        """Test correlation computation with single subsystem"""
        self.correlator.register_subsystem("test_system", SubsystemType.VISUAL, (8, 8))
        
        # Should return zero matrix for single subsystem
        correlation_matrix = self.correlator.compute_correlations()
        
        self.assertEqual(correlation_matrix.shape, (12, 12))
        self.assertTrue(np.all(correlation_matrix == 0))
    
    def test_compute_correlations_multiple_subsystems(self):
        """Test correlation computation with multiple subsystems"""
        # Register two subsystems
        self.correlator.register_subsystem("visual", SubsystemType.VISUAL, (8, 8))
        self.correlator.register_subsystem("auditory", SubsystemType.AUDITORY, (8, 8))
        
        # Update with different patterns
        visual_pattern = np.zeros((8, 8))
        visual_pattern[2:6, 2:6] = 1.0
        self.correlator.update_subsystem("visual", visual_pattern)
        
        auditory_pattern = np.zeros((8, 8))
        auditory_pattern[3:5, 3:5] = 0.8
        self.correlator.update_subsystem("auditory", auditory_pattern)
        
        # Compute correlations
        correlation_matrix = self.correlator.compute_correlations()
        
        self.assertEqual(correlation_matrix.shape, (12, 12))
        self.assertGreater(np.mean(correlation_matrix), 0)  # Should have some correlations
    
    def test_intensity_correlation(self):
        """Test intensity correlation computation"""
        self.correlator.register_subsystem("system1", SubsystemType.VISUAL, (4, 4))
        self.correlator.register_subsystem("system2", SubsystemType.AUDITORY, (4, 4))
        
        # High intensity pattern
        pattern1 = np.ones((4, 4)) * 0.8
        pattern2 = np.ones((4, 4)) * 0.9
        
        self.correlator.update_subsystem("system1", pattern1)
        self.correlator.update_subsystem("system2", pattern2)
        
        correlation_matrix = self.correlator.compute_correlations()
        
        # Should have high intensity correlation
        intensity_corr = correlation_matrix[CategoryDimension.INTENSITY.value, 0]
        self.assertGreater(intensity_corr, 0.5)
    
    def test_coherence_correlation(self):
        """Test coherence correlation computation"""
        self.correlator.register_subsystem("system1", SubsystemType.VISUAL, (6, 6))
        self.correlator.register_subsystem("system2", SubsystemType.AUDITORY, (6, 6))
        
        # Structured patterns
        pattern1 = np.zeros((6, 6))
        pattern1[2:4, 2:4] = 1.0  # Structured
        
        pattern2 = np.zeros((6, 6))
        pattern2[1:5, 1:5] = 0.5  # Also structured
        
        self.correlator.update_subsystem("system1", pattern1)
        self.correlator.update_subsystem("system2", pattern2)
        
        correlation_matrix = self.correlator.compute_correlations()
        
        # Should have some coherence correlation
        coherence_corr = correlation_matrix[CategoryDimension.COHERENCE.value, 0]
        self.assertGreaterEqual(coherence_corr, 0)
    
    def test_temporal_correlation(self):
        """Test temporal correlation with memory"""
        self.correlator.register_subsystem("system1", SubsystemType.VISUAL, (4, 4))
        self.correlator.register_subsystem("system2", SubsystemType.AUDITORY, (4, 4))
        
        # Current patterns
        pattern1 = np.ones((4, 4)) * 0.8
        pattern2 = np.ones((4, 4)) * 0.7
        
        # Memory patterns (similar to current)
        memory1 = pattern1 * 0.9
        memory2 = pattern2 * 0.8
        
        self.correlator.update_subsystem("system1", pattern1, memory1)
        self.correlator.update_subsystem("system2", pattern2, memory2)
        
        correlation_matrix = self.correlator.compute_correlations()
        
        # Should have temporal correlation
        temporal_corr = correlation_matrix[CategoryDimension.TEMPORAL.value, 0]
        self.assertGreaterEqual(temporal_corr, 0)
    
    def test_novelty_correlation(self):
        """Test novelty correlation computation"""
        self.correlator.register_subsystem("system1", SubsystemType.VISUAL, (4, 4))
        self.correlator.register_subsystem("system2", SubsystemType.AUDITORY, (4, 4))
        
        # Current patterns
        pattern1 = np.ones((4, 4)) * 0.8
        pattern2 = np.ones((4, 4)) * 0.7
        
        # Very different memory patterns
        memory1 = 1.0 - pattern1  # Opposite
        memory2 = 1.0 - pattern2  # Opposite
        
        self.correlator.update_subsystem("system1", pattern1, memory1)
        self.correlator.update_subsystem("system2", pattern2, memory2)
        
        correlation_matrix = self.correlator.compute_correlations()
        
        # Should have novelty correlation
        novelty_corr = correlation_matrix[CategoryDimension.NOVELTY.value, 0]
        self.assertGreaterEqual(novelty_corr, 0)
    
    def test_get_summary(self):
        """Test summary generation"""
        self.correlator.register_subsystem("visual", SubsystemType.VISUAL, (8, 8))
        self.correlator.register_subsystem("auditory", SubsystemType.AUDITORY, (8, 8))
        
        # Add some patterns
        visual_pattern = np.random.random((8, 8))
        auditory_pattern = np.random.random((8, 8))
        
        self.correlator.update_subsystem("visual", visual_pattern)
        self.correlator.update_subsystem("auditory", auditory_pattern)
        
        # Compute correlations
        self.correlator.compute_correlations()
        
        # Get summary
        summary = self.correlator.get_summary()
        
        self.assertIn('overall_correlation', summary)
        self.assertIn('peak_correlation', summary)
        self.assertIn('system_coherence', summary)
        self.assertIn('system_stability', summary)
        self.assertIn('subsystem_count', summary)
        self.assertIn('dimensions', summary)
        self.assertIn('recommendations', summary)
    
    def test_get_state(self):
        """Test state retrieval"""
        self.correlator.register_subsystem("test_system", SubsystemType.VISUAL, (8, 8))
        
        state = self.correlator.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test system reset"""
        # Register subsystems and compute correlations
        self.correlator.register_subsystem("visual", SubsystemType.VISUAL, (8, 8))
        self.correlator.register_subsystem("auditory", SubsystemType.AUDITORY, (8, 8))
        
        visual_pattern = np.random.random((8, 8))
        auditory_pattern = np.random.random((8, 8))
        
        self.correlator.update_subsystem("visual", visual_pattern)
        self.correlator.update_subsystem("auditory", auditory_pattern)
        self.correlator.compute_correlations()
        
        # Reset
        self.correlator.reset()
        
        # Check that state is reset
        self.assertEqual(len(self.correlator.state['subsystems']), 0)
        self.assertEqual(self.correlator.state['processed_correlations'], 0)
        self.assertTrue(np.all(self.correlator.state['correlation_matrix'] == 0))
    
    def test_export_results(self):
        """Test results export"""
        self.correlator.register_subsystem("visual", SubsystemType.VISUAL, (8, 8))
        self.correlator.register_subsystem("auditory", SubsystemType.AUDITORY, (8, 8))
        
        visual_pattern = np.random.random((8, 8))
        auditory_pattern = np.random.random((8, 8))
        
        self.correlator.update_subsystem("visual", visual_pattern)
        self.correlator.update_subsystem("auditory", auditory_pattern)
        self.correlator.compute_correlations()
        
        # Export should complete without error
        try:
            self.correlator.export_results("test_export.json")
            # Clean up
            import os
            if os.path.exists("test_export.json"):
                os.remove("test_export.json")
        except Exception as e:
            self.fail(f"Export failed with error: {e}")
    
    def test_multiple_subsystem_types(self):
        """Test with different subsystem types"""
        subsystem_types = [
            SubsystemType.VISUAL,
            SubsystemType.AUDITORY,
            SubsystemType.MOTOR,
            SubsystemType.MEMORY,
            SubsystemType.ATTENTION,
            SubsystemType.EMOTION
        ]
        
        for i, subsystem_type in enumerate(subsystem_types):
            name = f"system_{i}"
            self.correlator.register_subsystem(name, subsystem_type, (8, 8))
            
            # Add some pattern
            pattern = np.random.random((8, 8))
            self.correlator.update_subsystem(name, pattern)
        
        # Compute correlations
        correlation_matrix = self.correlator.compute_correlations()
        
        self.assertEqual(correlation_matrix.shape, (12, 12))
        self.assertGreater(np.mean(correlation_matrix), 0)
    
    def test_correlation_dimensions(self):
        """Test all 12 correlation dimensions"""
        self.correlator.register_subsystem("system1", SubsystemType.VISUAL, (8, 8))
        self.correlator.register_subsystem("system2", SubsystemType.AUDITORY, (8, 8))
        
        pattern1 = np.random.random((8, 8))
        pattern2 = np.random.random((8, 8))
        
        self.correlator.update_subsystem("system1", pattern1)
        self.correlator.update_subsystem("system2", pattern2)
        
        correlation_matrix = self.correlator.compute_correlations()
        
        # Check that all dimensions have values
        for dim in CategoryDimension:
            idx = dim.value
            self.assertGreaterEqual(correlation_matrix[idx, 0], 0)  # Overall correlation
            self.assertLessEqual(correlation_matrix[idx, 0], 1)     # Should be normalized


if __name__ == '__main__':
    unittest.main(verbosity=2)

