# ============================================================================
# Attention Molecules Tests
# ============================================================================

"""
Comprehensive tests for attention molecules:
- Focus
- Saliency
- NoveltyDetector
- AttentionShift
"""

import numpy as np
import unittest
from unittest.mock import Mock

try:
    from ...core import NDAnalogField
    from .focus import Focus
    from .saliency import Saliency
    from .novelty_detector import NoveltyDetector
    from .attention_shift import AttentionShift
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.molecules.attention.focus import Focus
    from combinatronix.molecules.attention.saliency import Saliency
    from combinatronix.molecules.attention.novelty_detector import NoveltyDetector
    from combinatronix.molecules.attention.attention_shift import AttentionShift


class TestFocus(unittest.TestCase):
    """Test Focus molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.focus = Focus(location=(4, 4), strength=0.3, damping_threshold=0.8)
    
    def test_initialization(self):
        """Test focus initialization"""
        self.assertEqual(self.focus.focus_location, (4, 4))
        self.assertEqual(self.focus.attractor.strength, 0.3)
        self.assertEqual(self.focus.damper.threshold, 0.8)
    
    def test_apply_focus(self):
        """Test applying focus to field"""
        self.field.activation[3:5, 3:5] = 1.0
        
        result = self.focus.apply(self.field)
        
        self.assertIsNotNone(result)
        self.assertGreater(self.focus.focus_strength, 0)
        self.assertIsNotNone(self.focus.attention_map)
    
    def test_update_location(self):
        """Test updating focus location"""
        new_location = (6, 6)
        self.focus.update_location(new_location)
        
        self.assertEqual(self.focus.focus_location, new_location)
        self.assertEqual(self.focus.attractor.location, new_location)
        self.assertEqual(self.focus.focus_count, 1)
    
    def test_get_attention_map(self):
        """Test getting attention map"""
        self.field.activation[3:5, 3:5] = 1.0
        self.focus.apply(self.field)
        
        attention_map = self.focus.get_attention_map()
        
        self.assertIsNotNone(attention_map)
        self.assertEqual(attention_map.shape, self.field.shape)
    
    def test_get_focus_region(self):
        """Test getting focus region coordinates"""
        self.field.activation[3:5, 3:5] = 1.0
        self.focus.apply(self.field)
        
        focus_region = self.focus.get_focus_region()
        
        self.assertIsInstance(focus_region, list)
        self.assertGreater(len(focus_region), 0)
    
    def test_focus_statistics(self):
        """Test focus statistics"""
        self.field.activation[3:5, 3:5] = 1.0
        self.focus.apply(self.field)
        
        stats = self.focus.get_focus_statistics()
        
        self.assertIn('focus_location', stats)
        self.assertIn('focus_strength', stats)
        self.assertIn('background_damping', stats)
        self.assertEqual(stats['focus_location'], (4, 4))
    
    def test_shift_focus(self):
        """Test shifting focus to new location"""
        self.field.activation[3:5, 3:5] = 1.0
        self.focus.apply(self.field)
        
        new_location = (6, 6)
        result = self.focus.shift_focus(new_location, transition_steps=3)
        
        self.assertEqual(self.focus.focus_location, new_location)
    
    def test_is_focused_on(self):
        """Test checking if focus is on specific location"""
        self.field.activation[3:5, 3:5] = 1.0
        self.focus.apply(self.field)
        
        # Test focus location
        self.assertTrue(self.focus.is_focused_on((4, 4)))
        self.assertFalse(self.focus.is_focused_on((0, 0)))
    
    def test_reset(self):
        """Test focus reset"""
        self.field.activation[3:5, 3:5] = 1.0
        self.focus.apply(self.field)
        
        self.focus.reset()
        
        self.assertIsNone(self.focus.focus_location)
        self.assertEqual(self.focus.focus_count, 0)


class TestSaliency(unittest.TestCase):
    """Test Saliency molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.saliency = Saliency(amplification_gain=2.0, saliency_threshold=0.6)
    
    def test_initialization(self):
        """Test saliency initialization"""
        self.assertEqual(self.saliency.amplifier.gain, 2.0)
        self.assertEqual(self.saliency.threshold.threshold, 0.6)
        self.assertEqual(len(self.saliency.salient_regions), 0)
    
    def test_detect_saliency(self):
        """Test detecting salient regions"""
        self.field.activation[3:5, 3:5] = 0.8
        
        result = self.saliency.detect(self.field)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(self.saliency.saliency_map)
        self.assertGreater(self.saliency.total_processed, 0)
    
    def test_get_salient_regions(self):
        """Test getting salient regions"""
        self.field.activation[3:5, 3:5] = 0.8
        self.saliency.detect(self.field)
        
        regions = self.saliency.get_salient_regions()
        
        self.assertIsInstance(regions, list)
        # May or may not have regions depending on threshold
    
    def test_get_saliency_map(self):
        """Test getting saliency map"""
        self.field.activation[3:5, 3:5] = 0.8
        self.saliency.detect(self.field)
        
        saliency_map = self.saliency.get_saliency_map()
        
        self.assertIsNotNone(saliency_map)
        self.assertEqual(saliency_map.shape, self.field.shape)
    
    def test_saliency_statistics(self):
        """Test saliency statistics"""
        self.field.activation[3:5, 3:5] = 0.8
        self.saliency.detect(self.field)
        
        stats = self.saliency.get_saliency_statistics()
        
        self.assertIn('region_count', stats)
        self.assertIn('saliency_strength', stats)
        self.assertIn('total_processed', stats)
    
    def test_enhance_saliency(self):
        """Test enhancing salient regions"""
        self.field.activation[3:5, 3:5] = 0.8
        self.saliency.detect(self.field)
        
        original_energy = np.sum(np.abs(self.field.activation))
        enhanced = self.saliency.enhance_saliency(self.field, enhancement_factor=1.5)
        enhanced_energy = np.sum(np.abs(enhanced.activation))
        
        self.assertGreater(enhanced_energy, original_energy)
    
    def test_suppress_background(self):
        """Test suppressing background"""
        self.field.activation[3:5, 3:5] = 0.8
        self.saliency.detect(self.field)
        
        original_energy = np.sum(np.abs(self.field.activation))
        suppressed = self.saliency.suppress_background(self.field, suppression_factor=0.3)
        suppressed_energy = np.sum(np.abs(suppressed.activation))
        
        self.assertLess(suppressed_energy, original_energy)
    
    def test_reset(self):
        """Test saliency reset"""
        self.field.activation[3:5, 3:5] = 0.8
        self.saliency.detect(self.field)
        
        self.saliency.reset()
        
        self.assertIsNone(self.saliency.saliency_map)
        self.assertEqual(len(self.saliency.salient_regions), 0)


class TestNoveltyDetector(unittest.TestCase):
    """Test NoveltyDetector molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.detector = NoveltyDetector(sensitivity=0.3, novelty_threshold=0.5)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.memory_trace.threshold, 0.3)
        self.assertEqual(self.detector.novelty_threshold, 0.5)
        self.assertFalse(self.detector.baseline_established)
    
    def test_update_memory(self):
        """Test updating memory trace"""
        self.field.activation[3:5, 3:5] = 1.0
        
        result = self.detector.update_memory(self.field)
        
        self.assertTrue(self.detector.baseline_established)
        self.assertIsNotNone(self.detector.memory_trace.trace)
    
    def test_detect_novelty(self):
        """Test detecting novelty"""
        # First, establish baseline
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        
        # Create novel pattern
        self.field.activation[1:3, 1:3] = 0.8
        result = self.detector.detect_novelty(self.field)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(self.detector.novelty_map)
        self.assertGreater(self.detector.detection_count, 0)
    
    def test_get_novel_regions(self):
        """Test getting novel regions"""
        # Establish baseline
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        
        # Create novel pattern
        self.field.activation[1:3, 1:3] = 0.8
        self.detector.detect_novelty(self.field)
        
        regions = self.detector.get_novel_regions()
        
        self.assertIsInstance(regions, list)
    
    def test_get_novelty_map(self):
        """Test getting novelty map"""
        # Establish baseline
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        
        # Create novel pattern
        self.field.activation[1:3, 1:3] = 0.8
        self.detector.detect_novelty(self.field)
        
        novelty_map = self.detector.get_novelty_map()
        
        self.assertIsNotNone(novelty_map)
        self.assertEqual(novelty_map.shape, self.field.shape)
    
    def test_novelty_statistics(self):
        """Test novelty statistics"""
        # Establish baseline
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        
        # Create novel pattern
        self.field.activation[1:3, 1:3] = 0.8
        self.detector.detect_novelty(self.field)
        
        stats = self.detector.get_novelty_statistics()
        
        self.assertIn('region_count', stats)
        self.assertIn('novelty_strength', stats)
        self.assertIn('baseline_established', stats)
        self.assertTrue(stats['baseline_established'])
    
    def test_adapt_sensitivity(self):
        """Test adapting sensitivity"""
        # Establish baseline
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        
        # Create some novelty history
        for i in range(10):
            self.field.activation.fill(0)
            self.field.activation[i % 4, i % 4] = 0.8
            self.detector.detect_novelty(self.field)
        
        original_threshold = self.detector.memory_trace.threshold
        self.detector.adapt_sensitivity(adaptation_rate=0.1)
        
        # Threshold should have changed
        self.assertNotEqual(self.detector.memory_trace.threshold, original_threshold)
    
    def test_reset_memory(self):
        """Test resetting memory while keeping detector state"""
        # Establish baseline
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        
        self.detector.reset_memory()
        
        self.assertFalse(self.detector.baseline_established)
        self.assertIsNone(self.detector.memory_trace.trace)
    
    def test_reset(self):
        """Test complete reset"""
        # Establish baseline and detect novelty
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.update_memory(self.field)
        self.field.activation[1:3, 1:3] = 0.8
        self.detector.detect_novelty(self.field)
        
        self.detector.reset()
        
        self.assertFalse(self.detector.baseline_established)
        self.assertEqual(self.detector.detection_count, 0)


class TestAttentionShift(unittest.TestCase):
    """Test AttentionShift molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.shift = AttentionShift(gradient_strength=0.4, vortex_strength=0.2)
    
    def test_initialization(self):
        """Test shift initialization"""
        self.assertEqual(self.shift.gradient.strength, 0.4)
        self.assertEqual(self.shift.vortex.strength, 0.2)
        self.assertEqual(self.shift.shift_count, 0)
    
    def test_shift_attention(self):
        """Test shifting attention to target location"""
        self.field.activation[3:5, 3:5] = 1.0
        
        target_location = (6, 6)
        result = self.shift.shift_attention(self.field, target_location)
        
        self.assertIsNotNone(result)
        self.assertEqual(self.shift.target_focus, target_location)
        self.assertGreater(self.shift.shift_count, 0)
    
    def test_auto_detect_target(self):
        """Test auto-detecting attention target"""
        self.field.activation[3:5, 3:5] = 1.0
        
        result = self.shift.shift_attention(self.field)  # No target specified
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(self.shift.target_focus)
    
    def test_get_flow_field(self):
        """Test getting flow field"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6))
        
        flow_field = self.shift.get_flow_field()
        
        self.assertIsNotNone(flow_field)
        self.assertIn('flow_y', flow_field)
        self.assertIn('flow_x', flow_field)
        self.assertIn('center', flow_field)
    
    def test_get_transition_path(self):
        """Test getting transition path"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6), transition_steps=5)
        
        path = self.shift.get_transition_path()
        
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
    
    def test_get_attention_flow(self):
        """Test getting attention flow magnitude"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6))
        
        flow = self.shift.get_attention_flow(self.field)
        
        self.assertIsNotNone(flow)
        self.assertEqual(flow.shape, self.field.shape)
    
    def test_get_attention_direction(self):
        """Test getting attention flow direction"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6))
        
        direction = self.shift.get_attention_direction(self.field)
        
        self.assertIsNotNone(direction)
        self.assertEqual(direction.shape, self.field.shape)
    
    def test_shift_statistics(self):
        """Test shift statistics"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6))
        
        stats = self.shift.get_shift_statistics()
        
        self.assertIn('total_shifts', stats)
        self.assertIn('current_focus', stats)
        self.assertIn('target_focus', stats)
        self.assertGreater(stats['total_shifts'], 0)
    
    def test_detect_attention_conflicts(self):
        """Test detecting attention conflicts"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6))
        
        conflicts = self.shift.detect_attention_conflicts(self.field)
        
        self.assertIsInstance(conflicts, list)
    
    def test_smooth_attention_transition(self):
        """Test smooth attention transition"""
        self.field.activation[3:5, 3:5] = 1.0
        target_location = (6, 6)
        
        transition_fields = self.shift.smooth_attention_transition(
            self.field, target_location, steps=5
        )
        
        self.assertIsInstance(transition_fields, list)
        self.assertEqual(len(transition_fields), 6)  # 5 steps + 1 initial
    
    def test_reset(self):
        """Test attention shift reset"""
        self.field.activation[3:5, 3:5] = 1.0
        self.shift.shift_attention(self.field, (6, 6))
        
        self.shift.reset()
        
        self.assertIsNone(self.shift.current_focus)
        self.assertIsNone(self.shift.target_focus)
        self.assertEqual(self.shift.shift_count, 0)


class TestAttentionIntegration(unittest.TestCase):
    """Test integration between attention molecules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((10, 10))
        self.focus = Focus(location=(5, 5), strength=0.3)
        self.saliency = Saliency(amplification_gain=2.0, saliency_threshold=0.6)
        self.novelty = NoveltyDetector(sensitivity=0.3, novelty_threshold=0.5)
        self.shift = AttentionShift(gradient_strength=0.4, vortex_strength=0.2)
    
    def test_attention_pipeline(self):
        """Test complete attention pipeline"""
        # Create pattern
        self.field.activation[4:6, 4:6] = 1.0
        
        # Step 1: Detect saliency
        salient_field = self.saliency.detect(self.field)
        
        # Step 2: Focus on salient region
        focused_field = self.focus.apply(salient_field)
        
        # Step 3: Update novelty detector
        self.novelty.update_memory(focused_field)
        
        # Step 4: Shift attention
        shifted_field = self.shift.shift_attention(focused_field, (7, 7))
        
        # All should complete without error
        self.assertIsNotNone(salient_field)
        self.assertIsNotNone(focused_field)
        self.assertIsNotNone(shifted_field)
    
    def test_attention_coordination(self):
        """Test coordination between attention mechanisms"""
        # Create initial pattern
        self.field.activation[3:5, 3:5] = 1.0
        
        # Detect saliency and focus
        self.saliency.detect(self.field)
        self.focus.apply(self.field)
        
        # Check if focus is on salient region
        salient_regions = self.saliency.get_salient_regions()
        if salient_regions:
            region_center = salient_regions[0]['center']
            self.assertTrue(self.focus.is_focused_on(region_center, tolerance=2.0))
    
    def test_attention_adaptation(self):
        """Test adaptive attention based on novelty"""
        # Establish baseline
        self.field.activation[4:6, 4:6] = 1.0
        self.novelty.update_memory(self.field)
        
        # Create novel pattern
        self.field.activation[2:4, 2:4] = 0.8
        self.novelty.detect_novelty(self.field)
        
        # Focus should adapt to novel region
        novel_regions = self.novelty.get_novel_regions()
        if novel_regions:
            novel_center = novel_regions[0]['center']
            self.focus.update_location(novel_center)
            self.assertTrue(self.focus.is_focused_on(novel_center))


def create_test_field_with_pattern():
    """Helper function to create test field with pattern"""
    field = NDAnalogField((8, 8))
    field.activation[3:5, 3:5] = 1.0
    return field


def create_test_attention_field():
    """Helper function to create test field for attention"""
    field = NDAnalogField((10, 10))
    field.activation[4:6, 4:6] = 0.8
    field.activation[2:3, 2:3] = 0.6
    return field


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

