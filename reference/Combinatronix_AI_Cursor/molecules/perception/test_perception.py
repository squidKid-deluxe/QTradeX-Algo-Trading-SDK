# ============================================================================
# Perception Molecules Tests
# ============================================================================

"""
Comprehensive tests for perception molecules:
- EdgeDetector
- MotionDetector  
- PatternRecognizer
- ObjectTracker
"""

import numpy as np
import unittest
from unittest.mock import Mock

try:
    from ...core import NDAnalogField
    from .edge_detector import EdgeDetector
    from .motion_detector import MotionDetector
    from .pattern_recognizer import PatternRecognizer
    from .object_tracker import ObjectTracker
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.molecules.perception.edge_detector import EdgeDetector
    from combinatronix.molecules.perception.motion_detector import MotionDetector
    from combinatronix.molecules.perception.pattern_recognizer import PatternRecognizer
    from combinatronix.molecules.perception.object_tracker import ObjectTracker


class TestEdgeDetector(unittest.TestCase):
    """Test EdgeDetector molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.detector = EdgeDetector(threshold=0.3, strength=1.0)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.threshold.threshold, 0.3)
        self.assertEqual(self.detector.gradient.strength, 1.0)
        self.assertIsNone(self.detector.edge_map)
    
    def test_edge_detection(self):
        """Test basic edge detection"""
        # Create a simple edge pattern
        self.field.activation[3:5, 2:6] = 1.0  # Horizontal bar
        
        result = self.detector.process(self.field)
        
        # Should detect edges
        self.assertIsNotNone(self.detector.edge_map)
        self.assertGreater(self.detector.edge_count, 0)
        self.assertGreater(np.max(self.detector.edge_map), 0)
    
    def test_edge_statistics(self):
        """Test edge statistics computation"""
        # Create edge pattern
        self.field.activation[2:4, 2:4] = 1.0
        self.detector.process(self.field)
        
        stats = self.detector.get_edge_statistics()
        
        self.assertIn('edge_count', stats)
        self.assertIn('max_edge_strength', stats)
        self.assertIn('mean_edge_strength', stats)
        self.assertIn('edge_density', stats)
        self.assertGreaterEqual(stats['edge_density'], 0.0)
        self.assertLessEqual(stats['edge_density'], 1.0)
    
    def test_edge_enhancement(self):
        """Test edge enhancement functionality"""
        # Create edge pattern
        self.field.activation[2:4, 2:4] = 0.5
        self.detector.process(self.field)
        
        original_energy = np.sum(np.abs(self.field.activation))
        enhanced = self.detector.enhance_edges(self.field, enhancement_factor=1.5)
        enhanced_energy = np.sum(np.abs(enhanced.activation))
        
        self.assertGreater(enhanced_energy, original_energy)
    
    def test_reset(self):
        """Test detector reset"""
        self.field.activation[2:4, 2:4] = 1.0
        self.detector.process(self.field)
        
        self.detector.reset()
        
        self.assertIsNone(self.detector.edge_map)
        self.assertEqual(self.detector.edge_count, 0)


class TestMotionDetector(unittest.TestCase):
    """Test MotionDetector molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.detector = MotionDetector(sensitivity=0.2, memory_decay=0.9)
    
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.memory.threshold, 0.2)
        self.assertEqual(self.detector.memory.decay_rate, 0.9)
        self.assertFalse(self.detector.has_previous_frame)
    
    def test_first_frame(self):
        """Test processing first frame"""
        self.field.activation[3:5, 3:5] = 1.0
        
        result = self.detector.process(self.field)
        
        self.assertTrue(self.detector.has_previous_frame)
        self.assertEqual(self.detector.frame_count, 1)
        self.assertEqual(self.detector.motion_strength, 0.0)  # No motion yet
    
    def test_motion_detection(self):
        """Test motion detection between frames"""
        # First frame
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.process(self.field)
        
        # Second frame - moved object
        self.field.activation[4:6, 4:6] = 1.0  # Moved down and right
        result = self.detector.process(self.field)
        
        self.assertGreater(self.detector.motion_strength, 0)
        self.assertGreater(len(self.detector.motion_regions), 0)
    
    def test_motion_statistics(self):
        """Test motion statistics computation"""
        # Process two frames
        self.field.activation[2:4, 2:4] = 1.0
        self.detector.process(self.field)
        
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.process(self.field)
        
        stats = self.detector.get_motion_statistics()
        
        self.assertIn('motion_strength', stats)
        self.assertIn('motion_regions_count', stats)
        self.assertIn('frame_count', stats)
        self.assertEqual(stats['frame_count'], 2)
    
    def test_motion_events(self):
        """Test motion event detection"""
        # Create motion
        self.field.activation[2:4, 2:4] = 1.0
        self.detector.process(self.field)
        
        self.field.activation[3:5, 3:5] = 1.0
        self.detector.process(self.field)
        
        events = self.detector.detect_motion_events(event_threshold=0.1)
        
        # Should detect some motion events
        self.assertIsInstance(events, list)
    
    def test_reset(self):
        """Test detector reset"""
        self.field.activation[2:4, 2:4] = 1.0
        self.detector.process(self.field)
        
        self.detector.reset()
        
        self.assertFalse(self.detector.has_previous_frame)
        self.assertEqual(self.detector.frame_count, 0)
        self.assertIsNone(self.detector.motion_map)


class TestPatternRecognizer(unittest.TestCase):
    """Test PatternRecognizer molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((8, 8))
        self.recognizer = PatternRecognizer(recognition_threshold=0.7)
        
        # Add test templates
        circle_pattern = np.zeros((4, 4))
        circle_pattern[1:3, 1:3] = 1.0
        self.recognizer.add_template("circle", circle_pattern, "Simple circle")
        
        line_pattern = np.zeros((4, 4))
        line_pattern[1, :] = 1.0
        self.recognizer.add_template("line", line_pattern, "Horizontal line")
    
    def test_initialization(self):
        """Test recognizer initialization"""
        self.assertEqual(self.recognizer.resonator.threshold, 0.7)
        self.assertEqual(len(self.recognizer.templates), 2)
    
    def test_template_management(self):
        """Test template addition and removal"""
        # Test adding template
        square_pattern = np.ones((3, 3))
        self.recognizer.add_template("square", square_pattern)
        
        self.assertEqual(len(self.recognizer.templates), 3)
        self.assertIn("square", self.recognizer.templates)
        
        # Test removing template
        self.recognizer.remove_template("square")
        self.assertEqual(len(self.recognizer.templates), 2)
        self.assertNotIn("square", self.recognizer.templates)
    
    def test_pattern_recognition(self):
        """Test pattern recognition"""
        # Create a pattern similar to circle template
        self.field.activation[2:4, 2:4] = 0.8
        
        result = self.recognizer.recognize(self.field)
        
        # Should recognize the pattern
        self.assertGreater(len(self.recognizer.current_matches), 0)
        self.assertGreater(self.recognizer.get_recognition_confidence(), 0)
    
    def test_template_statistics(self):
        """Test template statistics"""
        # Recognize a pattern
        self.field.activation[2:4, 2:4] = 0.8
        self.recognizer.recognize(self.field)
        
        stats = self.recognizer.get_template_statistics()
        
        self.assertIn("circle", stats)
        self.assertIn("line", stats)
        self.assertIn("recognition_count", stats["circle"])
        self.assertIn("pattern_shape", stats["circle"])
    
    def test_similar_patterns(self):
        """Test finding similar patterns"""
        test_pattern = np.zeros((4, 4))
        test_pattern[1:3, 1:3] = 0.9  # Similar to circle
        
        similar = self.recognizer.find_similar_patterns(test_pattern, similarity_threshold=0.5)
        
        self.assertGreater(len(similar), 0)
        self.assertEqual(similar[0]["name"], "circle")  # Should match circle best
    
    def test_reset(self):
        """Test recognizer reset"""
        self.field.activation[2:4, 2:4] = 0.8
        self.recognizer.recognize(self.field)
        
        self.recognizer.reset()
        
        self.assertEqual(len(self.recognizer.current_matches), 0)
        self.assertEqual(self.recognizer.recognition_count, 0)


class TestObjectTracker(unittest.TestCase):
    """Test ObjectTracker molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((10, 10))
        self.tracker = ObjectTracker(max_objects=5, tracking_radius=2)
    
    def test_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(self.tracker.max_objects, 5)
        self.assertEqual(self.tracker.tracking_radius, 2)
        self.assertEqual(len(self.tracker.objects), 0)
    
    def test_add_object(self):
        """Test adding objects to track"""
        self.tracker.add_object("obj1", (5, 5), confidence=0.8)
        
        self.assertEqual(len(self.tracker.objects), 1)
        self.assertIn("obj1", self.tracker.objects)
        self.assertEqual(self.tracker.objects["obj1"]["location"], (5, 5))
        self.assertEqual(self.tracker.objects["obj1"]["confidence"], 0.8)
    
    def test_object_tracking(self):
        """Test object tracking across frames"""
        # Add object
        self.tracker.add_object("obj1", (5, 5), confidence=0.8)
        
        # Update tracking
        self.field.activation[5, 5] = 1.0
        result = self.tracker.update(self.field)
        
        self.assertEqual(self.tracker.frame_count, 1)
        self.assertIsNotNone(self.tracker.tracking_field)
    
    def test_object_positions(self):
        """Test getting object positions"""
        self.tracker.add_object("obj1", (3, 4), confidence=0.9)
        self.tracker.add_object("obj2", (7, 2), confidence=0.7)
        
        positions = self.tracker.get_object_positions()
        
        self.assertEqual(positions["obj1"], (3, 4))
        self.assertEqual(positions["obj2"], (7, 2))
    
    def test_object_trajectories(self):
        """Test getting object trajectories"""
        self.tracker.add_object("obj1", (5, 5), confidence=0.8)
        
        # Update several times
        for i in range(3):
            self.field.activation[5 + i, 5 + i] = 1.0
            self.tracker.update(self.field)
        
        trajectories = self.tracker.get_object_trajectories()
        
        self.assertIn("obj1", trajectories)
        self.assertGreater(len(trajectories["obj1"]), 1)
    
    def test_collision_detection(self):
        """Test collision detection"""
        self.tracker.add_object("obj1", (5, 5), confidence=0.8)
        self.tracker.add_object("obj2", (6, 6), confidence=0.8)
        
        collisions = self.tracker.detect_collisions(collision_distance=2.0)
        
        self.assertGreater(len(collisions), 0)
        self.assertIn("obj1", collisions[0]["object1"])
        self.assertIn("obj2", collisions[0]["object2"])
    
    def test_cleanup_lost_objects(self):
        """Test cleanup of lost objects"""
        self.tracker.add_object("obj1", (5, 5), confidence=0.8)
        
        # Simulate many frames without updates
        for i in range(15):
            self.tracker.frame_count += 1
        
        lost_count = self.tracker.cleanup_lost_objects(max_frames_missing=10)
        
        self.assertEqual(lost_count, 1)
        self.assertEqual(len(self.tracker.objects), 0)
    
    def test_reset(self):
        """Test tracker reset"""
        self.tracker.add_object("obj1", (5, 5), confidence=0.8)
        self.tracker.update(self.field)
        
        self.tracker.reset()
        
        self.assertEqual(len(self.tracker.objects), 0)
        self.assertEqual(self.tracker.frame_count, 0)
        self.assertIsNone(self.tracker.tracking_field)


class TestPerceptionIntegration(unittest.TestCase):
    """Test integration between perception molecules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.field = NDAnalogField((12, 12))
        self.edge_detector = EdgeDetector(threshold=0.3)
        self.motion_detector = MotionDetector(sensitivity=0.2)
        self.pattern_recognizer = PatternRecognizer(recognition_threshold=0.6)
        self.object_tracker = ObjectTracker(max_objects=3)
    
    def test_perception_pipeline(self):
        """Test complete perception pipeline"""
        # Create a moving object pattern
        self.field.activation[4:6, 4:6] = 1.0
        
        # Step 1: Detect edges
        edge_result = self.edge_detector.process(self.field)
        
        # Step 2: Detect motion
        motion_result = self.motion_detector.process(self.field)
        
        # Step 3: Recognize patterns
        pattern_result = self.pattern_recognizer.recognize(self.field)
        
        # Step 4: Track objects
        self.object_tracker.add_object("moving_obj", (5, 5), confidence=0.8)
        tracking_result = self.object_tracker.update(self.field)
        
        # All should complete without error
        self.assertIsNotNone(edge_result)
        self.assertIsNotNone(motion_result)
        self.assertIsNotNone(pattern_result)
        self.assertIsNotNone(tracking_result)
    
    def test_multi_frame_processing(self):
        """Test processing multiple frames"""
        # Frame 1
        self.field.activation[3:5, 3:5] = 1.0
        self.motion_detector.process(self.field)
        
        # Frame 2 - object moved
        self.field.activation[4:6, 4:6] = 1.0
        motion_result = self.motion_detector.process(self.field)
        
        # Should detect motion
        self.assertGreater(self.motion_detector.motion_strength, 0)
        
        # Add object and track
        self.object_tracker.add_object("moving_obj", (4, 4), confidence=0.8)
        tracking_result = self.object_tracker.update(self.field)
        
        self.assertIsNotNone(tracking_result)


def create_test_field_with_edges():
    """Helper function to create test field with edges"""
    field = NDAnalogField((8, 8))
    field.activation[2:4, 2:6] = 1.0  # Horizontal bar
    field.activation[4:6, 3:5] = 1.0  # Vertical bar
    return field


def create_test_pattern():
    """Helper function to create test pattern"""
    field = NDAnalogField((6, 6))
    field.activation[2:4, 2:4] = 0.8
    return field


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

