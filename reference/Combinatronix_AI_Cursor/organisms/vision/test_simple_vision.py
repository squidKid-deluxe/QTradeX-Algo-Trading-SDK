# ============================================================================
# SimpleVision Test Suite
# ============================================================================

"""
Test suite for SimpleVision organism
"""

import numpy as np
import unittest

try:
    from .simple_vision import SimpleVision, FeatureType, Detection
except ImportError:
    from combinatronix.organisms.vision.simple_vision import SimpleVision, FeatureType, Detection


class TestSimpleVision(unittest.TestCase):
    """Test SimpleVision organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.vision = SimpleVision({'detection_threshold': 0.3, 'enable_visualization': False})
    
    def test_initialization(self):
        """Test vision system initialization"""
        self.assertIsNotNone(self.vision.atoms)
        self.assertIsNotNone(self.vision.molecules)
        self.assertIsNotNone(self.vision.fields)
        self.assertEqual(len(self.vision.atoms), 6)
        self.assertEqual(len(self.vision.molecules), 3)
    
    def test_process_small_image(self):
        """Test processing small image"""
        # Create 4x4 test image
        image = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        detections = self.vision.process(image)
        
        self.assertIsInstance(detections, dict)
        self.assertIn(FeatureType.HORIZONTAL_LINE, detections)
        self.assertIn(FeatureType.BRIGHTNESS, detections)
    
    def test_process_large_image(self):
        """Test processing large image"""
        # Create 32x32 test image
        image = np.zeros((32, 32))
        image[10:20, 10:30] = 1.0  # Horizontal line
        
        detections = self.vision.process(image)
        
        self.assertIsInstance(detections, dict)
        self.assertGreater(len(detections[FeatureType.HORIZONTAL_LINE]), 0)
    
    def test_edge_detection(self):
        """Test edge detection"""
        # Create edge pattern
        image = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        
        detections = self.vision.process(image)
        
        self.assertIn(FeatureType.EDGE, detections)
        self.assertGreater(len(detections[FeatureType.EDGE]), 0)
    
    def test_motion_detection(self):
        """Test motion detection"""
        # First frame
        frame1 = np.zeros((8, 8))
        frame1[4, 4] = 1.0
        
        detections1 = self.vision.process(frame1)
        
        # Second frame with moved object
        frame2 = np.zeros((8, 8))
        frame2[4, 5] = 1.0
        
        detections2 = self.vision.process(frame2)
        
        # Should detect motion in second frame
        self.assertIn(FeatureType.MOTION, detections2)
    
    def test_brightness_detection(self):
        """Test brightness detection"""
        # Create bright region
        image = np.zeros((6, 6))
        image[2:4, 2:4] = 0.8
        
        detections = self.vision.process(image)
        
        self.assertIn(FeatureType.BRIGHTNESS, detections)
        self.assertGreater(len(detections[FeatureType.BRIGHTNESS]), 0)
    
    def test_line_detection(self):
        """Test line detection"""
        # Horizontal line
        h_image = np.zeros((6, 6))
        h_image[2, :] = 1.0
        
        h_detections = self.vision.process(h_image)
        self.assertGreater(len(h_detections[FeatureType.HORIZONTAL_LINE]), 0)
        
        # Vertical line
        v_image = np.zeros((6, 6))
        v_image[:, 2] = 1.0
        
        v_detections = self.vision.process(v_image)
        self.assertGreater(len(v_detections[FeatureType.VERTICAL_LINE]), 0)
        
        # Diagonal line
        d_image = np.zeros((6, 6))
        for i in range(6):
            d_image[i, i] = 1.0
        
        d_detections = self.vision.process(d_image)
        self.assertGreater(len(d_detections[FeatureType.DIAGONAL_LINE]), 0)
    
    def test_get_summary(self):
        """Test summary generation"""
        image = np.zeros((4, 4))
        image[1, :] = 1.0  # Horizontal line
        
        detections = self.vision.process(image)
        summary = self.vision.get_summary(detections)
        
        self.assertIsInstance(summary, str)
        self.assertIn("horizontal", summary.lower())
    
    def test_get_state(self):
        """Test state retrieval"""
        image = np.zeros((4, 4))
        self.vision.process(image)
        
        state = self.vision.get_state()
        
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test system reset"""
        # Process some images
        image1 = np.zeros((4, 4))
        image2 = np.zeros((4, 4))
        
        self.vision.process(image1)
        self.vision.process(image2)
        
        # Reset
        self.vision.reset()
        
        # Check that state is reset
        self.assertEqual(self.vision.state['processed_frames'], 0)
        self.assertIsNone(self.vision.state['current_frame_shape'])
    
    def test_adaptation(self):
        """Test adaptation without backprop"""
        # Create training data
        training_data = [
            np.zeros((4, 4)),
            np.ones((4, 4)),
            np.array([[1, 0], [0, 1]] * 2)
        ]
        
        # Train system
        self.vision.train(training_data, epochs=2)
        
        # Should complete without error
        self.assertTrue(True)
    
    def test_different_image_sizes(self):
        """Test processing images of different sizes"""
        sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]
        
        for h, w in sizes:
            image = np.zeros((h, w))
            image[h//2, :] = 1.0  # Horizontal line
            
            detections = self.vision.process(image)
            
            self.assertIsInstance(detections, dict)
            self.assertIn(FeatureType.HORIZONTAL_LINE, detections)


if __name__ == '__main__':
    unittest.main(verbosity=2)

