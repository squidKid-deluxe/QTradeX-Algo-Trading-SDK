# ============================================================================
# Emotion Molecules Tests
# ============================================================================

"""
Comprehensive tests for emotion molecules:
- MoodRegulator
- EmotionalMemory
- EmpathySimulator
- EmotionalAmplifier
"""

import numpy as np
import unittest
from unittest.mock import Mock

try:
    from ...core import NDAnalogField
    from .mood_regulator import MoodRegulator
    from .emotional_memory import EmotionalMemory
    from .empathy_simulator import EmpathySimulator
    from .emotional_amplifier import EmotionalAmplifier
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.molecules.emotion.mood_regulator import MoodRegulator
    from combinatronix.molecules.emotion.emotional_memory import EmotionalMemory
    from combinatronix.molecules.emotion.empathy_simulator import EmpathySimulator
    from combinatronix.molecules.emotion.emotional_amplifier import EmotionalAmplifier


class TestMoodRegulator(unittest.TestCase):
    """Test MoodRegulator molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mood_regulator = MoodRegulator(equilibrium_rate=0.3, damping_threshold=0.8)
        self.emotional_field = NDAnalogField((8, 8))
        self.emotional_field.activation[2:4, 2:4] = 0.9
        self.emotional_field.activation[6:8, 6:8] = 0.7
    
    def test_initialization(self):
        """Test mood regulator initialization"""
        self.assertEqual(self.mood_regulator.balancer.equilibrium_rate, 0.3)
        self.assertEqual(self.mood_regulator.damper.threshold, 0.8)
        self.assertEqual(len(self.mood_regulator.mood_history), 0)
    
    def test_regulate_mood(self):
        """Test regulating mood in emotional field"""
        regulated_field = self.mood_regulator.regulate_mood(self.emotional_field)
        
        self.assertIsNotNone(regulated_field)
        self.assertGreater(len(self.mood_regulator.mood_history), 0)
        self.assertIsNotNone(self.mood_regulator.current_mood)
    
    def test_regulate_mood_with_target(self):
        """Test regulating mood with target mood"""
        regulated_field = self.mood_regulator.regulate_mood(
            self.emotional_field, target_mood="calm"
        )
        
        self.assertIsNotNone(regulated_field)
        self.assertEqual(self.mood_regulator.mood_history[-1]['target_mood'], "calm")
    
    def test_get_mood_state(self):
        """Test getting current mood state"""
        self.mood_regulator.regulate_mood(self.emotional_field)
        
        mood_state = self.mood_regulator.get_mood_state()
        
        self.assertIn('current_mood', mood_state)
        self.assertIn('emotional_energy', mood_state)
        self.assertIn('mood_stability', mood_state)
        self.assertIsNotNone(mood_state['current_mood'])
    
    def test_get_mood_history(self):
        """Test getting mood history"""
        # Create some mood history
        for i in range(5):
            field = NDAnalogField((8, 8))
            field.activation[i:i+2, i:i+2] = 0.8
            self.mood_regulator.regulate_mood(field)
        
        history = self.mood_regulator.get_mood_history(window_size=3)
        
        self.assertIsInstance(history, list)
        self.assertLessEqual(len(history), 3)
    
    def test_get_mood_transitions(self):
        """Test getting mood transitions"""
        # Create mood transitions
        field1 = NDAnalogField((8, 8))
        field1.activation[1:3, 1:3] = 0.2  # Low energy - serene
        self.mood_regulator.regulate_mood(field1)
        
        field2 = NDAnalogField((8, 8))
        field2.activation[1:3, 1:3] = 0.9  # High energy - excited
        self.mood_regulator.regulate_mood(field2)
        
        transitions = self.mood_regulator.get_mood_transitions()
        
        self.assertIsInstance(transitions, list)
        if transitions:  # May or may not have transitions depending on mood detection
            self.assertIn('from', transitions[0])
            self.assertIn('to', transitions[0])
    
    def test_get_mood_statistics(self):
        """Test getting mood statistics"""
        # Create some mood history
        for i in range(5):
            field = NDAnalogField((8, 8))
            field.activation[i:i+2, i:i+2] = 0.5
            self.mood_regulator.regulate_mood(field)
        
        stats = self.mood_regulator.get_mood_statistics()
        
        self.assertIn('total_regulations', stats)
        self.assertIn('mood_diversity', stats)
        self.assertIn('average_stability', stats)
        self.assertGreater(stats['total_regulations'], 0)
    
    def test_detect_emotional_instability(self):
        """Test detecting emotional instability"""
        # Create unstable mood pattern
        for i in range(10):
            field = NDAnalogField((8, 8))
            field.activation[i % 4:i % 4 + 2, i % 4:i % 4 + 2] = 0.8
            self.mood_regulator.regulate_mood(field)
        
        is_unstable = self.mood_regulator.detect_emotional_instability()
        
        self.assertIsInstance(is_unstable, bool)
    
    def test_stabilize_mood(self):
        """Test stabilizing unstable mood"""
        # Create unstable mood
        for i in range(10):
            field = NDAnalogField((8, 8))
            field.activation[i % 4:i % 4 + 2, i % 4:i % 4 + 2] = 0.8
            self.mood_regulator.regulate_mood(field)
        
        field = NDAnalogField((8, 8))
        field.activation[2:4, 2:4] = 0.9
        
        stabilized_field = self.mood_regulator.stabilize_mood(field)
        
        self.assertIsNotNone(stabilized_field)
    
    def test_get_emotional_energy(self):
        """Test getting emotional energy"""
        self.mood_regulator.regulate_mood(self.emotional_field)
        
        energy = self.mood_regulator.get_emotional_energy()
        
        self.assertIsInstance(energy, float)
        self.assertGreaterEqual(energy, 0.0)
    
    def test_get_mood_trend(self):
        """Test getting mood trend"""
        # Create trend data
        for i in range(8):
            field = NDAnalogField((8, 8))
            field.activation[1:3, 1:3] = 0.1 + i * 0.1
            self.mood_regulator.regulate_mood(field)
        
        trend = self.mood_regulator.get_mood_trend()
        
        self.assertIn(trend, ['stable', 'increasing', 'decreasing', 'volatile', 'insufficient_data'])
    
    def test_reset(self):
        """Test mood regulator reset"""
        # Create some data
        self.mood_regulator.regulate_mood(self.emotional_field)
        
        self.mood_regulator.reset()
        
        self.assertEqual(len(self.mood_regulator.mood_history), 0)
        self.assertIsNone(self.mood_regulator.current_mood)


class TestEmotionalMemory(unittest.TestCase):
    """Test EmotionalMemory molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotional_memory = EmotionalMemory(accumulation_rate=0.2, resonance_threshold=0.6)
        self.emotional_field = NDAnalogField((8, 8))
        self.emotional_field.activation[2:4, 2:4] = 0.8
    
    def test_initialization(self):
        """Test emotional memory initialization"""
        self.assertEqual(self.emotional_memory.memory_trace.accumulation_rate, 0.2)
        self.assertEqual(self.emotional_memory.resonator.threshold, 0.6)
        self.assertEqual(len(self.emotional_memory.emotional_experiences), 0)
    
    def test_store_experience(self):
        """Test storing emotional experience"""
        experience_id = self.emotional_memory.store_experience(
            self.emotional_field, "joy", context="birthday", intensity=0.8
        )
        
        self.assertIsNotNone(experience_id)
        self.assertIn(experience_id, self.emotional_memory.emotional_experiences)
        self.assertEqual(self.emotional_memory.emotional_experiences[experience_id]['emotion'], "joy")
    
    def test_recall_by_emotion(self):
        """Test recalling experiences by emotion"""
        # Store some experiences
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        self.emotional_memory.store_experience(self.emotional_field, "sadness", "funeral")
        
        recalled = self.emotional_memory.recall_by_emotion("joy")
        
        self.assertIsInstance(recalled, list)
        self.assertGreater(len(recalled), 0)
        self.assertEqual(recalled[0]['emotion'], "joy")
    
    def test_recall_by_context(self):
        """Test recalling experiences by context"""
        # Store some experiences
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        self.emotional_memory.store_experience(self.emotional_field, "sadness", "funeral")
        
        recalled = self.emotional_memory.recall_by_context("birthday")
        
        self.assertIsInstance(recalled, list)
        self.assertGreater(len(recalled), 0)
        self.assertEqual(recalled[0]['context'], "birthday")
    
    def test_find_similar_emotions(self):
        """Test finding similar emotions using resonance"""
        # Store some experiences
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        
        query_field = NDAnalogField((8, 8))
        query_field.activation[2:4, 2:4] = 0.7  # Similar pattern
        
        similar = self.emotional_memory.find_similar_emotions(query_field)
        
        self.assertIsInstance(similar, list)
    
    def test_recall_with_resonance(self):
        """Test recalling with resonance"""
        # Store some experiences
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        
        query_field = NDAnalogField((8, 8))
        query_field.activation[2:4, 2:4] = 0.7
        
        recalled = self.emotional_memory.recall_with_resonance(query_field, emotion_filter="joy")
        
        self.assertIsInstance(recalled, list)
    
    def test_get_emotional_pattern(self):
        """Test getting emotional pattern for emotion type"""
        # Store multiple experiences of same emotion
        for i in range(3):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.8
            self.emotional_memory.store_experience(field, "joy", f"event_{i}")
        
        pattern = self.emotional_memory.get_emotional_pattern("joy")
        
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.shape, (8, 8))
    
    def test_get_emotion_statistics(self):
        """Test getting emotion statistics"""
        # Store some experiences
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        self.emotional_memory.store_experience(self.emotional_field, "sadness", "funeral")
        
        stats = self.emotional_memory.get_emotion_statistics()
        
        self.assertIn('total_experiences', stats)
        self.assertIn('emotion_types', stats)
        self.assertIn('context_types', stats)
        self.assertGreater(stats['total_experiences'], 0)
    
    def test_get_experience_by_id(self):
        """Test getting experience by ID"""
        experience_id = self.emotional_memory.store_experience(
            self.emotional_field, "joy", "birthday"
        )
        
        experience = self.emotional_memory.get_experience_by_id(experience_id)
        
        self.assertIsNotNone(experience)
        self.assertEqual(experience['emotion'], "joy")
    
    def test_update_experience_intensity(self):
        """Test updating experience intensity"""
        experience_id = self.emotional_memory.store_experience(
            self.emotional_field, "joy", "birthday", intensity=0.5
        )
        
        self.emotional_memory.update_experience_intensity(experience_id, 0.8)
        
        updated_experience = self.emotional_memory.get_experience_by_id(experience_id)
        self.assertEqual(updated_experience['intensity'], 0.8)
    
    def test_get_emotional_timeline(self):
        """Test getting emotional timeline"""
        # Store some experiences
        for i in range(5):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.8
            self.emotional_memory.store_experience(field, "joy", f"event_{i}")
        
        timeline = self.emotional_memory.get_emotional_timeline("joy", window_size=3)
        
        self.assertIsInstance(timeline, list)
        self.assertLessEqual(len(timeline), 3)
    
    def test_reset(self):
        """Test emotional memory reset"""
        # Store some data
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        
        self.emotional_memory.reset()
        
        self.assertEqual(len(self.emotional_memory.emotional_experiences), 0)
        self.assertEqual(len(self.emotional_memory.emotion_index), 0)


class TestEmpathySimulator(unittest.TestCase):
    """Test EmpathySimulator molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.empathy_sim = EmpathySimulator(resonance_threshold=0.7, translation_strength=0.8)
        self.others_emotion_field = NDAnalogField((8, 8))
        self.others_emotion_field.activation[2:4, 2:4] = 0.9
    
    def test_initialization(self):
        """Test empathy simulator initialization"""
        self.assertEqual(self.empathy_sim.resonator.threshold, 0.7)
        self.assertEqual(self.empathy_sim.translator.scale_factor, 1.0)
        self.assertEqual(len(self.empathy_sim.empathy_responses), 0)
    
    def test_simulate_empathy(self):
        """Test simulating empathy"""
        empathetic_response = self.empathy_sim.simulate_empathy(
            self.others_emotion_field, context="test", empathy_type="emotional"
        )
        
        self.assertIsNotNone(empathetic_response)
        self.assertGreater(len(self.empathy_sim.empathy_responses), 0)
    
    def test_simulate_empathy_different_types(self):
        """Test simulating different types of empathy"""
        empathy_types = ['cognitive', 'emotional', 'compassionate']
        
        for emp_type in empathy_types:
            response = self.empathy_sim.simulate_empathy(
                self.others_emotion_field, empathy_type=emp_type
            )
            self.assertIsNotNone(response)
    
    def test_create_emotional_mirror(self):
        """Test creating emotional mirror"""
        mirror_id = self.empathy_sim.create_emotional_mirror(
            self.others_emotion_field, "test_mirror"
        )
        
        self.assertIsNotNone(mirror_id)
        self.assertIn(mirror_id, self.empathy_sim.emotional_mirrors)
        self.assertEqual(self.empathy_sim.emotional_mirrors[mirror_id]['name'], "test_mirror")
    
    def test_update_emotional_mirror(self):
        """Test updating emotional mirror"""
        mirror_id = self.empathy_sim.create_emotional_mirror(self.others_emotion_field)
        
        current_field = NDAnalogField((8, 8))
        current_field.activation[3:5, 3:5] = 0.8
        
        result = self.empathy_sim.update_emotional_mirror(mirror_id, current_field)
        
        self.assertTrue(result)
        self.assertGreater(self.empathy_sim.emotional_mirrors[mirror_id]['usage_count'], 0)
    
    def test_get_empathy_level(self):
        """Test getting empathy level"""
        self.empathy_sim.simulate_empathy(self.others_emotion_field)
        
        empathy_level = self.empathy_sim.get_empathy_level()
        
        self.assertIsInstance(empathy_level, float)
        self.assertGreaterEqual(empathy_level, 0.0)
    
    def test_get_empathy_statistics(self):
        """Test getting empathy statistics"""
        # Create some empathy responses
        for i in range(3):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.8
            self.empathy_sim.simulate_empathy(field, empathy_type="emotional")
        
        stats = self.empathy_sim.get_empathy_statistics()
        
        self.assertIn('total_responses', stats)
        self.assertIn('average_empathy_level', stats)
        self.assertIn('empathy_types', stats)
        self.assertGreater(stats['total_responses'], 0)
    
    def test_get_emotional_mirror(self):
        """Test getting emotional mirror by ID"""
        mirror_id = self.empathy_sim.create_emotional_mirror(
            self.others_emotion_field, "test_mirror"
        )
        
        mirror = self.empathy_sim.get_emotional_mirror(mirror_id)
        
        self.assertIsNotNone(mirror)
        self.assertEqual(mirror['name'], "test_mirror")
    
    def test_get_empathy_history(self):
        """Test getting empathy history"""
        # Create some empathy responses
        for i in range(5):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.8
            self.empathy_sim.simulate_empathy(field)
        
        history = self.empathy_sim.get_empathy_history(window_size=3)
        
        self.assertIsInstance(history, list)
        self.assertLessEqual(len(history), 3)
    
    def test_detect_empathy_fatigue(self):
        """Test detecting empathy fatigue"""
        # Create low empathy responses
        for i in range(5):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.1  # Low energy
            self.empathy_sim.simulate_empathy(field)
        
        is_fatigued = self.empathy_sim.detect_empathy_fatigue()
        
        self.assertIsInstance(is_fatigued, bool)
    
    def test_restore_empathy(self):
        """Test restoring empathy"""
        original_level = self.empathy_sim.get_empathy_level()
        
        new_level = self.empathy_sim.restore_empathy(0.3)
        
        self.assertGreaterEqual(new_level, original_level)
    
    def test_get_empathy_trend(self):
        """Test getting empathy trend"""
        # Create trend data
        for i in range(8):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.1 + i * 0.1
            self.empathy_sim.simulate_empathy(field)
        
        trend = self.empathy_sim.get_empathy_trend()
        
        self.assertIn(trend, ['increasing', 'decreasing', 'stable', 'volatile', 'insufficient_data'])
    
    def test_reset(self):
        """Test empathy simulator reset"""
        # Create some data
        self.empathy_sim.simulate_empathy(self.others_emotion_field)
        
        self.empathy_sim.reset()
        
        self.assertEqual(len(self.empathy_sim.empathy_responses), 0)
        self.assertEqual(len(self.empathy_sim.emotional_mirrors), 0)


class TestEmotionalAmplifier(unittest.TestCase):
    """Test EmotionalAmplifier molecule"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.emotional_amp = EmotionalAmplifier(amplification_gain=2.0, vortex_strength=0.3)
        self.emotional_field = NDAnalogField((8, 8))
        self.emotional_field.activation[2:4, 2:4] = 0.5
    
    def test_initialization(self):
        """Test emotional amplifier initialization"""
        self.assertEqual(self.emotional_amp.amplifier.gain, 2.0)
        self.assertEqual(self.emotional_amp.vortex.strength, 0.3)
        self.assertEqual(len(self.emotional_amp.amplification_history), 0)
    
    def test_amplify_emotion(self):
        """Test amplifying emotion"""
        amplified_field = self.emotional_amp.amplify_emotion(
            self.emotional_field, intensity_level=1.5
        )
        
        self.assertIsNotNone(amplified_field)
        self.assertGreater(len(self.emotional_amp.amplification_history), 0)
        self.assertGreater(self.emotional_amp.emotional_intensity, 0.0)
    
    def test_create_dynamic_emotion(self):
        """Test creating dynamic emotion"""
        dynamic_fields = self.emotional_amp.create_dynamic_emotion(
            self.emotional_field, emotion_type="dynamic", duration=5
        )
        
        self.assertIsInstance(dynamic_fields, list)
        self.assertEqual(len(dynamic_fields), 5)
        self.assertGreater(len(self.emotional_amp.dynamic_patterns), 0)
    
    def test_get_emotional_intensity(self):
        """Test getting emotional intensity"""
        self.emotional_amp.amplify_emotion(self.emotional_field)
        
        intensity = self.emotional_amp.get_emotional_intensity()
        
        self.assertIsInstance(intensity, float)
        self.assertGreaterEqual(intensity, 0.0)
        self.assertLessEqual(intensity, 1.0)
    
    def test_get_intensity_trend(self):
        """Test getting intensity trend"""
        # Create trend data
        for i in range(8):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.1 + i * 0.1
            self.emotional_amp.amplify_emotion(field)
        
        trend = self.emotional_amp.get_intensity_trend()
        
        self.assertIn(trend, ['increasing', 'decreasing', 'stable', 'volatile', 'insufficient_data'])
    
    def test_get_amplification_statistics(self):
        """Test getting amplification statistics"""
        # Create some amplifications
        for i in range(3):
            field = NDAnalogField((8, 8))
            field.activation[2:4, 2:4] = 0.5
            self.emotional_amp.amplify_emotion(field)
        
        stats = self.emotional_amp.get_amplification_statistics()
        
        self.assertIn('total_amplifications', stats)
        self.assertIn('average_intensity', stats)
        self.assertIn('average_energy_increase', stats)
        self.assertGreater(stats['total_amplifications'], 0)
    
    def test_get_dynamic_pattern(self):
        """Test getting dynamic pattern by ID"""
        dynamic_fields = self.emotional_amp.create_dynamic_emotion(
            self.emotional_field, emotion_type="test", duration=3
        )
        
        pattern_id = list(self.emotional_amp.dynamic_patterns.keys())[0]
        pattern = self.emotional_amp.get_dynamic_pattern(pattern_id)
        
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern['emotion_type'], "test")
    
    def test_get_vortex_flow_field(self):
        """Test getting vortex flow field"""
        self.emotional_amp.amplify_emotion(self.emotional_field)
        
        flow_y, flow_x = self.emotional_amp.get_vortex_flow_field(self.emotional_field)
        
        if flow_y is not None and flow_x is not None:
            self.assertEqual(flow_y.shape, self.emotional_field.shape)
            self.assertEqual(flow_x.shape, self.emotional_field.shape)
    
    def test_create_emotional_cascade(self):
        """Test creating emotional cascade"""
        cascade_field = self.emotional_amp.create_emotional_cascade(
            self.emotional_field, cascade_strength=0.8
        )
        
        self.assertIsNotNone(cascade_field)
        self.assertEqual(cascade_field.shape, self.emotional_field.shape)
    
    def test_detect_emotional_overload(self):
        """Test detecting emotional overload"""
        # Create high intensity field
        high_intensity_field = NDAnalogField((8, 8))
        high_intensity_field.activation.fill(0.9)
        
        self.emotional_amp.amplify_emotion(high_intensity_field, intensity_level=2.0)
        
        is_overloaded = self.emotional_amp.detect_emotional_overload()
        
        self.assertIsInstance(is_overloaded, bool)
    
    def test_dampen_emotion(self):
        """Test dampening emotion"""
        self.emotional_amp.amplify_emotion(self.emotional_field)
        original_intensity = self.emotional_amp.get_emotional_intensity()
        
        dampened_field = self.emotional_amp.dampen_emotion(self.emotional_field, 0.5)
        
        self.assertIsNotNone(dampened_field)
        self.assertLess(self.emotional_amp.get_emotional_intensity(), original_intensity)
    
    def test_reset(self):
        """Test emotional amplifier reset"""
        # Create some data
        self.emotional_amp.amplify_emotion(self.emotional_field)
        
        self.emotional_amp.reset()
        
        self.assertEqual(len(self.emotional_amp.amplification_history), 0)
        self.assertEqual(self.emotional_amp.emotional_intensity, 0.0)


class TestEmotionIntegration(unittest.TestCase):
    """Test integration between emotion molecules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mood_regulator = MoodRegulator()
        self.emotional_memory = EmotionalMemory()
        self.empathy_sim = EmpathySimulator()
        self.emotional_amp = EmotionalAmplifier()
        
        self.emotional_field = NDAnalogField((8, 8))
        self.emotional_field.activation[2:4, 2:4] = 0.8
    
    def test_emotion_pipeline(self):
        """Test complete emotion processing pipeline"""
        # Step 1: Store emotional experience
        experience_id = self.emotional_memory.store_experience(
            self.emotional_field, "joy", "test_context"
        )
        
        # Step 2: Regulate mood
        regulated_field = self.mood_regulator.regulate_mood(self.emotional_field)
        
        # Step 3: Simulate empathy
        empathetic_response = self.empathy_sim.simulate_empathy(regulated_field)
        
        # Step 4: Amplify emotion
        amplified_field = self.emotional_amp.amplify_emotion(empathetic_response)
        
        # All should complete without error
        self.assertIsNotNone(experience_id)
        self.assertIsNotNone(regulated_field)
        self.assertIsNotNone(empathetic_response)
        self.assertIsNotNone(amplified_field)
    
    def test_emotion_coordination(self):
        """Test coordination between emotion mechanisms"""
        # Store emotional experience
        self.emotional_memory.store_experience(self.emotional_field, "joy", "birthday")
        
        # Regulate mood based on stored experience
        regulated_field = self.mood_regulator.regulate_mood(self.emotional_field)
        
        # Find similar emotions from memory
        similar_emotions = self.emotional_memory.find_similar_emotions(regulated_field)
        
        # Use similar emotions to guide empathy
        if similar_emotions:
            empathy_response = self.empathy_sim.simulate_empathy(regulated_field)
            self.assertIsNotNone(empathy_response)
    
    def test_emotion_adaptation(self):
        """Test adaptive emotion processing"""
        # Create emotional overload
        high_intensity_field = NDAnalogField((8, 8))
        high_intensity_field.activation.fill(0.9)
        
        # Amplify emotion
        amplified_field = self.emotional_amp.amplify_emotion(high_intensity_field, 2.0)
        
        # Detect overload and regulate mood
        if self.emotional_amp.detect_emotional_overload():
            regulated_field = self.mood_regulator.stabilize_mood(amplified_field)
            self.assertIsNotNone(regulated_field)


def create_test_emotional_field():
    """Helper function to create test emotional field"""
    field = NDAnalogField((8, 8))
    field.activation[2:4, 2:4] = 0.8
    return field


def create_test_high_intensity_field():
    """Helper function to create high intensity emotional field"""
    field = NDAnalogField((8, 8))
    field.activation.fill(0.9)
    return field


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

