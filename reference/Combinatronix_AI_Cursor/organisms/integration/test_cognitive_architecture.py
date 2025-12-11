# ============================================================================
# CognitiveArchitecture Test Suite
# ============================================================================

"""
Test suite for CognitiveArchitecture organism
"""

import numpy as np
import unittest
from unittest.mock import Mock, patch

try:
    from .cognitive_architecture import CognitiveArchitecture, CognitiveState, ActionPlan, ThoughtProcess
except ImportError:
    from combinatronix.organisms.integration.cognitive_architecture import CognitiveArchitecture, CognitiveState, ActionPlan, ThoughtProcess


class TestCognitiveArchitecture(unittest.TestCase):
    """Test CognitiveArchitecture organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mind = CognitiveArchitecture({'enable_visualization': False})
    
    def test_initialization(self):
        """Test mind initialization"""
        self.assertIsNotNone(self.mind.subsystems)
        self.assertIsNotNone(self.mind.integration_components)
        self.assertIsNotNone(self.mind.control_systems)
        self.assertEqual(len(self.mind.subsystems), 9)  # All subsystems
        self.assertEqual(len(self.mind.integration_components), 15)  # All integration components
        self.assertEqual(len(self.mind.control_systems), 5)  # All control systems
        self.assertEqual(self.mind.state['tick_counter'], 0)
        self.assertEqual(self.mind.state['cognitive_state'], 'idle')
        self.assertEqual(len(self.mind.state['goal_stack']), 0)
    
    def test_perceive(self):
        """Test perception"""
        sensory_input = {
            'visual': np.random.random((32, 32)) * 0.8,
            'timestamp': 0
        }
        
        features = self.mind.perceive(sensory_input)
        
        self.assertIsInstance(features, dict)
        self.assertIn('visual', features)
        self.assertEqual(self.mind.state['tick_counter'], 1)
    
    def test_perceive_auditory(self):
        """Test auditory perception (placeholder)"""
        sensory_input = {
            'auditory': {'placeholder': True},
            'timestamp': 0
        }
        
        features = self.mind.perceive(sensory_input)
        
        self.assertIsInstance(features, dict)
        self.assertIn('auditory', features)
        self.assertEqual(features['auditory']['placeholder'], True)
    
    def test_think_deliberate(self):
        """Test deliberate thinking"""
        thought = self.mind.think("deliberate")
        
        self.assertIsInstance(thought, ThoughtProcess)
        self.assertEqual(thought.thought_type, "deliberate")
        self.assertIsInstance(thought.concepts_involved, list)
        self.assertIsInstance(thought.reasoning_steps, list)
        self.assertGreaterEqual(thought.confidence, 0)
        self.assertLessEqual(thought.confidence, 1)
        self.assertEqual(self.mind.state['total_thoughts'], 1)
        self.assertEqual(self.mind.state['cognitive_state'], 'thinking')
    
    def test_think_associative(self):
        """Test associative thinking"""
        thought = self.mind.think("associative")
        
        self.assertIsInstance(thought, ThoughtProcess)
        self.assertEqual(thought.thought_type, "associative")
        self.assertIsInstance(thought.concepts_involved, list)
        self.assertIsInstance(thought.reasoning_steps, list)
    
    def test_think_creative(self):
        """Test creative thinking"""
        thought = self.mind.think("creative")
        
        self.assertIsInstance(thought, ThoughtProcess)
        self.assertEqual(thought.thought_type, "creative")
        self.assertIsInstance(thought.concepts_involved, list)
        self.assertIsInstance(thought.reasoning_steps, list)
    
    def test_think_reflexive(self):
        """Test reflexive thinking"""
        thought = self.mind.think("reflexive")
        
        self.assertIsInstance(thought, ThoughtProcess)
        self.assertEqual(thought.thought_type, "reflexive")
        self.assertIsInstance(thought.concepts_involved, list)
        self.assertIsInstance(thought.reasoning_steps, list)
    
    def test_integrate(self):
        """Test integration"""
        integration_result = self.mind.integrate()
        
        self.assertIsInstance(integration_result, dict)
        self.assertIn('coherence', integration_result)
        self.assertIn('correlation_matrix', integration_result)
        self.assertIn('subsystem_states', integration_result)
        self.assertGreaterEqual(integration_result['coherence'], 0)
        self.assertLessEqual(integration_result['coherence'], 1)
        self.assertEqual(self.mind.state['total_integrations'], 1)
    
    def test_act_with_goals(self):
        """Test action generation with goals"""
        # Add a goal first
        self.mind.add_goal("test_goal", priority=0.8)
        
        action = self.mind.act()
        
        # May or may not generate action depending on confidence
        if action:
            self.assertIsInstance(action, ActionPlan)
            self.assertIn(action.action_type, ["predicted"])
            self.assertGreaterEqual(action.confidence, 0)
            self.assertLessEqual(action.confidence, 1)
            self.assertIsInstance(action.action_field, NDAnalogField)
            self.assertIsInstance(action.expected_outcome, NDAnalogField)
            self.assertIsInstance(action.reasoning_chain, list)
            self.assertIsInstance(action.alternatives, list)
            self.assertEqual(self.mind.state['total_actions'], 1)
    
    def test_act_without_goals(self):
        """Test action generation without goals"""
        action = self.mind.act()
        
        self.assertIsNone(action)
        self.assertEqual(self.mind.state['total_actions'], 0)
    
    def test_add_goal(self):
        """Test adding goals"""
        self.mind.add_goal("test_goal", priority=0.8)
        
        self.assertEqual(len(self.mind.state['goal_stack']), 1)
        self.assertEqual(self.mind.state['goal_stack'][0]['goal'], "test_goal")
        self.assertEqual(self.mind.state['goal_stack'][0]['priority'], 0.8)
    
    def test_add_goal_max_limit(self):
        """Test adding goals with max limit"""
        # Add max goals
        for i in range(self.mind.config['max_goals']):
            self.mind.add_goal(f"goal_{i}", priority=0.5)
        
        # Try to add one more
        self.mind.add_goal("overflow_goal", priority=0.5)
        
        # Should not exceed max
        self.assertEqual(len(self.mind.state['goal_stack']), self.mind.config['max_goals'])
    
    def test_remove_goal(self):
        """Test removing goals"""
        self.mind.add_goal("test_goal", priority=0.8)
        self.mind.add_goal("another_goal", priority=0.6)
        
        self.mind.remove_goal("test_goal")
        
        self.assertEqual(len(self.mind.state['goal_stack']), 1)
        self.assertEqual(self.mind.state['goal_stack'][0]['goal'], "another_goal")
    
    def test_remove_nonexistent_goal(self):
        """Test removing nonexistent goal"""
        self.mind.add_goal("test_goal", priority=0.8)
        
        self.mind.remove_goal("nonexistent_goal")
        
        # Should not change
        self.assertEqual(len(self.mind.state['goal_stack']), 1)
        self.assertEqual(self.mind.state['goal_stack'][0]['goal'], "test_goal")
    
    def test_get_cognitive_state(self):
        """Test cognitive state retrieval"""
        # Add some data first
        self.mind.add_goal("test_goal", priority=0.8)
        self.mind.think("deliberate")
        
        cognitive_state = self.mind.get_cognitive_state()
        
        self.assertIsInstance(cognitive_state, CognitiveState)
        self.assertEqual(cognitive_state.timestamp, self.mind.state['tick_counter'])
        self.assertGreaterEqual(cognitive_state.global_workspace_energy, 0)
        self.assertIsInstance(cognitive_state.active_concepts, list)
        self.assertIsInstance(cognitive_state.current_goals, list)
        self.assertGreaterEqual(cognitive_state.system_coherence, 0)
        self.assertLessEqual(cognitive_state.system_coherence, 1)
        self.assertGreaterEqual(cognitive_state.self_awareness_level, 0)
        self.assertLessEqual(cognitive_state.self_awareness_level, 1)
        self.assertGreaterEqual(cognitive_state.memory_activation, 0)
        self.assertGreaterEqual(cognitive_state.reasoning_activity, 0)
        self.assertGreaterEqual(cognitive_state.language_activity, 0)
        self.assertGreaterEqual(cognitive_state.integration_quality, 0)
        self.assertLessEqual(cognitive_state.integration_quality, 1)
    
    def test_get_system_summary(self):
        """Test system summary retrieval"""
        # Add some data first
        self.mind.add_goal("test_goal", priority=0.8)
        self.mind.think("deliberate")
        self.mind.integrate()
        
        summary = self.mind.get_system_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('tick', summary)
        self.assertIn('cognitive_state', summary)
        self.assertIn('goals', summary)
        self.assertIn('thoughts', summary)
        self.assertIn('actions', summary)
        self.assertIn('integrations', summary)
        self.assertIn('subsystems', summary)
        self.assertIn('integration_components', summary)
        self.assertIn('control_systems', summary)
        self.assertIn('global_workspace_energy', summary)
        self.assertIn('attention_energy', summary)
        self.assertIn('integration_energy', summary)
        
        # Check specific values
        self.assertEqual(summary['goals'], 1)
        self.assertEqual(summary['thoughts'], 1)
        self.assertEqual(summary['integrations'], 1)
        self.assertEqual(summary['subsystems'], 9)
        self.assertEqual(summary['integration_components'], 15)
        self.assertEqual(summary['control_systems'], 5)
    
    def test_stream_of_consciousness(self):
        """Test stream of consciousness"""
        # Add a goal first
        self.mind.add_goal("test_goal", priority=0.8)
        
        actions_generated = 0
        for action in self.mind.stream_of_consciousness(duration=5):
            actions_generated += 1
            self.assertIsInstance(action, ActionPlan)
            
            if actions_generated >= 3:  # Limit for test
                break
        
        # Should have run some cycles
        self.assertGreater(self.mind.state['tick_counter'], 0)
        self.assertGreater(self.mind.state['total_thoughts'], 0)
    
    def test_bind_to_semantics(self):
        """Test semantic binding"""
        features = {
            'edges': np.random.random((32, 32)) * 0.8
        }
        
        concepts = self.mind._bind_to_semantics(features)
        
        self.assertIsInstance(concepts, list)
        self.assertTrue(all(isinstance(concept, str) for concept in concepts))
    
    def test_is_significant(self):
        """Test significance detection"""
        # Test significant features
        significant_features = {
            'edges': np.random.random((32, 32)) * 0.8
        }
        self.assertTrue(self.mind._is_significant(significant_features))
        
        # Test insignificant features
        insignificant_features = {
            'edges': np.random.random((32, 32)) * 0.1
        }
        self.assertFalse(self.mind._is_significant(insignificant_features))
    
    def test_resize_to_global_workspace(self):
        """Test field resizing"""
        # Test smaller field
        small_field = np.random.random((16, 16))
        resized = self.mind._resize_to_global_workspace(small_field)
        self.assertEqual(resized.shape, self.mind.config['global_workspace_size'])
        
        # Test larger field
        large_field = np.random.random((128, 128))
        resized = self.mind._resize_to_global_workspace(large_field)
        self.assertEqual(resized.shape, self.mind.config['global_workspace_size'])
        
        # Test same size field
        same_field = np.random.random(self.mind.config['global_workspace_size'])
        resized = self.mind._resize_to_global_workspace(same_field)
        self.assertEqual(resized.shape, self.mind.config['global_workspace_size'])
        np.testing.assert_array_equal(resized, same_field)
    
    def test_deliberate_thinking(self):
        """Test deliberate thinking process"""
        concepts, steps = self.mind._deliberate_thinking()
        
        self.assertIsInstance(concepts, list)
        self.assertIsInstance(steps, list)
        self.assertTrue(all(isinstance(concept, str) for concept in concepts))
        self.assertTrue(all(isinstance(step, str) for step in steps))
    
    def test_associative_thinking(self):
        """Test associative thinking process"""
        concepts, steps = self.mind._associative_thinking()
        
        self.assertIsInstance(concepts, list)
        self.assertIsInstance(steps, list)
        self.assertTrue(all(isinstance(concept, str) for concept in concepts))
        self.assertTrue(all(isinstance(step, str) for step in steps))
    
    def test_creative_thinking(self):
        """Test creative thinking process"""
        concepts, steps = self.mind._creative_thinking()
        
        self.assertIsInstance(concepts, list)
        self.assertIsInstance(steps, list)
        self.assertTrue(all(isinstance(concept, str) for concept in concepts))
        self.assertTrue(all(isinstance(step, str) for step in steps))
    
    def test_reflexive_thinking(self):
        """Test reflexive thinking process"""
        concepts, steps = self.mind._reflexive_thinking()
        
        self.assertIsInstance(concepts, list)
        self.assertIsInstance(steps, list)
        self.assertTrue(all(isinstance(concept, str) for concept in concepts))
        self.assertTrue(all(isinstance(step, str) for step in steps))
    
    def test_compute_thought_confidence(self):
        """Test thought confidence computation"""
        concepts = ["concept1", "concept2", "concept3"]
        steps = ["step1", "step2"]
        
        confidence = self.mind._compute_thought_confidence(concepts, steps)
        
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_gather_subsystem_states(self):
        """Test subsystem state gathering"""
        states = self.mind._gather_subsystem_states()
        
        self.assertIsInstance(states, dict)
        # Should have some states even if empty
        self.assertGreaterEqual(len(states), 0)
    
    def test_create_field_from_energies(self):
        """Test field creation from energies"""
        energies = {
            'energy1': 0.8,
            'energy2': 0.6,
            'energy3': 0.4
        }
        
        field = self.mind._create_field_from_energies(energies)
        
        self.assertIsInstance(field, NDAnalogField)
        self.assertEqual(field.shape, self.mind.config['global_workspace_size'])
    
    def test_resolve_conflicts(self):
        """Test conflict resolution"""
        correlation_matrix = np.random.random((5, 5))
        
        # Should not raise exception
        self.mind._resolve_conflicts(correlation_matrix)
    
    def test_integrate_into_workspace(self):
        """Test workspace integration"""
        subsystem_states = {
            'test_subsystem': NDAnalogField((32, 32))
        }
        
        # Should not raise exception
        self.mind._integrate_into_workspace(subsystem_states)
    
    def test_simulate_action(self):
        """Test action simulation"""
        action_field = NDAnalogField((32, 32))
        action_field.activation = np.random.random((32, 32)) * 0.8
        
        outcome = self.mind._simulate_action(action_field)
        
        self.assertIsInstance(outcome, NDAnalogField)
        self.assertEqual(outcome.shape, self.mind.config['global_workspace_size'])
    
    def test_evaluate_outcome(self):
        """Test outcome evaluation"""
        outcome = NDAnalogField((32, 32))
        outcome.activation = np.random.random((32, 32)) * 0.8
        
        confidence = self.mind._evaluate_outcome(outcome, "test_goal")
        
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_generate_reasoning_chain(self):
        """Test reasoning chain generation"""
        chain = self.mind._generate_reasoning_chain()
        
        self.assertIsInstance(chain, list)
        self.assertGreater(len(chain), 0)
        self.assertTrue(all(isinstance(step, str) for step in chain))
    
    def test_generate_alternatives(self):
        """Test alternative generation"""
        alternatives = self.mind._generate_alternatives()
        
        self.assertIsInstance(alternatives, list)
        self.assertGreater(len(alternatives), 0)
        self.assertTrue(all(isinstance(alt, str) for alt in alternatives))
    
    def test_has_input(self):
        """Test input availability check"""
        has_input = self.mind._has_input()
        
        self.assertIsInstance(has_input, bool)
    
    def test_get_input(self):
        """Test input retrieval"""
        input_data = self.mind._get_input()
        
        self.assertIsInstance(input_data, dict)
        self.assertIn('visual', input_data)
        self.assertIn('timestamp', input_data)
    
    def test_propagate_all(self):
        """Test field propagation"""
        # Should not raise exception
        self.mind._propagate_all()
    
    def test_update_cognitive_state(self):
        """Test cognitive state update"""
        # Should not raise exception
        self.mind._update_cognitive_state()
    
    def test_cognitive_state_creation(self):
        """Test CognitiveState dataclass"""
        state = CognitiveState(
            timestamp=10,
            global_workspace_energy=0.8,
            attention_focus="visual",
            active_concepts=["concept1", "concept2"],
            current_goals=["goal1"],
            system_coherence=0.7,
            self_awareness_level=0.6,
            memory_activation=0.5,
            reasoning_activity=0.4,
            language_activity=0.3,
            integration_quality=0.2
        )
        
        self.assertEqual(state.timestamp, 10)
        self.assertEqual(state.global_workspace_energy, 0.8)
        self.assertEqual(state.attention_focus, "visual")
        self.assertEqual(state.active_concepts, ["concept1", "concept2"])
        self.assertEqual(state.current_goals, ["goal1"])
        self.assertEqual(state.system_coherence, 0.7)
        self.assertEqual(state.self_awareness_level, 0.6)
        self.assertEqual(state.memory_activation, 0.5)
        self.assertEqual(state.reasoning_activity, 0.4)
        self.assertEqual(state.language_activity, 0.3)
        self.assertEqual(state.integration_quality, 0.2)
    
    def test_action_plan_creation(self):
        """Test ActionPlan dataclass"""
        action_field = NDAnalogField((32, 32))
        outcome_field = NDAnalogField((32, 32))
        
        plan = ActionPlan(
            action_type="test_action",
            action_field=action_field,
            confidence=0.8,
            expected_outcome=outcome_field,
            reasoning_chain=["step1", "step2"],
            alternatives=["alt1", "alt2"]
        )
        
        self.assertEqual(plan.action_type, "test_action")
        self.assertEqual(plan.action_field, action_field)
        self.assertEqual(plan.confidence, 0.8)
        self.assertEqual(plan.expected_outcome, outcome_field)
        self.assertEqual(plan.reasoning_chain, ["step1", "step2"])
        self.assertEqual(plan.alternatives, ["alt1", "alt2"])
    
    def test_thought_process_creation(self):
        """Test ThoughtProcess dataclass"""
        process = ThoughtProcess(
            thought_type="deliberate",
            duration=100,
            concepts_involved=["concept1", "concept2"],
            reasoning_steps=["step1", "step2"],
            outcome="completed",
            confidence=0.8
        )
        
        self.assertEqual(process.thought_type, "deliberate")
        self.assertEqual(process.duration, 100)
        self.assertEqual(process.concepts_involved, ["concept1", "concept2"])
        self.assertEqual(process.reasoning_steps, ["step1", "step2"])
        self.assertEqual(process.outcome, "completed")
        self.assertEqual(process.confidence, 0.8)
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'global_workspace_size': (32, 32), 'max_goals': 5},
            {'global_workspace_size': (64, 64), 'integration_frequency': 3},
            {'reflection_frequency': 5, 'coherence_threshold': 0.5}
        ]
        
        for config in configs:
            mind = CognitiveArchitecture(config)
            
            # Should work with different configs
            thought = mind.think("deliberate")
            self.assertIsInstance(thought, ThoughtProcess)
            
            integration = mind.integrate()
            self.assertIsInstance(integration, dict)
    
    def test_complete_cognitive_cycle(self):
        """Test complete cognitive cycle"""
        # Add goals
        self.mind.add_goal("test_goal", priority=0.8)
        
        # Perception
        sensory_input = {'visual': np.random.random((32, 32)) * 0.8}
        features = self.mind.perceive(sensory_input)
        self.assertIsInstance(features, dict)
        
        # Thinking
        thought = self.mind.think("deliberate")
        self.assertIsInstance(thought, ThoughtProcess)
        
        # Integration
        integration = self.mind.integrate()
        self.assertIsInstance(integration, dict)
        
        # Action
        action = self.mind.act()
        # May or may not generate action
        
        # Cognitive state
        cognitive_state = self.mind.get_cognitive_state()
        self.assertIsInstance(cognitive_state, CognitiveState)
        
        # Check that all operations were recorded
        self.assertGreater(self.mind.state['tick_counter'], 0)
        self.assertGreater(self.mind.state['total_thoughts'], 0)
        self.assertGreater(self.mind.state['total_integrations'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

