# ============================================================================
# SelfModel Test Suite
# ============================================================================

"""
Test suite for SelfModel organism
"""

import numpy as np
import unittest

try:
    from .self_model import SelfModel, SelfSnapshot, CapabilityAssessment, SelfImprovementGoal
except ImportError:
    from combinatronix.organisms.reasoning.self_model import SelfModel, SelfSnapshot, CapabilityAssessment, SelfImprovementGoal


class TestSelfModel(unittest.TestCase):
    """Test SelfModel organism"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SelfModel({'enable_visualization': False})
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.atoms)
        self.assertIsNotNone(self.model.molecules)
        self.assertIsNotNone(self.model.fields)
        self.assertEqual(len(self.model.atoms), 6)
        self.assertEqual(len(self.model.molecules), 3)
        self.assertEqual(len(self.model.fields), 8)
        self.assertEqual(len(self.model.state['abilities']), 0)
        self.assertEqual(len(self.model.state['limitations']), 0)
        self.assertEqual(len(self.model.state['current_goals']), 0)
    
    def test_observe_self(self):
        """Test self observation"""
        snapshot = self.model.observe_self()
        
        self.assertIsInstance(snapshot, SelfSnapshot)
        self.assertEqual(snapshot.timestamp, 1)
        self.assertGreaterEqual(snapshot.surprise_level, 0)
        self.assertLessEqual(snapshot.surprise_level, 1)
        self.assertGreaterEqual(snapshot.prediction_accuracy, 0)
        self.assertLessEqual(snapshot.prediction_accuracy, 1)
        self.assertGreaterEqual(snapshot.self_consistency, 0)
        self.assertLessEqual(snapshot.self_consistency, 1)
        
        # Check that snapshot was stored
        self.assertEqual(len(self.model.state['self_history']), 1)
        self.assertEqual(self.model.state['total_observations'], 1)
    
    def test_predict_own_action(self):
        """Test action prediction"""
        situation_field = NDAnalogField((16, 16))
        situation_field.activation = np.random.random((16, 16)) * 0.8
        
        predicted_action = self.model.predict_own_action(situation_field)
        
        self.assertIsInstance(predicted_action, NDAnalogField)
        self.assertEqual(predicted_action.shape, (16, 16))
        self.assertGreaterEqual(np.sum(predicted_action.activation), 0)
        
        # Check that prediction was stored
        self.assertEqual(self.model.state['total_predictions'], 1)
    
    def test_reflect(self):
        """Test self reflection"""
        # First, add some goals to make reflection meaningful
        self.model.add_goal("test_goal", priority=0.8)
        
        improvement_goal = self.model.reflect()
        
        # May or may not generate improvement goal depending on discrepancy
        if improvement_goal:
            self.assertIsInstance(improvement_goal, SelfImprovementGoal)
            self.assertIn(improvement_goal.goal_type, ["ability", "belief", "behavior", "goal"])
            self.assertGreater(improvement_goal.priority, 0)
            self.assertLessEqual(improvement_goal.priority, 1)
            self.assertIsInstance(improvement_goal.improvement_plan, list)
        
        # Check that reflection was recorded
        self.assertEqual(self.model.state['total_reflections'], 1)
    
    def test_update_beliefs(self):
        """Test belief updating"""
        evidence_field = NDAnalogField((16, 16))
        evidence_field.activation = np.random.random((16, 16)) * 0.9
        
        update_strength = self.model.update_beliefs(evidence_field)
        
        self.assertGreaterEqual(update_strength, 0)
        self.assertLessEqual(update_strength, 1)
        
        # Check that belief update was recorded
        self.assertEqual(len(self.model.state['belief_history']), 1)
    
    def test_update_beliefs_weak_evidence(self):
        """Test belief updating with weak evidence"""
        evidence_field = NDAnalogField((16, 16))
        evidence_field.activation = np.random.random((16, 16)) * 0.1  # Weak evidence
        
        update_strength = self.model.update_beliefs(evidence_field)
        
        # Should not update beliefs with weak evidence
        self.assertEqual(update_strength, 0)
        self.assertEqual(len(self.model.state['belief_history']), 0)
    
    def test_assess_capability(self):
        """Test capability assessment"""
        task_field = NDAnalogField((16, 16))
        task_field.activation = np.random.random((16, 16)) * 0.7
        
        assessment = self.model.assess_capability(task_field)
        
        self.assertIsInstance(assessment, CapabilityAssessment)
        self.assertGreaterEqual(assessment.confidence_score, 0)
        self.assertLessEqual(assessment.confidence_score, 1)
        self.assertIsInstance(assessment.required_abilities, list)
        self.assertIsInstance(assessment.missing_abilities, list)
        self.assertGreaterEqual(assessment.estimated_difficulty, 0)
        self.assertLessEqual(assessment.estimated_difficulty, 1)
        self.assertIsInstance(assessment.recommended_approach, str)
    
    def test_theory_of_self(self):
        """Test theory of self generation"""
        # Add some abilities and goals first
        self.model.add_ability("test_ability")
        self.model.add_goal("test_goal", priority=0.8)
        
        theory = self.model.theory_of_self()
        
        self.assertIsInstance(theory, dict)
        self.assertIn('goals', theory)
        self.assertIn('abilities', theory)
        self.assertIn('limitations', theory)
        self.assertIn('current_state', theory)
        self.assertIn('surprise_level', theory)
        self.assertIn('prediction_error', theory)
        self.assertIn('self_consistency', theory)
        self.assertIn('total_observations', theory)
        self.assertIn('total_predictions', theory)
        self.assertIn('total_reflections', theory)
        self.assertIn('improvement_goals', theory)
        self.assertIn('field_energies', theory)
        
        # Check specific values
        self.assertEqual(len(theory['abilities']), 1)
        self.assertEqual(len(theory['goals']), 1)
        self.assertGreaterEqual(theory['surprise_level'], 0)
        self.assertLessEqual(theory['surprise_level'], 1)
    
    def test_add_ability(self):
        """Test adding ability"""
        self.model.add_ability("test_ability")
        
        self.assertIn("test_ability", self.model.state['abilities'])
        self.assertEqual(len(self.model.state['abilities']), 1)
    
    def test_add_limitation(self):
        """Test adding limitation"""
        self.model.add_limitation("test_limitation")
        
        self.assertIn("test_limitation", self.model.state['limitations'])
        self.assertEqual(len(self.model.state['limitations']), 1)
    
    def test_add_goal(self):
        """Test adding goal"""
        self.model.add_goal("test_goal", priority=0.8)
        
        self.assertEqual(len(self.model.state['current_goals']), 1)
        goal = self.model.state['current_goals'][0]
        self.assertEqual(goal['goal'], "test_goal")
        self.assertEqual(goal['priority'], 0.8)
    
    def test_add_goal_default_priority(self):
        """Test adding goal with default priority"""
        self.model.add_goal("test_goal")
        
        goal = self.model.state['current_goals'][0]
        self.assertEqual(goal['priority'], 1.0)
    
    def test_get_self_summary(self):
        """Test self summary generation"""
        # Add some data first
        self.model.add_ability("test_ability")
        self.model.add_limitation("test_limitation")
        self.model.add_goal("test_goal")
        self.model.observe_self()
        
        summary = self.model.get_self_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('tick', summary)
        self.assertIn('abilities', summary)
        self.assertIn('limitations', summary)
        self.assertIn('goals', summary)
        self.assertIn('history_length', summary)
        self.assertIn('improvement_goals', summary)
        self.assertIn('total_observations', summary)
        self.assertIn('total_predictions', summary)
        self.assertIn('total_reflections', summary)
        
        # Check specific values
        self.assertEqual(summary['abilities'], 1)
        self.assertEqual(summary['limitations'], 1)
        self.assertEqual(summary['goals'], 1)
        self.assertEqual(summary['history_length'], 1)
        self.assertEqual(summary['total_observations'], 1)
    
    def test_get_state(self):
        """Test state retrieval"""
        state = self.model.get_state()
        
        self.assertIsInstance(state, dict)
        self.assertIn('config', state)
        self.assertIn('state', state)
        self.assertIn('field_shapes', state)
        self.assertIn('atom_states', state)
        self.assertIn('molecule_states', state)
    
    def test_reset(self):
        """Test model reset"""
        # Add some data first
        self.model.add_ability("test_ability")
        self.model.add_limitation("test_limitation")
        self.model.add_goal("test_goal")
        self.model.observe_self()
        
        # Reset
        self.model.reset()
        
        # Check that state is reset
        self.assertEqual(len(self.model.state['abilities']), 0)
        self.assertEqual(len(self.model.state['limitations']), 0)
        self.assertEqual(len(self.model.state['current_goals']), 0)
        self.assertEqual(len(self.model.state['self_history']), 0)
        self.assertEqual(len(self.model.state['belief_history']), 0)
        self.assertEqual(len(self.model.state['improvement_goals']), 0)
        self.assertEqual(self.model.state['tick_counter'], 0)
        self.assertEqual(self.model.state['total_observations'], 0)
        self.assertEqual(self.model.state['total_predictions'], 0)
        self.assertEqual(self.model.state['total_reflections'], 0)
    
    def test_compute_surprise_molecular(self):
        """Test molecular surprise computation"""
        surprise = self.model._compute_surprise_molecular()
        
        self.assertGreaterEqual(surprise, 0)
        self.assertLessEqual(surprise, 1)
    
    def test_compute_prediction_accuracy(self):
        """Test prediction accuracy computation"""
        # Test with no history
        accuracy = self.model._compute_prediction_accuracy()
        self.assertEqual(accuracy, 0.0)
        
        # Test with history
        self.model.observe_self()
        accuracy = self.model._compute_prediction_accuracy()
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_compute_self_consistency(self):
        """Test self-consistency computation"""
        consistency = self.model._compute_self_consistency()
        
        self.assertGreaterEqual(consistency, 0)
        self.assertLessEqual(consistency, 1)
    
    def test_compute_field_consistency(self):
        """Test field consistency computation"""
        field1 = NDAnalogField((16, 16))
        field1.activation = np.random.random((16, 16)) * 0.8
        
        field2 = NDAnalogField((16, 16))
        field2.activation = np.random.random((16, 16)) * 0.8
        
        consistency = self.model._compute_field_consistency(field1, field2)
        
        self.assertGreaterEqual(consistency, 0)
        self.assertLessEqual(consistency, 1)
    
    def test_weave_fields_molecular(self):
        """Test molecular field weaving"""
        field1 = NDAnalogField((16, 16))
        field1.activation = np.random.random((16, 16)) * 0.8
        
        field2 = NDAnalogField((16, 16))
        field2.activation = np.random.random((16, 16)) * 0.8
        
        woven = self.model._weave_fields_molecular(field1, field2)
        
        self.assertIsInstance(woven, NDAnalogField)
        self.assertEqual(woven.shape, (16, 16))
    
    def test_compute_self_discrepancy_molecular(self):
        """Test molecular self-discrepancy computation"""
        current_self = NDAnalogField((16, 16))
        current_self.activation = np.random.random((16, 16)) * 0.8
        
        ideal_self = NDAnalogField((16, 16))
        ideal_self.activation = np.random.random((16, 16)) * 0.8
        
        discrepancy = self.model._compute_self_discrepancy_molecular(current_self, ideal_self)
        
        self.assertGreaterEqual(discrepancy, 0)
        self.assertLessEqual(discrepancy, 1)
    
    def test_generate_improvement_goal_molecular(self):
        """Test molecular improvement goal generation"""
        discrepancy = 0.5
        
        goal = self.model._generate_improvement_goal_molecular(discrepancy)
        
        self.assertIsInstance(goal, SelfImprovementGoal)
        self.assertIn(goal.goal_type, ["ability", "belief", "behavior", "goal"])
        self.assertEqual(goal.priority, discrepancy)
        self.assertIsInstance(goal.improvement_plan, list)
        self.assertGreater(len(goal.improvement_plan), 0)
    
    def test_determine_improvement_type(self):
        """Test improvement type determination"""
        # Test different discrepancy levels
        self.assertEqual(self.model._determine_improvement_type(0.8), "behavior")
        self.assertEqual(self.model._determine_improvement_type(0.6), "belief")
        self.assertEqual(self.model._determine_improvement_type(0.4), "ability")
        self.assertEqual(self.model._determine_improvement_type(0.2), "goal")
    
    def test_generate_improvement_plan(self):
        """Test improvement plan generation"""
        plans = {
            "ability": self.model._generate_improvement_plan("ability", 0.5),
            "belief": self.model._generate_improvement_plan("belief", 0.5),
            "behavior": self.model._generate_improvement_plan("behavior", 0.5),
            "goal": self.model._generate_improvement_plan("goal", 0.5)
        }
        
        for goal_type, plan in plans.items():
            self.assertIsInstance(plan, list)
            self.assertGreater(len(plan), 0)
            self.assertTrue(all(isinstance(item, str) for item in plan))
    
    def test_compute_ability_match_molecular(self):
        """Test molecular ability match computation"""
        task_field = NDAnalogField((16, 16))
        task_field.activation = np.random.random((16, 16)) * 0.8
        
        match = self.model._compute_ability_match_molecular(task_field)
        
        self.assertGreaterEqual(match, 0)
        self.assertLessEqual(match, 1)
    
    def test_identify_required_abilities(self):
        """Test required ability identification"""
        task_field = NDAnalogField((16, 16))
        task_field.activation = np.random.random((16, 16)) * 0.8
        
        abilities = self.model._identify_required_abilities(task_field)
        
        self.assertIsInstance(abilities, list)
        self.assertTrue(all(isinstance(ability, str) for ability in abilities))
    
    def test_generate_approach_recommendation(self):
        """Test approach recommendation generation"""
        # Test different confidence levels
        high_conf = self.model._generate_approach_recommendation(0.9, [])
        self.assertIn("Direct approach", high_conf)
        
        med_conf = self.model._generate_approach_recommendation(0.6, [])
        self.assertIn("Cautious approach", med_conf)
        
        low_conf = self.model._generate_approach_recommendation(0.3, ["missing_ability"])
        self.assertIn("Learn missing abilities", low_conf)
        
        very_low_conf = self.model._generate_approach_recommendation(0.1, [])
        self.assertIn("Seek assistance", very_low_conf)
    
    def test_self_snapshot_creation(self):
        """Test SelfSnapshot dataclass"""
        body_state = NDAnalogField((16, 16))
        goal_state = NDAnalogField((16, 16))
        belief_state = NDAnalogField((16, 16))
        
        snapshot = SelfSnapshot(
            timestamp=10,
            body_state=body_state,
            goal_state=goal_state,
            belief_state=belief_state,
            state_energy=0.8,
            surprise_level=0.3,
            prediction_accuracy=0.7,
            self_consistency=0.6
        )
        
        self.assertEqual(snapshot.timestamp, 10)
        self.assertEqual(snapshot.body_state, body_state)
        self.assertEqual(snapshot.goal_state, goal_state)
        self.assertEqual(snapshot.belief_state, belief_state)
        self.assertEqual(snapshot.state_energy, 0.8)
        self.assertEqual(snapshot.surprise_level, 0.3)
        self.assertEqual(snapshot.prediction_accuracy, 0.7)
        self.assertEqual(snapshot.self_consistency, 0.6)
    
    def test_capability_assessment_creation(self):
        """Test CapabilityAssessment dataclass"""
        assessment = CapabilityAssessment(
            task_description="test_task",
            confidence_score=0.8,
            required_abilities=["ability1", "ability2"],
            missing_abilities=["ability2"],
            estimated_difficulty=0.3,
            recommended_approach="Direct approach"
        )
        
        self.assertEqual(assessment.task_description, "test_task")
        self.assertEqual(assessment.confidence_score, 0.8)
        self.assertEqual(assessment.required_abilities, ["ability1", "ability2"])
        self.assertEqual(assessment.missing_abilities, ["ability2"])
        self.assertEqual(assessment.estimated_difficulty, 0.3)
        self.assertEqual(assessment.recommended_approach, "Direct approach")
    
    def test_self_improvement_goal_creation(self):
        """Test SelfImprovementGoal dataclass"""
        goal = SelfImprovementGoal(
            goal_type="ability",
            description="Improve ability",
            priority=0.8,
            current_state=0.3,
            target_state=0.9,
            improvement_plan=["step1", "step2"]
        )
        
        self.assertEqual(goal.goal_type, "ability")
        self.assertEqual(goal.description, "Improve ability")
        self.assertEqual(goal.priority, 0.8)
        self.assertEqual(goal.current_state, 0.3)
        self.assertEqual(goal.target_state, 0.9)
        self.assertEqual(goal.improvement_plan, ["step1", "step2"])
    
    def test_different_configurations(self):
        """Test with different configurations"""
        configs = [
            {'field_shape': (8, 8), 'max_history': 50},
            {'field_shape': (32, 32), 'surprise_threshold': 0.5},
            {'reflection_frequency': 5, 'improvement_threshold': 0.2}
        ]
        
        for config in configs:
            model = SelfModel(config)
            
            # Should work with different configs
            snapshot = model.observe_self()
            self.assertIsInstance(snapshot, SelfSnapshot)
            
            task_field = NDAnalogField(config['field_shape'])
            task_field.activation = np.random.random(config['field_shape']) * 0.8
            assessment = model.assess_capability(task_field)
            self.assertIsInstance(assessment, CapabilityAssessment)
    
    def test_self_model_operation_sequence(self):
        """Test complete self model operation sequence"""
        # Add abilities and goals
        self.model.add_ability("pattern_recognition")
        self.model.add_ability("logical_reasoning")
        self.model.add_limitation("limited_memory")
        self.model.add_goal("improve_efficiency", priority=0.8)
        
        # Self observation
        snapshot = self.model.observe_self()
        self.assertIsInstance(snapshot, SelfSnapshot)
        
        # Action prediction
        situation_field = NDAnalogField((16, 16))
        situation_field.activation = np.random.random((16, 16)) * 0.8
        predicted_action = self.model.predict_own_action(situation_field)
        self.assertIsInstance(predicted_action, NDAnalogField)
        
        # Self reflection
        improvement_goal = self.model.reflect()
        # May or may not generate improvement goal
        
        # Belief update
        evidence_field = NDAnalogField((16, 16))
        evidence_field.activation = np.random.random((16, 16)) * 0.9
        update_strength = self.model.update_beliefs(evidence_field)
        self.assertGreaterEqual(update_strength, 0)
        
        # Capability assessment
        task_field = NDAnalogField((16, 16))
        task_field.activation = np.random.random((16, 16)) * 0.7
        assessment = self.model.assess_capability(task_field)
        self.assertIsInstance(assessment, CapabilityAssessment)
        
        # Theory of self
        theory = self.model.theory_of_self()
        self.assertIsInstance(theory, dict)
        
        # Check that all operations were recorded
        self.assertGreater(self.model.state['total_observations'], 0)
        self.assertGreater(self.model.state['total_predictions'], 0)
        self.assertGreater(self.model.state['total_reflections'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

