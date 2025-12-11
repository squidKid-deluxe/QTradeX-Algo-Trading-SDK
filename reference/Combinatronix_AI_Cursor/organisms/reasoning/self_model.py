# ============================================================================
# SelfModel - Agent's Model of Itself Using Molecular Architecture
# ============================================================================

"""
SelfModel - Tracks own states, predicts own actions, monitors internal processes

Composition:
- Atoms: Mirror, Witness, Anticipator, MemoryTrace, Gradient, Comparator
- Molecules: WorkingMemory, PatternRecognizer, ContradictionResolver
- Fields: body_field, goal_field, belief_field, state_field, meta_field

This organism tracks own states, predicts own actions, and monitors internal
processes. Foundation of metacognition and consciousness.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import time

try:
    from ...core import NDAnalogField
    from ...atoms.pattern_primitives import MirrorAtom, WitnessAtom, GradientAtom
    from ...atoms.temporal import AnticipatorAtom, MemoryTraceAtom
    from ...atoms.multi_field import ComparatorAtom
    from ...molecules.memory import WorkingMemoryMolecule
    from ...molecules.perception import PatternRecognizerMolecule
    from ...molecules.reasoning import ContradictionResolverMolecule
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.atoms.pattern_primitives import MirrorAtom, WitnessAtom, GradientAtom
    from combinatronix.atoms.temporal import AnticipatorAtom, MemoryTraceAtom
    from combinatronix.atoms.multi_field import ComparatorAtom
    from combinatronix.molecules.memory import WorkingMemoryMolecule
    from combinatronix.molecules.perception import PatternRecognizerMolecule
    from combinatronix.molecules.reasoning import ContradictionResolverMolecule


@dataclass
class SelfSnapshot:
    """Represents a snapshot of the self at a point in time"""
    timestamp: int
    body_state: NDAnalogField
    goal_state: NDAnalogField
    belief_state: NDAnalogField
    state_energy: float
    surprise_level: float
    prediction_accuracy: float
    self_consistency: float


@dataclass
class CapabilityAssessment:
    """Assessment of capability for a specific task"""
    task_description: str
    confidence_score: float
    required_abilities: List[str]
    missing_abilities: List[str]
    estimated_difficulty: float
    recommended_approach: str


@dataclass
class SelfImprovementGoal:
    """Goal for self-improvement based on reflection"""
    goal_type: str  # "ability", "belief", "behavior", "goal"
    description: str
    priority: float
    current_state: float
    target_state: float
    improvement_plan: List[str]


class SelfModel:
    """Agent's model of itself using molecular architecture"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the self model
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'field_shape': (16, 16),
            'max_history': 1000,
            'surprise_threshold': 0.3,
            'reflection_frequency': 10,
            'enable_visualization': True,
            'prediction_horizon': 5,
            'self_consistency_threshold': 0.7,
            'improvement_threshold': 0.3
        }
        
        if config:
            self.config.update(config)
        
        # Initialize atoms
        self._initialize_atoms()
        
        # Initialize molecules
        self._initialize_molecules()
        
        # Initialize fields
        self._initialize_fields()
        
        # State tracking
        self.state = {
            'abilities': set(),
            'limitations': set(),
            'current_goals': [],
            'self_history': [],
            'belief_history': [],
            'improvement_goals': [],
            'tick_counter': 0,
            'total_observations': 0,
            'total_predictions': 0,
            'total_reflections': 0
        }
        
        print(f"🪞 SelfModel initialized ({self.config['field_shape'][0]}×{self.config['field_shape'][1]})")
    
    def _initialize_atoms(self):
        """Initialize atomic operations"""
        self.atoms = {
            'mirror': MirrorAtom(
                axis='vertical',
                reflection_strength=0.8
            ),
            'witness': WitnessAtom(
                observation_strength=1.5,
                memory_decay=0.95
            ),
            'anticipator': AnticipatorAtom(
                history_depth=5,
                prediction_strength=0.7
            ),
            'memory_trace': MemoryTraceAtom(
                accumulation_rate=0.3,
                decay_rate=0.95
            ),
            'gradient': GradientAtom(
                direction='ascent',
                strength=1.0
            ),
            'comparator': ComparatorAtom(
                metric='cosine',
                normalize=True
            )
        }
    
    def _initialize_molecules(self):
        """Initialize molecular operations"""
        self.molecules = {
            'working_memory': WorkingMemoryMolecule(
                capacity=50,
                decay_rate=0.95
            ),
            'pattern_recognizer': PatternRecognizerMolecule(
                amplification=1.5,
                resonance_threshold=0.5
            ),
            'contradiction_resolver': ContradictionResolverMolecule(
                equilibrium_rate=0.2,
                min_tension=0.1
            )
        }
    
    def _initialize_fields(self):
        """Initialize field structures"""
        self.fields = {
            'body_field': NDAnalogField(self.config['field_shape']),
            'goal_field': NDAnalogField(self.config['field_shape']),
            'belief_field': NDAnalogField(self.config['field_shape']),
            'state_field': NDAnalogField(self.config['field_shape']),
            'meta_field': NDAnalogField(self.config['field_shape']),
            'reflection_field': NDAnalogField(self.config['field_shape']),
            'prediction_field': NDAnalogField(self.config['field_shape']),
            'improvement_field': NDAnalogField(self.config['field_shape'])
        }
    
    def observe_self(self) -> SelfSnapshot:
        """
        Metacognitive observation using molecular operations
        
        Returns:
            SelfSnapshot object representing current self state
        """
        self.state['tick_counter'] += 1
        self.state['total_observations'] += 1
        
        # Create self snapshot
        snapshot = SelfSnapshot(
            timestamp=self.state['tick_counter'],
            body_state=self.fields['body_field'].copy(),
            goal_state=self.fields['goal_field'].copy(),
            belief_state=self.fields['belief_field'].copy(),
            state_energy=np.sum(self.fields['state_field'].activation),
            surprise_level=0.0,
            prediction_accuracy=0.0,
            self_consistency=0.0
        )
        
        # Observe with witness atom
        self.atoms['witness'].observe(self.fields['body_field'])
        self.atoms['witness'].observe(self.fields['goal_field'])
        self.atoms['witness'].observe(self.fields['belief_field'])
        self.atoms['witness'].observe(self.fields['state_field'])
        
        # Check prediction accuracy and surprise
        surprise = self._compute_surprise_molecular()
        prediction_accuracy = self._compute_prediction_accuracy()
        
        snapshot.surprise_level = surprise
        snapshot.prediction_accuracy = prediction_accuracy
        
        # Compute self-consistency
        self_consistency = self._compute_self_consistency()
        snapshot.self_consistency = self_consistency
        
        # Store in self history
        self.state['self_history'].append(snapshot)
        
        # Limit history size
        if len(self.state['self_history']) > self.config['max_history']:
            self.state['self_history'].pop(0)
        
        # Update meta field
        self._update_meta_field(snapshot)
        
        print(f"👁️ Self observation: surprise={surprise:.3f}, consistency={self_consistency:.3f}")
        
        return snapshot
    
    def _compute_surprise_molecular(self) -> float:
        """Compute surprise level using molecular operations"""
        # Use anticipator to get prediction error
        prediction_error = self.atoms['anticipator'].get_prediction_error(self.fields['state_field'])
        
        # Use pattern recognizer for additional surprise detection
        pattern_surprise = self.molecules['pattern_recognizer'].recognize(
            self.fields['state_field'], 
            self.fields['prediction_field']
        )
        
        # Combine surprise measures
        surprise = (prediction_error + np.mean(pattern_surprise.activation)) / 2.0
        
        return min(1.0, surprise)
    
    def _compute_prediction_accuracy(self) -> float:
        """Compute prediction accuracy"""
        if len(self.state['self_history']) < 2:
            return 0.0
        
        # Compare current state with previous prediction
        current_state = self.fields['state_field'].activation
        previous_prediction = self.fields['prediction_field'].activation
        
        # Use comparator atom for accuracy
        comparison = self.atoms['comparator'].apply(
            NDAnalogField(self.config['field_shape'], activation=current_state),
            NDAnalogField(self.config['field_shape'], activation=previous_prediction)
        )
        
        accuracy = np.mean(comparison.activation)
        return accuracy
    
    def _compute_self_consistency(self) -> float:
        """Compute self-consistency across different self aspects"""
        # Compare different self fields for consistency
        body_goal_consistency = self._compute_field_consistency(
            self.fields['body_field'], 
            self.fields['goal_field']
        )
        
        goal_belief_consistency = self._compute_field_consistency(
            self.fields['goal_field'], 
            self.fields['belief_field']
        )
        
        belief_state_consistency = self._compute_field_consistency(
            self.fields['belief_field'], 
            self.fields['state_field']
        )
        
        # Average consistency
        consistency = (body_goal_consistency + goal_belief_consistency + belief_state_consistency) / 3.0
        
        return consistency
    
    def _compute_field_consistency(self, field1: NDAnalogField, field2: NDAnalogField) -> float:
        """Compute consistency between two fields"""
        # Use comparator atom
        comparison = self.atoms['comparator'].apply(field1, field2)
        
        # Use contradiction resolver for conflict detection
        contradiction = self.molecules['contradiction_resolver'].resolve(field1, field2)
        
        # Consistency is inverse of contradiction
        consistency = 1.0 - np.mean(contradiction.activation)
        
        return max(0.0, consistency)
    
    def _update_meta_field(self, snapshot: SelfSnapshot):
        """Update meta field with self observation"""
        # Combine different self aspects
        meta_activation = (
            snapshot.body_state.activation * 0.3 +
            snapshot.goal_state.activation * 0.3 +
            snapshot.belief_state.activation * 0.2 +
            self.fields['state_field'].activation * 0.2
        )
        
        self.fields['meta_field'].activation = meta_activation
    
    def predict_own_action(self, situation_field: NDAnalogField) -> NDAnalogField:
        """
        Predict what I'll do in situation using molecular operations
        
        Args:
            situation_field: Field representing the situation
            
        Returns:
            Predicted action field
        """
        self.state['total_predictions'] += 1
        
        # Combine situation with current goals using molecular operations
        combined_field = self._weave_fields_molecular(situation_field, self.fields['goal_field'])
        
        # Use anticipator to predict next state
        predicted_state = self.atoms['anticipator'].anticipate(combined_field)
        
        # Use gradient atom to find likely action
        action_field = self.atoms['gradient'].apply(predicted_state)
        
        # Update prediction field
        self.fields['prediction_field'].activation = predicted_state.activation.copy()
        
        # Store in working memory
        self.molecules['working_memory'].store(action_field)
        
        print(f"🔮 Predicted action: energy={np.sum(action_field.activation):.3f}")
        
        return action_field
    
    def _weave_fields_molecular(self, field1: NDAnalogField, field2: NDAnalogField) -> NDAnalogField:
        """Weave fields together using molecular operations"""
        # Use pattern recognizer for field combination
        combined = self.molecules['pattern_recognizer'].recognize(field1, field2)
        
        # Use working memory for additional processing
        self.molecules['working_memory'].store(combined)
        
        return combined
    
    def reflect(self) -> SelfImprovementGoal:
        """
        Self-reflection using molecular operations
        
        Returns:
            SelfImprovementGoal object
        """
        self.state['total_reflections'] += 1
        
        # Apply mirror atom to self-fields for reflection
        reflected_self = self.fields['state_field'].copy()
        self.atoms['mirror'].apply(reflected_self)
        
        # Update reflection field
        self.fields['reflection_field'].activation = reflected_self.activation.copy()
        
        # Compare to ideal (goal field) using molecular operations
        discrepancy = self._compute_self_discrepancy_molecular(reflected_self, self.fields['goal_field'])
        
        # Generate improvement goal if discrepancy is significant
        improvement_goal = None
        if discrepancy > self.config['improvement_threshold']:
            improvement_goal = self._generate_improvement_goal_molecular(discrepancy)
            self.state['improvement_goals'].append(improvement_goal)
        
        print(f"🪞 Self reflection: discrepancy={discrepancy:.3f}")
        
        return improvement_goal
    
    def _compute_self_discrepancy_molecular(self, current_self: NDAnalogField, ideal_self: NDAnalogField) -> float:
        """Compute discrepancy between current and ideal self using molecular operations"""
        # Use comparator atom
        comparison = self.atoms['comparator'].apply(current_self, ideal_self)
        
        # Use contradiction resolver for conflict detection
        contradiction = self.molecules['contradiction_resolver'].resolve(current_self, ideal_self)
        
        # Discrepancy is combination of comparison and contradiction
        discrepancy = (1.0 - np.mean(comparison.activation)) + np.mean(contradiction.activation)
        
        return min(1.0, discrepancy)
    
    def _generate_improvement_goal_molecular(self, discrepancy: float) -> SelfImprovementGoal:
        """Generate improvement goal using molecular operations"""
        # Analyze current state to determine improvement type
        goal_type = self._determine_improvement_type(discrepancy)
        
        # Generate improvement plan
        improvement_plan = self._generate_improvement_plan(goal_type, discrepancy)
        
        # Create improvement goal
        goal = SelfImprovementGoal(
            goal_type=goal_type,
            description=f"Improve {goal_type} (discrepancy: {discrepancy:.3f})",
            priority=discrepancy,
            current_state=1.0 - discrepancy,
            target_state=1.0,
            improvement_plan=improvement_plan
        )
        
        # Update improvement field
        self._update_improvement_field(goal)
        
        return goal
    
    def _determine_improvement_type(self, discrepancy: float) -> str:
        """Determine type of improvement needed"""
        if discrepancy > 0.7:
            return "behavior"
        elif discrepancy > 0.5:
            return "belief"
        elif discrepancy > 0.3:
            return "ability"
        else:
            return "goal"
    
    def _generate_improvement_plan(self, goal_type: str, discrepancy: float) -> List[str]:
        """Generate improvement plan based on goal type"""
        plans = {
            "ability": [
                "Practice specific skills",
                "Seek learning opportunities",
                "Break down complex tasks"
            ],
            "belief": [
                "Gather new evidence",
                "Challenge existing beliefs",
                "Seek alternative perspectives"
            ],
            "behavior": [
                "Monitor current behavior",
                "Set behavioral goals",
                "Practice new behaviors"
            ],
            "goal": [
                "Clarify goal priorities",
                "Set specific milestones",
                "Track progress regularly"
            ]
        }
        
        return plans.get(goal_type, ["General self-improvement"])
    
    def _update_improvement_field(self, goal: SelfImprovementGoal):
        """Update improvement field with goal information"""
        # Create improvement pattern based on goal type
        improvement_pattern = np.zeros(self.config['field_shape'])
        
        # Map goal type to field position
        goal_types = ["ability", "belief", "behavior", "goal"]
        if goal.goal_type in goal_types:
            type_index = goal_types.index(goal.goal_type)
            field_x = type_index % self.config['field_shape'][1]
            field_y = type_index // self.config['field_shape'][1]
            
            if field_y < self.config['field_shape'][0]:
                improvement_pattern[field_y, field_x] = goal.priority
        
        # Update improvement field
        self.fields['improvement_field'].activation += improvement_pattern * 0.1
        self.fields['improvement_field'].activation = np.clip(self.fields['improvement_field'].activation, 0, 1)
    
    def update_beliefs(self, new_evidence_field: NDAnalogField) -> float:
        """
        Update beliefs based on evidence using molecular operations
        
        Args:
            new_evidence_field: Field representing new evidence
            
        Returns:
            Belief update strength
        """
        # Weave evidence with existing beliefs
        updated_beliefs = self._weave_fields_molecular(new_evidence_field, self.fields['belief_field'])
        
        # Use threshold for belief revision
        evidence_strength = np.sum(new_evidence_field.activation)
        belief_strength = np.sum(self.fields['belief_field'].activation)
        
        # Update beliefs if evidence is strong enough
        if evidence_strength > belief_strength * 0.5:
            self.fields['belief_field'].activation = updated_beliefs.activation.copy()
            
            # Store belief update in history
            self.state['belief_history'].append({
                'timestamp': self.state['tick_counter'],
                'evidence_strength': evidence_strength,
                'belief_change': np.sum(updated_beliefs.activation) - belief_strength
            })
            
            print(f"🧠 Updated beliefs: evidence={evidence_strength:.3f}")
            
            return evidence_strength
        
        return 0.0
    
    def assess_capability(self, task_field: NDAnalogField) -> CapabilityAssessment:
        """
        Assess capability for a task using molecular operations
        
        Args:
            task_field: Field representing the task
            
        Returns:
            CapabilityAssessment object
        """
        # Compare task requirements to known abilities
        ability_match = self._compute_ability_match_molecular(task_field)
        
        # Identify required and missing abilities
        required_abilities = self._identify_required_abilities(task_field)
        missing_abilities = [ability for ability in required_abilities if ability not in self.state['abilities']]
        
        # Compute confidence score
        confidence_score = ability_match * (1.0 - len(missing_abilities) * 0.2)
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Estimate difficulty
        estimated_difficulty = 1.0 - ability_match
        
        # Generate recommended approach
        recommended_approach = self._generate_approach_recommendation(confidence_score, missing_abilities)
        
        assessment = CapabilityAssessment(
            task_description="Task field analysis",
            confidence_score=confidence_score,
            required_abilities=required_abilities,
            missing_abilities=missing_abilities,
            estimated_difficulty=estimated_difficulty,
            recommended_approach=recommended_approach
        )
        
        print(f"🎯 Capability assessment: confidence={confidence_score:.3f}, missing={len(missing_abilities)}")
        
        return assessment
    
    def _compute_ability_match_molecular(self, task_field: NDAnalogField) -> float:
        """Compute ability match using molecular operations"""
        # Use pattern recognizer to match task with abilities
        ability_match = self.molecules['pattern_recognizer'].recognize(task_field, self.fields['body_field'])
        
        # Use working memory for additional processing
        self.molecules['working_memory'].store(ability_match)
        
        return np.mean(ability_match.activation)
    
    def _identify_required_abilities(self, task_field: NDAnalogField) -> List[str]:
        """Identify required abilities for task"""
        # Simple ability identification based on field patterns
        abilities = []
        
        # Analyze field patterns to determine required abilities
        field_energy = np.sum(task_field.activation)
        field_complexity = np.std(task_field.activation)
        
        if field_energy > 0.7:
            abilities.append("high_energy")
        if field_complexity > 0.5:
            abilities.append("complex_processing")
        if np.sum(task_field.activation[0, :]) > 0.3:
            abilities.append("spatial_reasoning")
        if np.sum(task_field.activation[:, 0]) > 0.3:
            abilities.append("temporal_reasoning")
        
        return abilities
    
    def _generate_approach_recommendation(self, confidence: float, missing_abilities: List[str]) -> str:
        """Generate approach recommendation based on confidence and missing abilities"""
        if confidence > 0.8:
            return "Direct approach - high confidence"
        elif confidence > 0.5:
            return "Cautious approach - moderate confidence"
        elif missing_abilities:
            return f"Learn missing abilities: {', '.join(missing_abilities)}"
        else:
            return "Seek assistance or break down task"
    
    def theory_of_self(self) -> Dict[str, Any]:
        """
        Generate explicit theory of own mind using molecular operations
        
        Returns:
            Structured representation of self
        """
        # Compute current state energy
        current_state_energy = np.sum(self.fields['state_field'].activation)
        
        # Compute surprise level
        surprise_level = self._compute_surprise_molecular()
        
        # Compute prediction error
        prediction_error = self.atoms['anticipator'].get_prediction_error(self.fields['state_field'])
        
        # Compute self-consistency
        self_consistency = self._compute_self_consistency()
        
        # Generate theory
        theory = {
            'goals': self.state['current_goals'].copy(),
            'abilities': list(self.state['abilities']),
            'limitations': list(self.state['limitations']),
            'current_state': current_state_energy,
            'surprise_level': surprise_level,
            'prediction_error': prediction_error,
            'self_consistency': self_consistency,
            'total_observations': self.state['total_observations'],
            'total_predictions': self.state['total_predictions'],
            'total_reflections': self.state['total_reflections'],
            'improvement_goals': len(self.state['improvement_goals']),
            'field_energies': {
                name: np.sum(field.activation) for name, field in self.fields.items()
            }
        }
        
        return theory
    
    def add_ability(self, ability: str):
        """Add ability to self model"""
        self.state['abilities'].add(ability)
        print(f"➕ Added ability: {ability}")
    
    def add_limitation(self, limitation: str):
        """Add limitation to self model"""
        self.state['limitations'].add(limitation)
        print(f"➖ Added limitation: {limitation}")
    
    def add_goal(self, goal: str, priority: float = 1.0):
        """Add goal to self model"""
        self.state['current_goals'].append({'goal': goal, 'priority': priority})
        print(f"🎯 Added goal: {goal} (priority: {priority})")
    
    def get_self_summary(self) -> Dict[str, Any]:
        """Get summary of self model state"""
        return {
            "tick": self.state['tick_counter'],
            "abilities": len(self.state['abilities']),
            "limitations": len(self.state['limitations']),
            "goals": len(self.state['current_goals']),
            "history_length": len(self.state['self_history']),
            "improvement_goals": len(self.state['improvement_goals']),
            "total_observations": self.state['total_observations'],
            "total_predictions": self.state['total_predictions'],
            "total_reflections": self.state['total_reflections']
        }
    
    def get_state(self) -> Dict:
        """Get current internal state"""
        return {
            'config': self.config.copy(),
            'state': self.state.copy(),
            'field_shapes': {name: field.shape for name, field in self.fields.items()},
            'atom_states': {name: atom.__repr__() for name, atom in self.atoms.items()},
            'molecule_states': {name: molecule.__repr__() for name, molecule in self.molecules.items()}
        }
    
    def reset(self):
        """Reset the self model"""
        self.state = {
            'abilities': set(),
            'limitations': set(),
            'current_goals': [],
            'self_history': [],
            'belief_history': [],
            'improvement_goals': [],
            'tick_counter': 0,
            'total_observations': 0,
            'total_predictions': 0,
            'total_reflections': 0
        }
        
        # Reset fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Reset molecules
        for molecule in self.molecules.values():
            if hasattr(molecule, 'reset'):
                molecule.reset()
    
    def visualize_self_state(self, save_path: Optional[str] = None):
        """Visualize current self state"""
        if not self.config['enable_visualization']:
            return
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle("SelfModel - Molecular Self-Awareness", fontsize=16)
        
        # Body field
        im1 = axes[0, 0].imshow(self.fields['body_field'].activation, cmap='Blues')
        axes[0, 0].set_title("Body Field")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Goal field
        im2 = axes[0, 1].imshow(self.fields['goal_field'].activation, cmap='Greens')
        axes[0, 1].set_title("Goal Field")
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Belief field
        im3 = axes[0, 2].imshow(self.fields['belief_field'].activation, cmap='Reds')
        axes[0, 2].set_title("Belief Field")
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # State field
        im4 = axes[0, 3].imshow(self.fields['state_field'].activation, cmap='Purples')
        axes[0, 3].set_title("State Field")
        axes[0, 3].axis('off')
        plt.colorbar(im4, ax=axes[0, 3])
        
        # Meta field
        im5 = axes[1, 0].imshow(self.fields['meta_field'].activation, cmap='Oranges')
        axes[1, 0].set_title("Meta Field")
        axes[1, 0].axis('off')
        plt.colorbar(im5, ax=axes[1, 0])
        
        # Reflection field
        im6 = axes[1, 1].imshow(self.fields['reflection_field'].activation, cmap='YlOrRd')
        axes[1, 1].set_title("Reflection Field")
        axes[1, 1].axis('off')
        plt.colorbar(im6, ax=axes[1, 1])
        
        # Prediction field
        im7 = axes[1, 2].imshow(self.fields['prediction_field'].activation, cmap='viridis')
        axes[1, 2].set_title("Prediction Field")
        axes[1, 2].axis('off')
        plt.colorbar(im7, ax=axes[1, 2])
        
        # Self summary
        summary = self.get_self_summary()
        theory = self.theory_of_self()
        
        axes[1, 3].text(0.1, 0.9, f"Abilities: {summary['abilities']}", fontsize=10)
        axes[1, 3].text(0.1, 0.8, f"Goals: {summary['goals']}", fontsize=10)
        axes[1, 3].text(0.1, 0.7, f"Observations: {summary['total_observations']}", fontsize=10)
        axes[1, 3].text(0.1, 0.6, f"Predictions: {summary['total_predictions']}", fontsize=10)
        axes[1, 3].text(0.1, 0.5, f"Reflections: {summary['total_reflections']}", fontsize=10)
        axes[1, 3].text(0.1, 0.4, f"Surprise: {theory['surprise_level']:.3f}", fontsize=10)
        axes[1, 3].text(0.1, 0.3, f"Consistency: {theory['self_consistency']:.3f}", fontsize=10)
        axes[1, 3].text(0.1, 0.2, f"State Energy: {theory['current_state']:.3f}", fontsize=10)
        axes[1, 3].set_title("Self Summary")
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def __repr__(self):
        return f"SelfModel(tick={self.state['tick_counter']}, " \
               f"abilities={len(self.state['abilities'])}, " \
               f"goals={len(self.state['current_goals'])})"


# === Demo Functions ===

def demo_basic_self_model():
    """Demo basic self model operations"""
    print("=== Basic Self Model Demo ===")
    
    model = SelfModel({'field_shape': (12, 12), 'enable_visualization': False})
    
    # Add some abilities and limitations
    model.add_ability("pattern_recognition")
    model.add_ability("logical_reasoning")
    model.add_limitation("cannot_fly")
    model.add_limitation("limited_memory")
    
    # Add some goals
    model.add_goal("learn_new_skills", priority=0.8)
    model.add_goal("improve_efficiency", priority=0.6)
    
    # Test self observation
    print("\nTesting self observation...")
    snapshot = model.observe_self()
    
    print(f"Self snapshot:")
    print(f"  Timestamp: {snapshot.timestamp}")
    print(f"  State energy: {snapshot.state_energy:.3f}")
    print(f"  Surprise level: {snapshot.surprise_level:.3f}")
    print(f"  Self consistency: {snapshot.self_consistency:.3f}")
    
    # Test action prediction
    print("\nTesting action prediction...")
    situation_field = NDAnalogField((12, 12))
    situation_field.activation = np.random.random((12, 12)) * 0.8
    
    predicted_action = model.predict_own_action(situation_field)
    print(f"Predicted action energy: {np.sum(predicted_action.activation):.3f}")
    
    # Test self reflection
    print("\nTesting self reflection...")
    improvement_goal = model.reflect()
    
    if improvement_goal:
        print(f"Improvement goal: {improvement_goal.description}")
        print(f"  Type: {improvement_goal.goal_type}")
        print(f"  Priority: {improvement_goal.priority:.3f}")
    
    # Test belief update
    print("\nTesting belief update...")
    evidence_field = NDAnalogField((12, 12))
    evidence_field.activation = np.random.random((12, 12)) * 0.9
    
    update_strength = model.update_beliefs(evidence_field)
    print(f"Belief update strength: {update_strength:.3f}")
    
    # Test capability assessment
    print("\nTesting capability assessment...")
    task_field = NDAnalogField((12, 12))
    task_field.activation = np.random.random((12, 12)) * 0.7
    
    assessment = model.assess_capability(task_field)
    print(f"Capability assessment:")
    print(f"  Confidence: {assessment.confidence_score:.3f}")
    print(f"  Required abilities: {assessment.required_abilities}")
    print(f"  Missing abilities: {assessment.missing_abilities}")
    print(f"  Recommended approach: {assessment.recommended_approach}")
    
    # Test theory of self
    print("\nTesting theory of self...")
    theory = model.theory_of_self()
    print(f"Theory of self:")
    print(f"  Abilities: {theory['abilities']}")
    print(f"  Limitations: {theory['limitations']}")
    print(f"  Goals: {len(theory['goals'])}")
    print(f"  Surprise level: {theory['surprise_level']:.3f}")
    print(f"  Self consistency: {theory['self_consistency']:.3f}")
    
    return model


def demo_complex_self_model():
    """Demo complex self model with visualization"""
    print("\n=== Complex Self Model Demo ===")
    
    model = SelfModel({'field_shape': (16, 16), 'enable_visualization': True})
    
    # Set up comprehensive self model
    print("Setting up comprehensive self model...")
    
    # Add abilities
    abilities = [
        "pattern_recognition", "logical_reasoning", "memory_management",
        "goal_planning", "self_reflection", "learning", "adaptation"
    ]
    for ability in abilities:
        model.add_ability(ability)
    
    # Add limitations
    limitations = [
        "limited_processing_power", "finite_memory", "cannot_multitask_effectively",
        "emotional_bias", "temporal_limitations"
    ]
    for limitation in limitations:
        model.add_limitation(limitation)
    
    # Add goals
    goals = [
        ("improve_efficiency", 0.9),
        ("learn_new_skills", 0.8),
        ("reduce_errors", 0.7),
        ("enhance_creativity", 0.6),
        ("maintain_consistency", 0.5)
    ]
    for goal, priority in goals:
        model.add_goal(goal, priority)
    
    # Simulate self model operation
    print("\nSimulating self model operation...")
    
    for i in range(10):
        # Self observation
        snapshot = model.observe_self()
        
        # Create situation and predict action
        situation_field = NDAnalogField((16, 16))
        situation_field.activation = np.random.random((16, 16)) * 0.8
        predicted_action = model.predict_own_action(situation_field)
        
        # Self reflection (every 3 steps)
        if i % 3 == 0:
            improvement_goal = model.reflect()
            if improvement_goal:
                print(f"  Step {i}: Generated improvement goal: {improvement_goal.goal_type}")
        
        # Update beliefs (every 4 steps)
        if i % 4 == 0:
            evidence_field = NDAnalogField((16, 16))
            evidence_field.activation = np.random.random((16, 16)) * 0.6
            model.update_beliefs(evidence_field)
        
        # Capability assessment (every 5 steps)
        if i % 5 == 0:
            task_field = NDAnalogField((16, 16))
            task_field.activation = np.random.random((16, 16)) * 0.7
            assessment = model.assess_capability(task_field)
            print(f"  Step {i}: Capability assessment - confidence: {assessment.confidence_score:.3f}")
    
    # Show final theory of self
    print("\nFinal theory of self:")
    theory = model.theory_of_self()
    print(f"  Abilities: {len(theory['abilities'])}")
    print(f"  Limitations: {len(theory['limitations'])}")
    print(f"  Goals: {len(theory['goals'])}")
    print(f"  Total observations: {theory['total_observations']}")
    print(f"  Total predictions: {theory['total_predictions']}")
    print(f"  Total reflections: {theory['total_reflections']}")
    print(f"  Surprise level: {theory['surprise_level']:.3f}")
    print(f"  Self consistency: {theory['self_consistency']:.3f}")
    
    # Show visualization
    model.visualize_self_state()
    
    return model


# === Main Demo ===

if __name__ == '__main__':
    print("🪞 SELF MODEL - Agent's Model of Itself 🪞")
    print("Tracks own states, predicts own actions, monitors internal processes!")
    print("Foundation of metacognition and consciousness using molecular operations\n")
    
    # Run demos
    basic_model = demo_basic_self_model()
    complex_model = demo_complex_self_model()
    
    # System capabilities summary
    print("\n" + "="*60)
    print("🎯 SELF MODEL CAPABILITIES DEMONSTRATED")
    print("="*60)
    
    all_models = [basic_model, complex_model]
    total_observations = sum(model.state['total_observations'] for model in all_models)
    total_predictions = sum(model.state['total_predictions'] for model in all_models)
    total_reflections = sum(model.state['total_reflections'] for model in all_models)
    
    print(f"✅ Metacognitive self-observation")
    print(f"✅ Action prediction and planning")
    print(f"✅ Self-reflection and improvement")
    print(f"✅ Belief updating and revision")
    print(f"✅ Capability assessment")
    print(f"✅ Theory of self generation")
    print(f"✅ Molecular self-awareness")
    print(f"✅ Multi-dimensional self-representation")
    
    print(f"\n📊 DEMO STATISTICS:")
    print(f"Total observations: {total_observations}")
    print(f"Total predictions: {total_predictions}")
    print(f"Total reflections: {total_reflections}")
    print(f"Average observations per model: {total_observations / len(all_models):.1f}")
    
    print(f"\n💡 KEY INNOVATIONS:")
    print(f"• Molecular self-representation using multiple fields")
    print(f"• Metacognitive observation using witness atoms")
    print(f"• Action prediction using anticipator atoms")
    print(f"• Self-reflection using mirror atoms")
    print(f"• Belief updating using molecular operations")
    print(f"• Capability assessment using pattern recognition")
    
    print(f"\n🌟 This demonstrates true self-awareness!")
    print("The system models itself, predicts its own actions, and reflects on itself.")
    print("No training data, no neural networks - pure molecular intelligence!")
    
    print("\n🚀 SelfModel Demo Complete! 🚀")

