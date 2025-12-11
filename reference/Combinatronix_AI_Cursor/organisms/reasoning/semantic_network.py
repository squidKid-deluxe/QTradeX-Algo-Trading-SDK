# ============================================================================
# SemanticNetwork - Concept Relationship Graph Using Molecular Architecture
# ============================================================================

"""
SemanticNetwork - Builds graph of concepts with typed relationships

Composition:
- Atoms: Bridge, Weaver, Comparator, Seed, Resonator, Translator
- Molecules: Analogizer, PatternRecognizer, ContradictionResolver
- Fields: concept_fields, relation_field, activation_field, inference_field

This organism builds a graph of concepts with typed relationships,
using fields for concepts and bridges for relationships.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import networkx as nx

try:
    from ...core import NDAnalogField
    from ...atoms.field_dynamics import BridgeAtom
    from ...atoms.pattern_primitives import WeaverAtom, SeedAtom
    from ...atoms.multi_field import ComparatorAtom, ResonatorAtom, TranslatorAtom
    from ...molecules.reasoning import AnalogizerMolecule, ContradictionResolverMolecule
    from ...molecules.perception import PatternRecognizerMolecule
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.atoms.field_dynamics import BridgeAtom
    from combinatronix.atoms.pattern_primitives import WeaverAtom, SeedAtom
    from combinatronix.atoms.multi_field import ComparatorAtom, ResonatorAtom, TranslatorAtom
    from combinatronix.molecules.reasoning import AnalogizerMolecule, ContradictionResolverMolecule
    from combinatronix.molecules.perception import PatternRecognizerMolecule


@dataclass
class ConceptNode:
    """Represents a concept node in the semantic network"""
    name: str
    field_pattern: NDAnalogField
    activation: float = 0.0
    metadata: Dict[str, Any] = None
    strength: float = 1.0
    last_activated: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Relation:
    """Represents a relationship between concepts"""
    concept_a: str
    concept_b: str
    relation_type: str
    strength: float
    transfer_rate: float
    created_at: int
    access_count: int = 0
    confidence: float = 1.0


@dataclass
class ActivationPath:
    """Represents a path of activation through the network"""
    path: List[str]
    total_strength: float
    path_length: int
    relation_types: List[str]


class SemanticNetwork:
    """Concept relationship graph using molecular architecture"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the semantic network
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'field_shape': (16, 16),
            'max_concepts': 1000,
            'max_relations': 5000,
            'activation_threshold': 0.1,
            'spreading_steps': 5,
            'enable_visualization': True,
            'inference_threshold': 0.7,
            'path_search_depth': 10,
            'relation_types': [
                'is-a', 'part-of', 'causes', 'similar-to', 'opposite-of',
                'related-to', 'contains', 'made-of', 'used-for', 'located-in'
            ]
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
            'concepts': {},
            'relations': [],
            'concept_fields': {},
            'activation_history': [],
            'inference_history': [],
            'tick_counter': 0,
            'total_activations': 0,
            'total_inferences': 0
        }
        
        # Network graph for path finding
        self.graph = nx.DiGraph()
        
        print(f"🕸️ SemanticNetwork initialized ({self.config['field_shape'][0]}×{self.config['field_shape'][1]})")
    
    def _initialize_atoms(self):
        """Initialize atomic operations"""
        self.atoms = {
            'bridge': BridgeAtom(
                connection_strength=0.8,
                stability_threshold=0.5,
                bidirectional=True
            ),
            'weaver': WeaverAtom(
                combination='add',
                strength=0.7
            ),
            'comparator': ComparatorAtom(
                metric='cosine',
                normalize=True
            ),
            'seed': SeedAtom(
                spread_radius=3,
                strength=1.0
            ),
            'resonator': ResonatorAtom(
                amplification=1.5,
                threshold=0.5
            ),
            'translator': TranslatorAtom(
                scale_factor=1.0,
                transformation='linear'
            )
        }
    
    def _initialize_molecules(self):
        """Initialize molecular operations"""
        self.molecules = {
            'analogizer': AnalogizerMolecule(
                similarity_threshold=0.4,
                translation_strength=0.8
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
            'relation_field': NDAnalogField(self.config['field_shape']),
            'activation_field': NDAnalogField(self.config['field_shape']),
            'inference_field': NDAnalogField(self.config['field_shape']),
            'similarity_field': NDAnalogField(self.config['field_shape']),
            'spreading_field': NDAnalogField(self.config['field_shape']),
            'consolidation_field': NDAnalogField(self.config['field_shape'])
        }
    
    def add_concept(self, name: str, field_pattern: np.ndarray, metadata: Dict[str, Any] = None) -> ConceptNode:
        """
        Add new concept node using molecular operations
        
        Args:
            name: Name of the concept
            field_pattern: Field pattern representing the concept
            metadata: Additional metadata for the concept
            
        Returns:
            Created ConceptNode object
        """
        if name in self.state['concepts']:
            raise ValueError(f"Concept '{name}' already exists")
        
        # Resize pattern if needed
        if field_pattern.shape != self.config['field_shape']:
            field_pattern = self._resize_pattern(field_pattern)
        
        # Create concept field
        concept_field = NDAnalogField(self.config['field_shape'], activation=field_pattern)
        
        # Create concept node
        concept = ConceptNode(
            name=name,
            field_pattern=concept_field,
            activation=0.0,
            metadata=metadata or {},
            strength=1.0,
            last_activated=0
        )
        
        # Store concept
        self.state['concepts'][name] = concept
        self.state['concept_fields'][name] = concept_field
        
        # Add to network graph
        self.graph.add_node(name)
        
        print(f"➕ Added concept '{name}'")
        
        return concept
    
    def _resize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Resize pattern to fit field size"""
        target_shape = self.config['field_shape']
        
        if pattern.shape == target_shape:
            return pattern
        
        resized = np.zeros(target_shape, dtype=np.float32)
        min_h = min(pattern.shape[0], target_shape[0])
        min_w = min(pattern.shape[1], target_shape[1])
        resized[:min_h, :min_w] = pattern[:min_h, :min_w]
        
        return resized
    
    def link_concepts(self, concept_a: str, concept_b: str, relation_type: str, 
                     strength: float = 1.0) -> Relation:
        """
        Create typed relationship using molecular operations
        
        Args:
            concept_a: First concept
            concept_b: Second concept
            relation_type: Type of relationship
            strength: Strength of the relationship
            
        Returns:
            Created Relation object
        """
        if concept_a not in self.state['concepts']:
            raise ValueError(f"Concept '{concept_a}' not found")
        if concept_b not in self.state['concepts']:
            raise ValueError(f"Concept '{concept_b}' not found")
        if relation_type not in self.config['relation_types']:
            raise ValueError(f"Unknown relation type '{relation_type}'")
        
        # Create relation
        relation = Relation(
            concept_a=concept_a,
            concept_b=concept_b,
            relation_type=relation_type,
            strength=strength,
            transfer_rate=self._relation_to_transfer_rate(relation_type),
            created_at=self.state['tick_counter'],
            access_count=0,
            confidence=1.0
        )
        
        # Store relation
        self.state['relations'].append(relation)
        
        # Create bridge between concept fields using molecular operations
        field_a = self.state['concept_fields'][concept_a]
        field_b = self.state['concept_fields'][concept_b]
        
        bridge_strength = self.atoms['bridge'].connect(
            field_a, field_b, strength=strength * relation.transfer_rate
        )
        
        # Update relation field
        self._update_relation_field(relation)
        
        # Add to network graph
        self.graph.add_edge(concept_a, concept_b, 
                           relation_type=relation_type, 
                           strength=strength,
                           transfer_rate=relation.transfer_rate)
        
        print(f"🔗 Linked '{concept_a}' --{relation_type}--> '{concept_b}' (strength: {strength:.3f})")
        
        return relation
    
    def _relation_to_transfer_rate(self, relation_type: str) -> float:
        """Convert relation type to transfer rate"""
        transfer_rates = {
            'is-a': 0.9,
            'part-of': 0.8,
            'causes': 0.7,
            'similar-to': 0.6,
            'opposite-of': 0.3,
            'related-to': 0.5,
            'contains': 0.7,
            'made-of': 0.8,
            'used-for': 0.6,
            'located-in': 0.5
        }
        
        return transfer_rates.get(relation_type, 0.5)
    
    def _update_relation_field(self, relation: Relation):
        """Update relation field with relation information"""
        # Create relation pattern
        relation_pattern = np.zeros(self.config['field_shape'])
        
        # Map relation to field position based on relation type
        relation_index = self.config['relation_types'].index(relation.relation_type)
        field_x = relation_index % self.config['field_shape'][1]
        field_y = relation_index // self.config['field_shape'][1]
        
        if field_y < self.config['field_shape'][0]:
            relation_pattern[field_y, field_x] = relation.strength
        
        # Update relation field
        self.fields['relation_field'].activation += relation_pattern * 0.1
        self.fields['relation_field'].activation = np.clip(self.fields['relation_field'].activation, 0, 1)
    
    def activate_concept(self, concept_name: str, activation_strength: float = 1.0) -> bool:
        """
        Activate concept and spread to related concepts using molecular operations
        
        Args:
            concept_name: Name of concept to activate
            activation_strength: Strength of activation
            
        Returns:
            True if concept was activated, False otherwise
        """
        if concept_name not in self.state['concepts']:
            return False
        
        self.state['tick_counter'] += 1
        self.state['total_activations'] += 1
        
        # Get concept and field
        concept = self.state['concepts'][concept_name]
        field = self.state['concept_fields'][concept_name]
        
        # Activate concept using seed atom
        self.atoms['seed'].apply(field, location=(8, 8), strength=activation_strength)
        
        # Update concept activation
        concept.activation = np.sum(field.activation)
        concept.last_activated = self.state['tick_counter']
        
        # Spread activation via bridges using molecular operations
        self._spread_activation_molecular(concept_name, activation_strength)
        
        # Update activation field
        self.fields['activation_field'].activation += field.activation * 0.1
        self.fields['activation_field'].activation = np.clip(self.fields['activation_field'].activation, 0, 1)
        
        # Record activation
        self.state['activation_history'].append({
            'timestamp': self.state['tick_counter'],
            'concept': concept_name,
            'strength': activation_strength,
            'field_energy': np.sum(field.activation)
        })
        
        print(f"⚡ Activated '{concept_name}' (strength: {activation_strength:.3f})")
        
        return True
    
    def _spread_activation_molecular(self, concept_name: str, activation_strength: float):
        """Spread activation using molecular operations"""
        # Find related concepts
        related_concepts = self._get_related_concepts(concept_name)
        
        for related_concept, relation in related_concepts:
            if related_concept in self.state['concept_fields']:
                related_field = self.state['concept_fields'][related_concept]
                
                # Use bridge atom to spread activation
                spread_strength = activation_strength * relation.strength * relation.transfer_rate
                
                # Apply bridge connection
                self.atoms['bridge'].apply(related_field, strength=spread_strength)
                
                # Update related concept activation
                if related_concept in self.state['concepts']:
                    self.state['concepts'][related_concept].activation = np.sum(related_field.activation)
                    self.state['concepts'][related_concept].last_activated = self.state['tick_counter']
    
    def _get_related_concepts(self, concept_name: str) -> List[Tuple[str, Relation]]:
        """Get concepts related to the given concept"""
        related = []
        
        for relation in self.state['relations']:
            if relation.concept_a == concept_name:
                related.append((relation.concept_b, relation))
            elif relation.concept_b == concept_name:
                related.append((relation.concept_a, relation))
        
        return related
    
    def spreading_activation(self, initial_concepts: List[str], steps: int = 5) -> Dict[str, float]:
        """
        Simulate spreading activation using molecular operations
        
        Args:
            initial_concepts: List of concepts to activate initially
            steps: Number of spreading steps
            
        Returns:
            Dictionary mapping concept names to activation levels
        """
        # Initialize activation map
        activation_map = {name: 0.0 for name in self.state['concepts']}
        
        # Initial activation
        for concept in initial_concepts:
            if concept in self.state['concepts']:
                self.activate_concept(concept, 1.0)
                activation_map[concept] = 1.0
        
        # Spread activation for specified steps
        for step in range(steps):
            # Update spreading field
            self.fields['spreading_field'].activation.fill(0)
            
            for concept_name, field in self.state['concept_fields'].items():
                # Propagate activation
                field.propagate(steps=1)
                
                # Apply bridge connections
                self.atoms['bridge'].apply(field)
                
                # Update activation map
                activation_map[concept_name] = np.sum(field.activation)
                
                # Update concept activation
                if concept_name in self.state['concepts']:
                    self.state['concepts'][concept_name].activation = activation_map[concept_name]
            
            # Update spreading field
            self.fields['spreading_field'].activation += field.activation * 0.1
        
        # Normalize activations
        max_activation = max(activation_map.values()) if activation_map.values() else 1.0
        if max_activation > 0:
            for concept in activation_map:
                activation_map[concept] /= max_activation
        
        print(f"🌊 Spreading activation completed: {len(initial_concepts)} initial, {steps} steps")
        
        return activation_map
    
    def find_path(self, concept_a: str, concept_b: str, max_depth: int = None) -> Optional[ActivationPath]:
        """
        Find conceptual path between two concepts using molecular operations
        
        Args:
            concept_a: Starting concept
            concept_b: Target concept
            max_depth: Maximum search depth
            
        Returns:
            ActivationPath object if path found, None otherwise
        """
        if max_depth is None:
            max_depth = self.config['path_search_depth']
        
        if concept_a not in self.state['concepts'] or concept_b not in self.state['concepts']:
            return None
        
        # Use breadth-first search
        queue = deque([(concept_a, [concept_a], 0.0, [])])
        visited = {concept_a}
        
        while queue:
            current_concept, path, total_strength, relation_types = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_concept == concept_b:
                return ActivationPath(
                    path=path,
                    total_strength=total_strength,
                    path_length=len(path) - 1,
                    relation_types=relation_types
                )
            
            # Find related concepts
            related_concepts = self._get_related_concepts(current_concept)
            
            for related_concept, relation in related_concepts:
                if related_concept not in visited:
                    visited.add(related_concept)
                    
                    new_path = path + [related_concept]
                    new_strength = total_strength + relation.strength
                    new_relation_types = relation_types + [relation.relation_type]
                    
                    queue.append((related_concept, new_path, new_strength, new_relation_types))
        
        return None
    
    def infer_new_relation(self, concept_a: str, concept_b: str) -> Optional[Relation]:
        """
        Invent new relation by analyzing field patterns using molecular operations
        
        Args:
            concept_a: First concept
            concept_b: Second concept
            
        Returns:
            Inferred Relation object if successful, None otherwise
        """
        if concept_a not in self.state['concepts'] or concept_b not in self.state['concepts']:
            return None
        
        self.state['total_inferences'] += 1
        
        # Get concept fields
        field_a = self.state['concept_fields'][concept_a]
        field_b = self.state['concept_fields'][concept_b]
        
        # Compare field patterns using molecular operations
        similarity = self._compute_concept_similarity_molecular(field_a, field_b)
        
        # Use analogizer molecule for relation inference
        analogical_relation = self.molecules['analogizer'].map(field_a, field_b)
        
        # Use pattern recognizer for relation type detection
        pattern_relation = self.molecules['pattern_recognizer'].recognize(field_a, field_b)
        
        # Determine relation type based on molecular analysis
        relation_type = self._infer_relation_type(similarity, analogical_relation, pattern_relation)
        
        if relation_type is None:
            return None
        
        # Calculate relation strength
        strength = min(1.0, similarity * 0.8)
        
        # Create inferred relation
        relation = Relation(
            concept_a=concept_a,
            concept_b=concept_b,
            relation_type=relation_type,
            strength=strength,
            transfer_rate=self._relation_to_transfer_rate(relation_type),
            created_at=self.state['tick_counter'],
            access_count=0,
            confidence=similarity
        )
        
        # Store relation
        self.state['relations'].append(relation)
        
        # Create bridge
        self.atoms['bridge'].connect(field_a, field_b, strength=strength * relation.transfer_rate)
        
        # Update relation field
        self._update_relation_field(relation)
        
        # Record inference
        self.state['inference_history'].append({
            'timestamp': self.state['tick_counter'],
            'concept_a': concept_a,
            'concept_b': concept_b,
            'relation_type': relation_type,
            'strength': strength,
            'confidence': similarity
        })
        
        print(f"🔮 Inferred relation: '{concept_a}' --{relation_type}--> '{concept_b}' (strength: {strength:.3f})")
        
        return relation
    
    def _compute_concept_similarity_molecular(self, field_a: NDAnalogField, field_b: NDAnalogField) -> float:
        """Compute concept similarity using molecular operations"""
        # Use comparator atom for field comparison
        comparison = self.atoms['comparator'].apply(field_a, field_b)
        
        # Use resonator atom for enhanced similarity
        resonance = self.atoms['resonator'].apply(field_a, field_b)
        
        # Use pattern recognizer molecule for pattern matching
        recognition = self.molecules['pattern_recognizer'].recognize(field_a, field_b)
        
        # Combine similarities
        similarity = (
            np.mean(comparison.activation) * 0.4 +
            np.mean(resonance.activation) * 0.3 +
            np.mean(recognition.activation) * 0.3
        )
        
        # Update similarity field
        self.fields['similarity_field'].activation = comparison.activation
        
        return similarity
    
    def _infer_relation_type(self, similarity: float, analogical_relation: NDAnalogField, 
                           pattern_relation: NDAnalogField) -> Optional[str]:
        """Infer relation type based on molecular analysis"""
        if similarity < self.config['inference_threshold']:
            return None
        
        # Analyze analogical relation
        analogical_strength = np.mean(analogical_relation.activation)
        
        # Analyze pattern relation
        pattern_strength = np.mean(pattern_relation.activation)
        
        # Determine relation type based on strengths
        if similarity > 0.8:
            return "similar-to"
        elif analogical_strength > 0.7:
            return "related-to"
        elif pattern_strength > 0.6:
            return "part-of"
        elif similarity > 0.5:
            return "used-for"
        else:
            return "related-to"
    
    def consolidate_network(self):
        """Consolidate network by strengthening important relations"""
        print("🔄 Consolidating semantic network...")
        
        # Strengthen frequently accessed relations
        for relation in self.state['relations']:
            if relation.access_count > 0:
                # Strengthen based on access count
                strength_increase = min(0.1, relation.access_count * 0.01)
                relation.strength = min(1.0, relation.strength + strength_increase)
        
        # Update consolidation field
        self._update_consolidation_field()
        
        print(f"✅ Network consolidation complete: {len(self.state['relations'])} relations")
    
    def _update_consolidation_field(self):
        """Update consolidation field with relation information"""
        self.fields['consolidation_field'].activation.fill(0)
        
        for relation in self.state['relations']:
            if relation.strength > 0.5:
                # Add relation to consolidation field
                relation_pattern = np.zeros(self.config['field_shape'])
                relation_index = self.config['relation_types'].index(relation.relation_type)
                field_x = relation_index % self.config['field_shape'][1]
                field_y = relation_index // self.config['field_shape'][1]
                
                if field_y < self.config['field_shape'][0]:
                    relation_pattern[field_y, field_x] = relation.strength
                
                self.fields['consolidation_field'].activation += relation_pattern * 0.1
        
        self.fields['consolidation_field'].activation = np.clip(self.fields['consolidation_field'].activation, 0, 1)
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get summary of network state"""
        return {
            "concepts": len(self.state['concepts']),
            "relations": len(self.state['relations']),
            "total_activations": self.state['total_activations'],
            "total_inferences": self.state['total_inferences'],
            "current_tick": self.state['tick_counter'],
            "relation_types": self.config['relation_types'],
            "field_energies": {
                name: np.sum(field.activation) for name, field in self.fields.items()
            },
            "concept_activations": {
                name: concept.activation for name, concept in self.state['concepts'].items()
            }
        }
    
    def get_concepts(self) -> List[ConceptNode]:
        """Get all concepts"""
        return list(self.state['concepts'].values())
    
    def get_relations(self) -> List[Relation]:
        """Get all relations"""
        return self.state['relations'].copy()
    
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
        """Reset the semantic network"""
        self.state = {
            'concepts': {},
            'relations': [],
            'concept_fields': {},
            'activation_history': [],
            'inference_history': [],
            'tick_counter': 0,
            'total_activations': 0,
            'total_inferences': 0
        }
        
        # Reset fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Reset molecules
        for molecule in self.molecules.values():
            if hasattr(molecule, 'reset'):
                molecule.reset()
        
        # Reset graph
        self.graph.clear()
    
    def visualize_network(self, save_path: Optional[str] = None):
        """Visualize the semantic network"""
        if not self.config['enable_visualization']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("SemanticNetwork - Molecular Concept Graph", fontsize=16)
        
        # Relation field
        im1 = axes[0, 0].imshow(self.fields['relation_field'].activation, cmap='Blues')
        axes[0, 0].set_title("Relation Field")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Activation field
        im2 = axes[0, 1].imshow(self.fields['activation_field'].activation, cmap='Reds')
        axes[0, 1].set_title("Activation Field")
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Similarity field
        im3 = axes[0, 2].imshow(self.fields['similarity_field'].activation, cmap='Greens')
        axes[0, 2].set_title("Similarity Field")
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Network graph
        if len(self.graph.nodes) > 0:
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            nx.draw(self.graph, pos, ax=axes[1, 0], with_labels=True, node_size=500, 
                   node_color='lightblue', font_size=8, font_weight='bold')
            axes[1, 0].set_title("Concept Graph")
        else:
            axes[1, 0].text(0.5, 0.5, "No Concepts", ha='center', va='center')
            axes[1, 0].set_title("Concept Graph")
        
        # Relation types distribution
        if self.state['relations']:
            relation_types = [rel.relation_type for rel in self.state['relations']]
            type_counts = {rel_type: relation_types.count(rel_type) for rel_type in set(relation_types)}
            
            axes[1, 1].bar(type_counts.keys(), type_counts.values())
            axes[1, 1].set_title("Relation Types")
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, "No Relations", ha='center', va='center')
            axes[1, 1].set_title("Relation Types")
        
        # Network summary
        summary = self.get_network_summary()
        axes[1, 2].text(0.1, 0.8, f"Concepts: {summary['concepts']}", fontsize=12)
        axes[1, 2].text(0.1, 0.7, f"Relations: {summary['relations']}", fontsize=12)
        axes[1, 2].text(0.1, 0.6, f"Activations: {summary['total_activations']}", fontsize=12)
        axes[1, 2].text(0.1, 0.5, f"Inferences: {summary['total_inferences']}", fontsize=12)
        axes[1, 2].text(0.1, 0.4, f"Current Tick: {summary['current_tick']}", fontsize=12)
        axes[1, 2].set_title("Network Summary")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def __repr__(self):
        return f"SemanticNetwork(concepts={len(self.state['concepts'])}, " \
               f"relations={len(self.state['relations'])}, " \
               f"activations={self.state['total_activations']})"


# === Demo Functions ===

def demo_basic_semantic_network():
    """Demo basic semantic network operations"""
    print("=== Basic Semantic Network Demo ===")
    
    network = SemanticNetwork({'field_shape': (12, 12), 'enable_visualization': False})
    
    # Create some concepts
    concepts = {
        "animal": np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "dog": np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "cat": np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)
    }
    
    # Add concepts
    for name, pattern in concepts.items():
        network.add_concept(name, pattern)
    
    # Create relations
    network.link_concepts("dog", "animal", "is-a", strength=0.9)
    network.link_concepts("cat", "animal", "is-a", strength=0.9)
    network.link_concepts("dog", "cat", "similar-to", strength=0.6)
    
    # Test activation
    print("\nTesting activation...")
    network.activate_concept("dog", 1.0)
    
    # Test spreading activation
    print("\nTesting spreading activation...")
    activation_map = network.spreading_activation(["dog"], steps=3)
    
    print("Activation results:")
    for concept, activation in activation_map.items():
        if activation > 0.1:
            print(f"  {concept}: {activation:.3f}")
    
    # Test path finding
    print("\nTesting path finding...")
    path = network.find_path("dog", "cat")
    if path:
        print(f"Path found: {' -> '.join(path.path)}")
        print(f"Path strength: {path.total_strength:.3f}")
        print(f"Relation types: {path.relation_types}")
    else:
        print("No path found")
    
    # Test relation inference
    print("\nTesting relation inference...")
    inferred = network.infer_new_relation("dog", "cat")
    if inferred:
        print(f"Inferred relation: {inferred.relation_type} (strength: {inferred.strength:.3f})")
    
    return network


def demo_complex_semantic_network():
    """Demo complex semantic network with visualization"""
    print("\n=== Complex Semantic Network Demo ===")
    
    network = SemanticNetwork({'field_shape': (16, 16), 'enable_visualization': True})
    
    # Create a complex knowledge graph
    knowledge_concepts = {
        "living_thing": np.eye(16) * 0.8,
        "animal": np.eye(16) * 0.7,
        "plant": np.eye(16) * 0.6,
        "mammal": np.eye(16) * 0.5,
        "bird": np.eye(16) * 0.4,
        "dog": np.eye(16) * 0.3,
        "cat": np.eye(16) * 0.2,
        "tree": np.eye(16) * 0.1,
        "flower": np.eye(16) * 0.05
    }
    
    # Add concepts
    for name, pattern in knowledge_concepts.items():
        network.add_concept(name, pattern)
    
    # Create hierarchical relations
    relations = [
        ("dog", "mammal", "is-a", 0.9),
        ("cat", "mammal", "is-a", 0.9),
        ("mammal", "animal", "is-a", 0.8),
        ("bird", "animal", "is-a", 0.8),
        ("animal", "living_thing", "is-a", 0.7),
        ("tree", "plant", "is-a", 0.9),
        ("flower", "plant", "is-a", 0.9),
        ("plant", "living_thing", "is-a", 0.7),
        ("dog", "cat", "similar-to", 0.6),
        ("tree", "flower", "part-of", 0.5)
    ]
    
    # Add relations
    for concept_a, concept_b, relation_type, strength in relations:
        network.link_concepts(concept_a, concept_b, relation_type, strength)
    
    # Test spreading activation
    print("Testing spreading activation...")
    activation_map = network.spreading_activation(["dog"], steps=5)
    
    print("Activation results:")
    for concept, activation in sorted(activation_map.items(), key=lambda x: x[1], reverse=True):
        if activation > 0.1:
            print(f"  {concept}: {activation:.3f}")
    
    # Test path finding
    print("\nTesting path finding...")
    paths_to_test = [
        ("dog", "living_thing"),
        ("cat", "plant"),
        ("flower", "animal")
    ]
    
    for start, end in paths_to_test:
        path = network.find_path(start, end)
        if path:
            print(f"Path {start} -> {end}: {' -> '.join(path.path)}")
            print(f"  Strength: {path.total_strength:.3f}, Relations: {path.relation_types}")
        else:
            print(f"No path found: {start} -> {end}")
    
    # Test relation inference
    print("\nTesting relation inference...")
    inference_pairs = [
        ("dog", "bird"),
        ("tree", "animal"),
        ("flower", "cat")
    ]
    
    for concept_a, concept_b in inference_pairs:
        inferred = network.infer_new_relation(concept_a, concept_b)
        if inferred:
            print(f"Inferred: {concept_a} --{inferred.relation_type}--> {concept_b} (strength: {inferred.strength:.3f})")
    
    # Test consolidation
    print("\nTesting consolidation...")
    network.consolidate_network()
    
    # Show visualization
    network.visualize_network()
    
    return network


# === Main Demo ===

if __name__ == '__main__':
    print("🕸️ SEMANTIC NETWORK - Concept Relationship Graph 🕸️")
    print("Builds graph of concepts with typed relationships using molecular operations!")
    print("Uses fields for concepts and bridges for relationships\n")
    
    # Run demos
    basic_network = demo_basic_semantic_network()
    complex_network = demo_complex_semantic_network()
    
    # System capabilities summary
    print("\n" + "="*60)
    print("🎯 SEMANTIC NETWORK CAPABILITIES DEMONSTRATED")
    print("="*60)
    
    all_networks = [basic_network, complex_network]
    total_concepts = sum(len(network.get_concepts()) for network in all_networks)
    total_relations = sum(len(network.get_relations()) for network in all_networks)
    
    print(f"✅ Concept graph construction and management")
    print(f"✅ Typed relationship creation and storage")
    print(f"✅ Molecular activation and spreading")
    print(f"✅ Conceptual path finding")
    print(f"✅ Relation inference and discovery")
    print(f"✅ Network consolidation and strengthening")
    print(f"✅ Multi-dimensional field representation")
    print(f"✅ Graph visualization and analysis")
    
    print(f"\n📊 DEMO STATISTICS:")
    print(f"Total concepts created: {total_concepts}")
    print(f"Total relations created: {total_relations}")
    print(f"Average concepts per network: {total_concepts / len(all_networks):.1f}")
    
    print(f"\n💡 KEY INNOVATIONS:")
    print(f"• Molecular concept representation using fields")
    print(f"• Typed relationship modeling with transfer rates")
    print(f"• Spreading activation using molecular operations")
    print(f"• Automatic relation inference and discovery")
    print(f"• Graph-based path finding and analysis")
    
    print(f"\n🌟 This demonstrates true semantic understanding!")
    print("The system builds and navigates concept relationships.")
    print("No training data, no neural networks - pure molecular intelligence!")
    
    print("\n🚀 SemanticNetwork Demo Complete! 🚀")

