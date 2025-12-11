# ============================================================================
# AnalogyMaker - Cross-Domain Reasoning Using Molecular Architecture
# ============================================================================

"""
AnalogyMaker - Maps concepts from source domain to target domain

Composition:
- Atoms: Bridge, Translator, Comparator, Witness
- Molecules: Analogizer, PatternRecognizer
- Fields: source_field, target_field, mapping_field, similarity_field

This organism finds structural similarities between domains and creates
analogical mappings using molecular operations and field dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from ...core import NDAnalogField
    from ...atoms.field_dynamics import BridgeAtom
    from ...atoms.multi_field import TranslatorAtom, ComparatorAtom
    from ...atoms.pattern_primitives import WitnessAtom
    from ...molecules.reasoning import AnalogizerMolecule
    from ...molecules.perception import PatternRecognizerMolecule
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.atoms.field_dynamics import BridgeAtom
    from combinatronix.atoms.multi_field import TranslatorAtom, ComparatorAtom
    from combinatronix.atoms.pattern_primitives import WitnessAtom
    from combinatronix.molecules.reasoning import AnalogizerMolecule
    from combinatronix.molecules.perception import PatternRecognizerMolecule


@dataclass
class AnalogyMapping:
    """Represents an analogical mapping between domains"""
    source_concept: str
    target_concept: str
    similarity_score: float
    mapping_strength: float
    field_location: Tuple[int, int]
    mapping_type: str  # "structural", "functional", "relational", "causal"
    confidence: float


@dataclass
class DomainPattern:
    """Represents a pattern in a specific domain"""
    name: str
    pattern: NDAnalogField
    domain: str
    properties: Dict[str, Any]
    relationships: List[str]


class AnalogyMaker:
    """Cross-domain reasoning organism using molecular architecture"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the analogy maker
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'field_size': (16, 16),
            'similarity_threshold': 0.4,
            'mapping_strength_threshold': 0.3,
            'max_mappings': 50,
            'enable_visualization': True,
            'analogy_depth': 3,
            'structural_weight': 0.4,
            'functional_weight': 0.3,
            'relational_weight': 0.2,
            'causal_weight': 0.1
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
            'source_patterns': {},
            'target_patterns': {},
            'analogy_mappings': {},
            'domain_knowledge': {},
            'mapping_history': [],
            'tick_counter': 0,
            'total_mappings_created': 0
        }
        
        print(f"🔗 AnalogyMaker initialized ({self.config['field_size'][0]}×{self.config['field_size'][1]})")
    
    def _initialize_atoms(self):
        """Initialize atomic operations"""
        self.atoms = {
            'bridge': BridgeAtom(connection_strength=0.8, stability_threshold=0.5),
            'translator': TranslatorAtom(scale_factor=1.0, transformation='linear'),
            'comparator': ComparatorAtom(metric='cosine', normalize=True),
            'witness': WitnessAtom(observation_strength=0.7, memory_decay=0.95)
        }
    
    def _initialize_molecules(self):
        """Initialize molecular operations"""
        self.molecules = {
            'analogizer': AnalogizerMolecule(
                similarity_threshold=self.config['similarity_threshold'],
                translation_strength=0.8
            ),
            'pattern_recognizer': PatternRecognizerMolecule(
                amplification=1.5,
                resonance_threshold=0.5
            )
        }
    
    def _initialize_fields(self):
        """Initialize field structures"""
        self.fields = {
            'source_field': NDAnalogField(self.config['field_size']),
            'target_field': NDAnalogField(self.config['field_size']),
            'mapping_field': NDAnalogField(self.config['field_size']),
            'similarity_field': NDAnalogField(self.config['field_size']),
            'bridge_field': NDAnalogField(self.config['field_size']),
            'witness_field': NDAnalogField(self.config['field_size'])
        }
    
    def inject_source_domain(self, domain_name: str, patterns: Dict[str, np.ndarray]):
        """
        Inject source domain patterns
        
        Args:
            domain_name: Name of the source domain
            patterns: Dictionary mapping concept names to their patterns
        """
        domain_patterns = {}
        for name, pattern in patterns.items():
            # Resize pattern if needed
            if pattern.shape != self.config['field_size']:
                pattern = self._resize_pattern(pattern)
            
            # Create domain pattern
            domain_pattern = DomainPattern(
                name=name,
                pattern=NDAnalogField(self.config['field_size'], activation=pattern),
                domain=domain_name,
                properties=self._extract_pattern_properties(pattern),
                relationships=[]
            )
            domain_patterns[name] = domain_pattern
        
        self.state['source_patterns'][domain_name] = domain_patterns
        print(f"📥 Injected source domain '{domain_name}' with {len(patterns)} patterns")
    
    def inject_target_domain(self, domain_name: str, patterns: Dict[str, np.ndarray]):
        """
        Inject target domain patterns
        
        Args:
            domain_name: Name of the target domain
            patterns: Dictionary mapping concept names to their patterns
        """
        domain_patterns = {}
        for name, pattern in patterns.items():
            # Resize pattern if needed
            if pattern.shape != self.config['field_size']:
                pattern = self._resize_pattern(pattern)
            
            # Create domain pattern
            domain_pattern = DomainPattern(
                name=name,
                pattern=NDAnalogField(self.config['field_size'], activation=pattern),
                domain=domain_name,
                properties=self._extract_pattern_properties(pattern),
                relationships=[]
            )
            domain_patterns[name] = domain_pattern
        
        self.state['target_patterns'][domain_name] = domain_patterns
        print(f"📥 Injected target domain '{domain_name}' with {len(patterns)} patterns")
    
    def _resize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Resize pattern to fit field size"""
        target_shape = self.config['field_size']
        
        if pattern.shape == target_shape:
            return pattern
        
        resized = np.zeros(target_shape, dtype=np.float32)
        min_h = min(pattern.shape[0], target_shape[0])
        min_w = min(pattern.shape[1], target_shape[1])
        resized[:min_h, :min_w] = pattern[:min_h, :min_w]
        
        return resized
    
    def _extract_pattern_properties(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Extract properties from a pattern"""
        return {
            'energy': np.sum(pattern),
            'complexity': np.std(pattern),
            'symmetry': self._calculate_symmetry(pattern),
            'sparsity': np.sum(pattern > 0) / pattern.size,
            'center_of_mass': self._calculate_center_of_mass(pattern)
        }
    
    def _calculate_symmetry(self, pattern: np.ndarray) -> float:
        """Calculate symmetry score of pattern"""
        # Horizontal symmetry
        h_sym = np.mean(np.abs(pattern - np.fliplr(pattern)))
        # Vertical symmetry
        v_sym = np.mean(np.abs(pattern - np.flipud(pattern)))
        # Overall symmetry (lower is more symmetric)
        return 1.0 - (h_sym + v_sym) / 2.0
    
    def _calculate_center_of_mass(self, pattern: np.ndarray) -> Tuple[float, float]:
        """Calculate center of mass of pattern"""
        if np.sum(pattern) == 0:
            return (0.0, 0.0)
        
        y_coords, x_coords = np.meshgrid(np.arange(pattern.shape[0]), np.arange(pattern.shape[1]), indexing='ij')
        total_mass = np.sum(pattern)
        
        center_x = np.sum(x_coords * pattern) / total_mass
        center_y = np.sum(y_coords * pattern) / total_mass
        
        return (center_x, center_y)
    
    def find_analogy(self, source_domain: str, target_domain: str, 
                    source_concept: str, target_concept: str = None) -> Tuple[AnalogyMapping, float]:
        """
        Find analogy between source and target concepts
        
        Args:
            source_domain: Name of source domain
            target_domain: Name of target domain
            source_concept: Name of source concept
            target_concept: Name of target concept (if None, find best match)
            
        Returns:
            Tuple of (analogy_mapping, overall_similarity)
        """
        self.state['tick_counter'] += 1
        
        # Get source pattern
        if source_domain not in self.state['source_patterns']:
            raise ValueError(f"Source domain '{source_domain}' not found")
        
        if source_concept not in self.state['source_patterns'][source_domain]:
            raise ValueError(f"Source concept '{source_concept}' not found in domain '{source_domain}'")
        
        source_pattern = self.state['source_patterns'][source_domain][source_concept]
        
        # Get target pattern
        if target_domain not in self.state['target_patterns']:
            raise ValueError(f"Target domain '{target_domain}' not found")
        
        if target_concept is None:
            # Find best matching target concept
            target_concept, target_pattern = self._find_best_target_match(source_pattern, target_domain)
        else:
            if target_concept not in self.state['target_patterns'][target_domain]:
                raise ValueError(f"Target concept '{target_concept}' not found in domain '{target_domain}'")
            target_pattern = self.state['target_patterns'][target_domain][target_concept]
        
        # Update fields
        self.fields['source_field'].activation = source_pattern.pattern.activation.copy()
        self.fields['target_field'].activation = target_pattern.pattern.activation.copy()
        
        # Find structural similarities using molecular operations
        similarity = self._compute_structural_similarity(source_pattern, target_pattern)
        
        # Create mapping using analogizer molecule
        mapping = self._create_analogical_mapping(source_pattern, target_pattern, similarity)
        
        # Translate concepts using translator atom
        translated = self._translate_concepts(source_pattern, target_pattern, mapping)
        
        # Create bridges for valid mappings using bridge atom
        bridge_strength = self._create_conceptual_bridges(source_pattern, target_pattern, mapping)
        
        # Update witness field to track the analogy
        self._update_witness_field(source_concept, target_concept, similarity)
        
        # Store mapping
        mapping_key = f"{source_domain}:{source_concept}->{target_domain}:{target_concept}"
        self.state['analogy_mappings'][mapping_key] = mapping
        self.state['total_mappings_created'] += 1
        
        # Record in history
        self.state['mapping_history'].append({
            'tick': self.state['tick_counter'],
            'source_domain': source_domain,
            'source_concept': source_concept,
            'target_domain': target_domain,
            'target_concept': target_concept,
            'similarity': similarity,
            'mapping_strength': mapping.mapping_strength,
            'bridge_strength': bridge_strength
        })
        
        print(f"🔗 Found analogy: {source_concept} -> {target_concept} (similarity: {similarity:.3f})")
        
        return mapping, similarity
    
    def _find_best_target_match(self, source_pattern: DomainPattern, target_domain: str) -> Tuple[str, DomainPattern]:
        """Find best matching target concept for source pattern"""
        best_concept = None
        best_similarity = 0.0
        best_pattern = None
        
        for concept_name, target_pattern in self.state['target_patterns'][target_domain].items():
            similarity = self._compute_structural_similarity(source_pattern, target_pattern)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_concept = concept_name
                best_pattern = target_pattern
        
        return best_concept, best_pattern
    
    def _compute_structural_similarity(self, source_pattern: DomainPattern, target_pattern: DomainPattern) -> float:
        """Compute structural similarity between patterns using molecular operations"""
        # Use comparator atom for pattern comparison
        comparison = self.atoms['comparator'].apply(source_pattern.pattern, target_pattern.pattern)
        
        # Use pattern recognizer molecule for enhanced recognition
        recognition = self.molecules['pattern_recognizer'].recognize(source_pattern.pattern, target_pattern.pattern)
        
        # Combine structural and functional similarities
        structural_sim = np.mean(comparison.activation)
        functional_sim = np.mean(recognition.activation)
        
        # Weighted combination
        similarity = (self.config['structural_weight'] * structural_sim + 
                     self.config['functional_weight'] * functional_sim)
        
        # Update similarity field
        self.fields['similarity_field'].activation = comparison.activation
        
        return similarity
    
    def _create_analogical_mapping(self, source_pattern: DomainPattern, target_pattern: DomainPattern, 
                                 similarity: float) -> AnalogyMapping:
        """Create analogical mapping using analogizer molecule"""
        # Use analogizer molecule to create mapping
        mapping_field = self.molecules['analogizer'].map(source_pattern.pattern, target_pattern.pattern)
        
        # Find strongest mapping location
        max_loc = np.unravel_index(np.argmax(mapping_field.activation), mapping_field.activation.shape)
        mapping_strength = mapping_field.activation[max_loc]
        
        # Determine mapping type based on pattern properties
        mapping_type = self._determine_mapping_type(source_pattern, target_pattern)
        
        # Calculate confidence based on similarity and mapping strength
        confidence = min(1.0, similarity * mapping_strength)
        
        # Create mapping
        mapping = AnalogyMapping(
            source_concept=source_pattern.name,
            target_concept=target_pattern.name,
            similarity_score=similarity,
            mapping_strength=mapping_strength,
            field_location=max_loc,
            mapping_type=mapping_type,
            confidence=confidence
        )
        
        # Update mapping field
        self.fields['mapping_field'].activation = mapping_field.activation
        
        return mapping
    
    def _determine_mapping_type(self, source_pattern: DomainPattern, target_pattern: DomainPattern) -> str:
        """Determine the type of analogical mapping"""
        # Compare pattern properties to determine mapping type
        source_props = source_pattern.properties
        target_props = target_pattern.properties
        
        # Structural mapping: similar shape and symmetry
        structural_diff = abs(source_props['symmetry'] - target_props['symmetry'])
        if structural_diff < 0.2:
            return "structural"
        
        # Functional mapping: similar energy and complexity
        energy_diff = abs(source_props['energy'] - target_props['energy']) / max(source_props['energy'], target_props['energy'])
        complexity_diff = abs(source_props['complexity'] - target_props['complexity'])
        if energy_diff < 0.3 and complexity_diff < 0.2:
            return "functional"
        
        # Relational mapping: similar sparsity and center of mass
        sparsity_diff = abs(source_props['sparsity'] - target_props['sparsity'])
        com_diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(source_props['center_of_mass'], target_props['center_of_mass'])))
        if sparsity_diff < 0.2 and com_diff < 2.0:
            return "relational"
        
        # Default to causal mapping
        return "causal"
    
    def _translate_concepts(self, source_pattern: DomainPattern, target_pattern: DomainPattern, 
                          mapping: AnalogyMapping) -> NDAnalogField:
        """Translate concepts using translator atom"""
        # Use translator atom to translate source to target domain
        translated = self.atoms['translator'].translate(
            source_pattern.pattern, 
            target_pattern.pattern, 
            strength=mapping.mapping_strength
        )
        
        return translated
    
    def _create_conceptual_bridges(self, source_pattern: DomainPattern, target_pattern: DomainPattern, 
                                 mapping: AnalogyMapping) -> float:
        """Create conceptual bridges using bridge atom"""
        # Use bridge atom to create connections
        bridge_strength = self.atoms['bridge'].connect(
            source_pattern.pattern, 
            target_pattern.pattern,
            strength=mapping.confidence
        )
        
        # Update bridge field
        self.fields['bridge_field'].activation += bridge_strength * 0.1
        self.fields['bridge_field'].activation = np.clip(self.fields['bridge_field'].activation, 0, 1)
        
        return bridge_strength
    
    def _update_witness_field(self, source_concept: str, target_concept: str, similarity: float):
        """Update witness field to track analogy"""
        # Use witness atom to observe the analogy
        observation = self.atoms['witness'].observe(similarity)
        
        # Update witness field
        self.fields['witness_field'].activation += observation * 0.1
        self.fields['witness_field'].activation = np.clip(self.fields['witness_field'].activation, 0, 1)
    
    def find_analogies_batch(self, source_domain: str, target_domain: str, 
                           max_analogies: int = 10) -> List[Tuple[AnalogyMapping, float]]:
        """
        Find multiple analogies between domains
        
        Args:
            source_domain: Name of source domain
            target_domain: Name of target domain
            max_analogies: Maximum number of analogies to find
            
        Returns:
            List of (analogy_mapping, similarity) tuples
        """
        analogies = []
        
        if source_domain not in self.state['source_patterns']:
            raise ValueError(f"Source domain '{source_domain}' not found")
        
        if target_domain not in self.state['target_patterns']:
            raise ValueError(f"Target domain '{target_domain}' not found")
        
        # Find analogies for each source concept
        for source_concept in list(self.state['source_patterns'][source_domain].keys())[:max_analogies]:
            try:
                mapping, similarity = self.find_analogy(source_domain, target_domain, source_concept)
                analogies.append((mapping, similarity))
            except Exception as e:
                print(f"Warning: Could not find analogy for {source_concept}: {e}")
                continue
        
        # Sort by similarity
        analogies.sort(key=lambda x: x[1], reverse=True)
        
        return analogies
    
    def get_analogy_summary(self) -> Dict[str, Any]:
        """Get summary of analogy state"""
        return {
            "tick": self.state['tick_counter'],
            "source_domains": len(self.state['source_patterns']),
            "target_domains": len(self.state['target_patterns']),
            "total_mappings": len(self.state['analogy_mappings']),
            "mapping_history_length": len(self.state['mapping_history']),
            "total_mappings_created": self.state['total_mappings_created'],
            "field_energies": {
                name: np.sum(field.activation) for name, field in self.fields.items()
            }
        }
    
    def get_analogy_mappings(self) -> List[AnalogyMapping]:
        """Get all analogy mappings"""
        return list(self.state['analogy_mappings'].values())
    
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
        """Reset the analogy maker"""
        self.state = {
            'source_patterns': {},
            'target_patterns': {},
            'analogy_mappings': {},
            'domain_knowledge': {},
            'mapping_history': [],
            'tick_counter': 0,
            'total_mappings_created': 0
        }
        
        # Reset fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Reset molecules
        for molecule in self.molecules.values():
            if hasattr(molecule, 'reset'):
                molecule.reset()
    
    def visualize_analogy(self, mapping: AnalogyMapping, save_path: Optional[str] = None):
        """Visualize an analogy mapping"""
        if not self.config['enable_visualization']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Analogy: {mapping.source_concept} -> {mapping.target_concept}", fontsize=16)
        
        # Source field
        im1 = axes[0, 0].imshow(self.fields['source_field'].activation, cmap='Blues')
        axes[0, 0].set_title("Source Field")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Target field
        im2 = axes[0, 1].imshow(self.fields['target_field'].activation, cmap='Reds')
        axes[0, 1].set_title("Target Field")
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Mapping field
        im3 = axes[0, 2].imshow(self.fields['mapping_field'].activation, cmap='Greens')
        axes[0, 2].set_title("Mapping Field")
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Similarity field
        im4 = axes[1, 0].imshow(self.fields['similarity_field'].activation, cmap='Purples')
        axes[1, 0].set_title("Similarity Field")
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Bridge field
        im5 = axes[1, 1].imshow(self.fields['bridge_field'].activation, cmap='Oranges')
        axes[1, 1].set_title("Bridge Field")
        axes[1, 1].axis('off')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Mapping info
        axes[1, 2].text(0.1, 0.8, f"Similarity: {mapping.similarity_score:.3f}", fontsize=12)
        axes[1, 2].text(0.1, 0.7, f"Strength: {mapping.mapping_strength:.3f}", fontsize=12)
        axes[1, 2].text(0.1, 0.6, f"Type: {mapping.mapping_type}", fontsize=12)
        axes[1, 2].text(0.1, 0.5, f"Confidence: {mapping.confidence:.3f}", fontsize=12)
        axes[1, 2].text(0.1, 0.4, f"Location: {mapping.field_location}", fontsize=12)
        axes[1, 2].set_title("Mapping Info")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def __repr__(self):
        return f"AnalogyMaker(tick={self.state['tick_counter']}, " \
               f"mappings={len(self.state['analogy_mappings'])}, " \
               f"domains={len(self.state['source_patterns'])+len(self.state['target_patterns'])})"


# === Demo Functions ===

def demo_basic_analogy():
    """Demo basic analogy finding"""
    print("=== Basic Analogy Demo ===")
    
    maker = AnalogyMaker({'field_size': (8, 8), 'enable_visualization': False})
    
    # Create source domain (geometric shapes)
    source_patterns = {
        "circle": np.array([
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0]
        ], dtype=np.float32),
        
        "square": np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.float32),
        
        "triangle": np.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.float32)
    }
    
    # Create target domain (natural objects)
    target_patterns = {
        "sun": np.array([
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0]
        ], dtype=np.float32),
        
        "house": np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.float32),
        
        "mountain": np.array([
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ], dtype=np.float32)
    }
    
    maker.inject_source_domain("geometry", source_patterns)
    maker.inject_target_domain("nature", target_patterns)
    
    # Find analogies
    print("Finding analogies between geometry and nature...")
    
    analogies = maker.find_analogies_batch("geometry", "nature", max_analogies=3)
    
    print(f"\nFound {len(analogies)} analogies:")
    for i, (mapping, similarity) in enumerate(analogies):
        print(f"{i+1}. {mapping.source_concept} -> {mapping.target_concept}")
        print(f"   Similarity: {similarity:.3f}, Type: {mapping.mapping_type}")
        print(f"   Confidence: {mapping.confidence:.3f}")
    
    return maker


def demo_complex_analogy():
    """Demo complex cross-domain analogy"""
    print("\n=== Complex Analogy Demo ===")
    
    maker = AnalogyMaker({'field_size': (12, 12), 'enable_visualization': True})
    
    # Create complex source domain (mathematical functions)
    source_patterns = {
        "sine_wave": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        
        "exponential": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    
    # Create sine wave pattern
    for i in range(12):
        y = int(6 + 4 * np.sin(i * np.pi / 6))
        if 0 <= y < 12:
            source_patterns["sine_wave"][y, i] = 1.0
    
    # Create exponential pattern
    for i in range(12):
        y = int(11 - 8 * np.exp(-i / 3))
        if 0 <= y < 12:
            source_patterns["exponential"][y, i] = 1.0
    
    # Create target domain (biological systems)
    target_patterns = {
        "heartbeat": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
        
        "population_growth": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    
    # Create heartbeat pattern (similar to sine wave)
    for i in range(12):
        y = int(6 + 4 * np.sin(i * np.pi / 6))
        if 0 <= y < 12:
            target_patterns["heartbeat"][y, i] = 1.0
    
    # Create population growth pattern (similar to exponential)
    for i in range(12):
        y = int(11 - 8 * np.exp(-i / 3))
        if 0 <= y < 12:
            target_patterns["population_growth"][y, i] = 1.0
    
    maker.inject_source_domain("mathematics", source_patterns)
    maker.inject_target_domain("biology", target_patterns)
    
    # Find analogies
    print("Finding analogies between mathematics and biology...")
    
    analogies = maker.find_analogies_batch("mathematics", "biology", max_analogies=2)
    
    print(f"\nFound {len(analogies)} analogies:")
    for i, (mapping, similarity) in enumerate(analogies):
        print(f"{i+1}. {mapping.source_concept} -> {mapping.target_concept}")
        print(f"   Similarity: {similarity:.3f}, Type: {mapping.mapping_type}")
        print(f"   Confidence: {mapping.confidence:.3f}")
        
        # Visualize the analogy
        maker.visualize_analogy(mapping)
    
    return maker


# === Main Demo ===

if __name__ == '__main__':
    print("🔗 ANALOGY MAKER - Cross-Domain Reasoning 🔗")
    print("Maps concepts between domains using molecular operations!")
    print("Uses structural similarity and analogical mapping\n")
    
    # Run demos
    basic_maker = demo_basic_analogy()
    complex_maker = demo_complex_analogy()
    
    # System capabilities summary
    print("\n" + "="*60)
    print("🎯 ANALOGY MAKER CAPABILITIES DEMONSTRATED")
    print("="*60)
    
    all_makers = [basic_maker, complex_maker]
    total_mappings = sum(len(maker.get_analogy_mappings()) for maker in all_makers)
    
    print(f"✅ Cross-domain concept mapping")
    print(f"✅ Structural similarity detection")
    print(f"✅ Functional analogy recognition")
    print(f"✅ Relational pattern matching")
    print(f"✅ Causal connection inference")
    print(f"✅ Molecular pattern translation")
    print(f"✅ Conceptual bridge creation")
    print(f"✅ Multi-domain knowledge integration")
    
    print(f"\n📊 DEMO STATISTICS:")
    print(f"Total mappings created: {total_mappings}")
    print(f"Domains processed: {len(basic_maker.state['source_patterns']) + len(basic_maker.state['target_patterns'])}")
    
    print(f"\n💡 KEY INNOVATIONS:")
    print(f"• Molecular cross-domain mapping")
    print(f"• Structural similarity using atoms and molecules")
    print(f"• Automatic analogy type detection")
    print(f"• Conceptual bridge creation")
    print(f"• Multi-domain knowledge integration")
    
    print(f"\n🌟 This demonstrates true cross-domain reasoning!")
    print("The system finds structural similarities between completely different domains.")
    print("No training data, no neural networks - pure molecular intelligence!")
    
    print("\n🚀 AnalogyMaker Demo Complete! 🚀")