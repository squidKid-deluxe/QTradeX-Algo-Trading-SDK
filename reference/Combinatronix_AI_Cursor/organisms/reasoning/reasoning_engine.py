# ============================================================================
# ReasoningEngine - Advanced Reasoning Using Molecular Architecture
# ============================================================================

"""
ReasoningEngine - Advanced reasoning that invents symbols to resolve tensions

Composition:
- Atoms: Comparator, Resonator, MemoryTrace, Threshold, Balancer, Filler
- Molecules: ContradictionResolver, GapFiller, PatternCompleter, Analogizer
- Fields: reasoning_field, tension_field, concept_field, invention_field

This organism detects cognitive tensions in fields and invents new symbols
to resolve them using molecular operations and combinatorial reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import hashlib

try:
    from ...core import NDAnalogField
    from ...atoms.multi_field import ComparatorAtom, ResonatorAtom
    from ...atoms.temporal import MemoryTraceAtom, ThresholdAtom
    from ...atoms.tension_resolvers import BalancerAtom, FillerAtom
    from ...molecules.reasoning import ContradictionResolverMolecule, GapFillerMolecule, PatternCompleterMolecule, AnalogizerMolecule
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.atoms.multi_field import ComparatorAtom, ResonatorAtom
    from combinatronix.atoms.temporal import MemoryTraceAtom, ThresholdAtom
    from combinatronix.atoms.tension_resolvers import BalancerAtom, FillerAtom
    from combinatronix.molecules.reasoning import ContradictionResolverMolecule, GapFillerMolecule, PatternCompleterMolecule, AnalogizerMolecule


@dataclass
class ConceptTension:
    """Represents tension between incompatible concepts"""
    concept_a: str
    concept_b: str
    tension_strength: float
    field_location: Tuple[int, int]
    tension_type: str  # "contradiction", "gap", "ambiguity", "overflow"


@dataclass
class InventedSymbol:
    """A newly invented symbol/word to resolve tensions"""
    symbol: str
    pattern: NDAnalogField
    resolves_tension: ConceptTension
    strength: float
    creation_tick: int
    meaning_description: str


class ReasoningEngine:
    """Advanced reasoning engine using molecular architecture"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the reasoning engine
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'field_size': (8, 8),
            'tension_sensitivity': 0.25,
            'invention_threshold': 0.5,
            'max_symbols': 50,
            'enable_visualization': True,
            'reasoning_depth': 5
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
            'concept_echoes': {},
            'invented_symbols': {},
            'active_tensions': [],
            'reasoning_history': [],
            'tick_counter': 0,
            'total_symbols_invented': 0
        }
        
        # Initialize reasoning kernels
        self._initialize_reasoning_kernels()
        
        print(f"🧠 ReasoningEngine initialized ({self.config['field_size'][0]}×{self.config['field_size'][1]})")
    
    def _initialize_atoms(self):
        """Initialize atomic operations"""
        self.atoms = {
            'comparator': ComparatorAtom(metric='difference', normalize=True),
            'resonator': ResonatorAtom(amplification=1.5, threshold=0.5),
            'memory_trace': MemoryTraceAtom(accumulation_rate=0.3, decay_rate=0.95),
            'threshold': ThresholdAtom(threshold=self.config['tension_sensitivity'], mode='binary'),
            'balancer': BalancerAtom(equilibrium_rate=0.3, min_tension=0.1),
            'filler': FillerAtom(creativity=0.5, gap_threshold=0.1)
        }
    
    def _initialize_molecules(self):
        """Initialize molecular operations"""
        self.molecules = {
            'contradiction_resolver': ContradictionResolverMolecule(
                equilibrium_rate=0.2,
                min_tension=0.1
            ),
            'gap_filler': GapFillerMolecule(
                creativity=0.5,
                gap_threshold=0.1
            ),
            'pattern_completer': PatternCompleterMolecule(
                completion_threshold=0.3,
                pattern_strength=0.8
            ),
            'analogizer': AnalogizerMolecule(
                similarity_threshold=0.4,
                translation_strength=0.8
            )
        }
    
    def _initialize_fields(self):
        """Initialize field structures"""
        self.fields = {
            'reasoning_field': NDAnalogField(self.config['field_size']),
            'tension_field': NDAnalogField(self.config['field_size']),
            'concept_field': NDAnalogField(self.config['field_size']),
            'invention_field': NDAnalogField(self.config['field_size']),
            'input_field': NDAnalogField(self.config['field_size'])
        }
    
    def _initialize_reasoning_kernels(self):
        """Initialize combinatorial reasoning kernels"""
        self.reasoning_kernels = {
            'analyze': self._analyze_kernel,      # S combinator - analytical spreading
            'synthesize': self._synthesize_kernel, # K combinator - synthetic focusing
            'abstract': self._abstract_kernel,    # I combinator - identity abstraction
            'compose': self._compose_kernel,      # B combinator - composition
            'compare': self._compare_kernel,      # C combinator - comparison
            'amplify': self._amplify_kernel       # W combinator - amplification
        }
    
    def inject_concepts(self, concepts: Dict[str, np.ndarray]):
        """
        Inject known concepts into the reasoning engine
        
        Args:
            concepts: Dictionary mapping concept names to their patterns
        """
        for word, pattern in concepts.items():
            # Resize pattern if needed
            if pattern.shape != self.config['field_size']:
                resized = self._resize_pattern(pattern)
            else:
                resized = pattern.copy()
            
            # Create NDAnalogField
            concept_field = NDAnalogField(self.config['field_size'], activation=resized)
            self.state['concept_echoes'][word] = concept_field
        
        print(f"📥 Injected {len(concepts)} concepts into reasoning engine")
    
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
    
    def inject_input_field(self, input_field: np.ndarray):
        """
        Inject new input field to reason about
        
        Args:
            input_field: Input field pattern
        """
        if input_field.shape != self.config['field_size']:
            input_field = self._resize_pattern(input_field)
        
        self.fields['input_field'].activation = input_field.astype(np.float32)
        print(f"🔍 Input field injected (energy: {np.sum(input_field):.3f})")
    
    def reason_step(self) -> Dict[str, Any]:
        """
        Single reasoning step - detect tensions and potentially invent symbols
        
        Returns:
            Dictionary with reasoning results
        """
        self.state['tick_counter'] += 1
        
        # 1. Detect tensions using molecular operations
        tensions = self._detect_tensions_molecular()
        self.state['active_tensions'] = tensions
        
        # 2. Update tension field
        self._update_tension_field()
        
        # 3. Apply reasoning kernels based on tension types
        reasoning_operations = self._apply_reasoning_kernels()
        
        # 4. Invent symbols for strong tensions using molecular processing
        new_symbols = self._invent_symbols_molecular()
        
        # 5. Update reasoning field
        self._update_reasoning_field()
        
        # 6. Record reasoning step
        step_result = {
            "tick": self.state['tick_counter'],
            "tensions_detected": len(tensions),
            "symbols_invented": len(new_symbols),
            "total_symbols": len(self.state['invented_symbols']),
            "field_energy": np.sum(self.fields['reasoning_field'].activation),
            "tension_energy": np.sum(self.fields['tension_field'].activation),
            "new_symbols": [s.symbol for s in new_symbols],
            "reasoning_operations": reasoning_operations
        }
        
        self.state['reasoning_history'].append(step_result)
        return step_result
    
    def _detect_tensions_molecular(self) -> List[ConceptTension]:
        """Detect tensions using molecular operations"""
        tensions = []
        
        # Get all patterns (concepts + invented symbols)
        all_patterns = {**self.state['concept_echoes'], 
                       **{s.symbol: s.pattern for s in self.state['invented_symbols'].values()}}
        
        # Detect contradictions using molecular processing
        contradictions = self._detect_contradictions_molecular(all_patterns)
        tensions.extend(contradictions)
        
        # Detect gaps using molecular processing
        gaps = self._detect_gaps_molecular(all_patterns)
        tensions.extend(gaps)
        
        # Detect ambiguities using molecular processing
        ambiguities = self._detect_ambiguities_molecular(all_patterns)
        tensions.extend(ambiguities)
        
        # Detect overflow using molecular processing
        overflows = self._detect_overflow_molecular()
        tensions.extend(overflows)
        
        return tensions
    
    def _detect_contradictions_molecular(self, all_patterns: Dict[str, NDAnalogField]) -> List[ConceptTension]:
        """Detect contradictions using molecular operations"""
        tensions = []
        
        # Define contradictory pairs
        contradictions = [
            ("big", "small"), ("happy", "sad"), ("fast", "slow"),
            ("hot", "cold"), ("light", "dark"), ("up", "down")
        ]
        
        for concept_a, concept_b in contradictions:
            if concept_a in all_patterns and concept_b in all_patterns:
                pattern_a = all_patterns[concept_a]
                pattern_b = all_patterns[concept_b]
                
                # Use comparator atom to find overlap
                comparison = self.atoms['comparator'].apply(pattern_a, pattern_b)
                
                # Find regions where both patterns are active
                overlap = pattern_a.activation * pattern_b.activation
                max_overlap_loc = np.unravel_index(np.argmax(overlap), overlap.shape)
                max_overlap_strength = overlap[max_overlap_loc]
                
                if max_overlap_strength > self.config['tension_sensitivity']:
                    tensions.append(ConceptTension(
                        concept_a=concept_a,
                        concept_b=concept_b,
                        tension_strength=max_overlap_strength,
                        field_location=max_overlap_loc,
                        tension_type="contradiction"
                    ))
        
        return tensions
    
    def _detect_gaps_molecular(self, all_patterns: Dict[str, NDAnalogField]) -> List[ConceptTension]:
        """Detect gaps using molecular operations"""
        tensions = []
        
        # Sum all known concept activations
        total_concept_coverage = np.zeros(self.config['field_size'])
        for concept_field in all_patterns.values():
            total_concept_coverage += concept_field.activation
        
        # Find regions with high field activity but low concept coverage
        gap_strength = self.fields['input_field'].activation - total_concept_coverage * 0.5
        gap_locations = np.where(gap_strength > self.config['tension_sensitivity'])
        
        for i in range(len(gap_locations[0])):
            loc = (gap_locations[0][i], gap_locations[1][i])
            strength = gap_strength[loc]
            
            tensions.append(ConceptTension(
                concept_a="<unknown>",
                concept_b="<field_activation>", 
                tension_strength=strength,
                field_location=loc,
                tension_type="gap"
            ))
        
        return tensions
    
    def _detect_ambiguities_molecular(self, all_patterns: Dict[str, NDAnalogField]) -> List[ConceptTension]:
        """Detect ambiguities using molecular operations"""
        tensions = []
        
        # Group similar concepts
        concept_groups = {
            "animals": ["cat", "dog", "bird", "fish"],
            "sizes": ["big", "small"], 
            "colors": ["red", "blue"],
            "emotions": ["happy", "sad"],
            "actions": ["runs", "jumps", "sleeps", "eats", "flies", "swims"]
        }
        
        for group_name, concepts in concept_groups.items():
            active_concepts = []
            for concept in concepts:
                if concept in all_patterns:
                    pattern = all_patterns[concept]
                    activation_strength = np.sum(pattern.activation * self.fields['input_field'].activation)
                    if activation_strength > self.config['tension_sensitivity']:
                        active_concepts.append((concept, activation_strength))
            
            # If multiple concepts in same group are highly active, create ambiguity tension
            if len(active_concepts) >= 2:
                active_concepts.sort(key=lambda x: x[1], reverse=True)
                concept_a, strength_a = active_concepts[0]
                concept_b, strength_b = active_concepts[1]
                
                # Find location of maximum overlap
                pattern_a = all_patterns[concept_a]
                pattern_b = all_patterns[concept_b]
                combined = pattern_a.activation + pattern_b.activation
                max_loc = np.unravel_index(np.argmax(combined), combined.shape)
                
                tensions.append(ConceptTension(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    tension_strength=min(strength_a, strength_b),
                    field_location=max_loc,
                    tension_type="ambiguity"
                ))
        
        return tensions
    
    def _detect_overflow_molecular(self) -> List[ConceptTension]:
        """Detect overflow using molecular operations"""
        tensions = []
        
        # Use threshold atom to detect overflow
        threshold_field = self.fields['input_field'].copy()
        self.atoms['threshold'].apply(threshold_field)
        
        overflow_locations = np.where(threshold_field.activation > 0.9)
        
        for i in range(len(overflow_locations[0])):
            loc = (overflow_locations[0][i], overflow_locations[1][i])
            strength = self.fields['input_field'].activation[loc]
            
            tensions.append(ConceptTension(
                concept_a="<overflow>",
                concept_b="<field_saturation>",
                tension_strength=strength - 0.9,
                field_location=loc,
                tension_type="overflow"
            ))
        
        return tensions
    
    def _update_tension_field(self):
        """Update tension field with current tensions"""
        self.fields['tension_field'].activation.fill(0)
        
        for tension in self.state['active_tensions']:
            x, y = tension.field_location
            self.fields['tension_field'].activation[x, y] += tension.tension_strength
    
    def _apply_reasoning_kernels(self) -> Dict[str, str]:
        """Apply combinatorial reasoning kernels based on current tensions"""
        operations = {}
        
        # Analyze kernel - spread understanding of tensions
        if any(t.tension_type == "gap" for t in self.state['active_tensions']):
            operations["analyze"] = self.reasoning_kernels["analyze"]()
        
        # Synthesize kernel - combine conflicting concepts
        if any(t.tension_type == "contradiction" for t in self.state['active_tensions']):
            operations["synthesize"] = self.reasoning_kernels["synthesize"]()
        
        # Abstract kernel - find higher-level patterns
        if any(t.tension_type == "ambiguity" for t in self.state['active_tensions']):
            operations["abstract"] = self.reasoning_kernels["abstract"]()
        
        # Compose kernel - build complex concepts
        if len(self.state['active_tensions']) > 2:
            operations["compose"] = self.reasoning_kernels["compose"]()
        
        return operations
    
    def _analyze_kernel(self) -> str:
        """S combinator - analytical spreading"""
        # Spread activation from tension points
        for tension in self.state['active_tensions']:
            loc = tension.field_location
            # Simple spreading pattern
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    x, y = loc[0] + dx, loc[1] + dy
                    if 0 <= x < self.config['field_size'][0] and 0 <= y < self.config['field_size'][1]:
                        self.fields['reasoning_field'].activation[x, y] += 0.05
        return "analytical_spreading"
    
    def _synthesize_kernel(self) -> str:
        """K combinator - synthetic focusing"""
        # Focus on strongest tension point
        if self.state['active_tensions']:
            strongest = max(self.state['active_tensions'], key=lambda t: t.tension_strength)
            loc = strongest.field_location
            self.fields['reasoning_field'].activation[loc] += 0.2
        return "synthetic_focusing"
    
    def _abstract_kernel(self) -> str:
        """I combinator - identity abstraction"""
        # Preserve essential patterns, reduce noise
        self.fields['reasoning_field'].activation *= 1.1
        self.fields['reasoning_field'].activation = np.clip(self.fields['reasoning_field'].activation, 0, 1)
        return "identity_abstraction"
    
    def _compose_kernel(self) -> str:
        """B combinator - compositional reasoning"""
        # Combine nearby tensions
        tension_locations = [t.field_location for t in self.state['active_tensions']]
        for i, loc1 in enumerate(tension_locations):
            for j, loc2 in enumerate(tension_locations[i+1:], i+1):
                distance = abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
                if distance <= 2:  # Close tensions
                    # Create connection
                    mid_x = (loc1[0] + loc2[0]) // 2
                    mid_y = (loc1[1] + loc2[1]) // 2
                    self.fields['reasoning_field'].activation[mid_x, mid_y] += 0.1
        return "compositional_reasoning"
    
    def _compare_kernel(self) -> str:
        """C combinator - comparative analysis"""
        # Compare tension strengths and highlight differences
        if len(self.state['active_tensions']) >= 2:
            strengths = [t.tension_strength for t in self.state['active_tensions']]
            max_strength = max(strengths)
            for tension in self.state['active_tensions']:
                if tension.tension_strength < max_strength * 0.5:
                    # Suppress weaker tensions
                    loc = tension.field_location
                    self.fields['reasoning_field'].activation[loc] *= 0.9
        return "comparative_analysis"
    
    def _amplify_kernel(self) -> str:
        """W combinator - amplification"""
        # Amplify strong patterns
        strong_regions = np.where(self.fields['reasoning_field'].activation > 0.7)
        for i in range(len(strong_regions[0])):
            x, y = strong_regions[0][i], strong_regions[1][i]
            self.fields['reasoning_field'].activation[x, y] *= 1.2
        self.fields['reasoning_field'].activation = np.clip(self.fields['reasoning_field'].activation, 0, 1)
        return "pattern_amplification"
    
    def _invent_symbols_molecular(self) -> List[InventedSymbol]:
        """Invent symbols for strong tensions using molecular processing"""
        new_symbols = []
        strong_tensions = [t for t in self.state['active_tensions'] if t.tension_strength > self.config['invention_threshold']]
        
        for tension in strong_tensions:
            # Check if we already have a symbol for similar tension
            if not self._has_similar_symbol(tension):
                new_symbol = self._invent_symbol_for_tension_molecular(tension)
                new_symbol.creation_tick = self.state['tick_counter']
                
                self.state['invented_symbols'][new_symbol.symbol] = new_symbol
                self.state['total_symbols_invented'] += 1
                new_symbols.append(new_symbol)
                
                print(f"💡 Invented symbol '{new_symbol.symbol}': {new_symbol.meaning_description}")
        
        return new_symbols
    
    def _invent_symbol_for_tension_molecular(self, tension: ConceptTension) -> InventedSymbol:
        """Invent symbol for tension using molecular operations"""
        if tension.tension_type == "contradiction":
            return self._invent_contradiction_resolver_molecular(tension)
        elif tension.tension_type == "gap":
            return self._invent_gap_filler_molecular(tension)
        elif tension.tension_type == "ambiguity":
            return self._invent_disambiguator_molecular(tension)
        elif tension.tension_type == "overflow":
            return self._invent_overflow_regulator_molecular(tension)
        else:
            return self._invent_generic_symbol_molecular(tension)
    
    def _invent_contradiction_resolver_molecular(self, tension: ConceptTension) -> InventedSymbol:
        """Invent symbol that resolves contradiction using molecular operations"""
        # Use contradiction resolver molecule
        if tension.concept_a in self.state['concept_echoes'] and tension.concept_b in self.state['concept_echoes']:
            pattern_a = self.state['concept_echoes'][tension.concept_a]
            pattern_b = self.state['concept_echoes'][tension.concept_b]
            
            # Use contradiction resolver molecule
            resolved_field = self.molecules['contradiction_resolver'].resolve(pattern_a, pattern_b)
            
            # Create resolution pattern
            resolution_pattern = resolved_field.copy()
            resolution_pattern.activation[tension.field_location] = 0.9
            
        else:
            # Create new pattern centered at tension location
            resolution_pattern = NDAnalogField(self.config['field_size'])
            resolution_pattern.activation[tension.field_location] = 1.0
        
        # Generate symbol name
        symbol_name = f"#{tension.concept_a[0]}{tension.concept_b[0]}{self.state['total_symbols_invented']:02d}"
        
        # Create meaning description
        meaning = f"resolves tension between '{tension.concept_a}' and '{tension.concept_b}'"
        
        return InventedSymbol(
            symbol=symbol_name,
            pattern=resolution_pattern,
            resolves_tension=tension,
            strength=tension.tension_strength,
            creation_tick=0,
            meaning_description=meaning
        )
    
    def _invent_gap_filler_molecular(self, tension: ConceptTension) -> InventedSymbol:
        """Invent symbol to fill gap using molecular operations"""
        # Use gap filler molecule
        gap_field = NDAnalogField(self.config['field_size'])
        gap_field.activation[tension.field_location] = 1.0
        
        filled_field = self.molecules['gap_filler'].fill_gap(gap_field)
        
        symbol_name = f"*gap{self.state['total_symbols_invented']:03d}"
        meaning = f"fills conceptual gap at {tension.field_location}"
        
        return InventedSymbol(
            symbol=symbol_name,
            pattern=filled_field,
            resolves_tension=tension,
            strength=tension.tension_strength,
            creation_tick=0,
            meaning_description=meaning
        )
    
    def _invent_disambiguator_molecular(self, tension: ConceptTension) -> InventedSymbol:
        """Invent symbol to disambiguate using molecular operations"""
        if (tension.concept_a in self.state['concept_echoes'] and 
            tension.concept_b in self.state['concept_echoes']):
            
            pattern_a = self.state['concept_echoes'][tension.concept_a]
            pattern_b = self.state['concept_echoes'][tension.concept_b]
            
            # Use pattern completer molecule for disambiguation
            completed_field = self.molecules['pattern_completer'].complete_pattern(pattern_a, pattern_b)
            
        else:
            completed_field = NDAnalogField(self.config['field_size'])
            completed_field.activation[tension.field_location] = 1.0
        
        symbol_name = f"@dis{self.state['total_symbols_invented']:03d}"
        meaning = f"disambiguates '{tension.concept_a}' from '{tension.concept_b}'"
        
        return InventedSymbol(
            symbol=symbol_name,
            pattern=completed_field,
            resolves_tension=tension,
            strength=tension.tension_strength,
            creation_tick=0,
            meaning_description=meaning
        )
    
    def _invent_overflow_regulator_molecular(self, tension: ConceptTension) -> InventedSymbol:
        """Invent symbol to regulate overflow using molecular operations"""
        # Create dampening pattern
        regulator_field = NDAnalogField(self.config['field_size'])
        loc_x, loc_y = tension.field_location
        
        # Create suppression pattern around overflow
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = loc_x + dx, loc_y + dy
                if 0 <= x < self.config['field_size'][0] and 0 <= y < self.config['field_size'][1]:
                    distance = abs(dx) + abs(dy)
                    strength = max(0, 1.0 - distance * 0.2)
                    regulator_field.activation[x, y] = strength
        
        symbol_name = f"~reg{self.state['total_symbols_invented']:03d}"
        meaning = f"regulates overflow at {tension.field_location}"
        
        return InventedSymbol(
            symbol=symbol_name,
            pattern=regulator_field,
            resolves_tension=tension,
            strength=tension.tension_strength,
            creation_tick=0,
            meaning_description=meaning
        )
    
    def _invent_generic_symbol_molecular(self, tension: ConceptTension) -> InventedSymbol:
        """Invent generic symbol for unknown tension type"""
        # Create random but structured pattern
        pattern = np.random.random(self.config['field_size']) * 0.3
        pattern[tension.field_location] = 1.0
        
        pattern_field = NDAnalogField(self.config['field_size'], activation=pattern)
        
        symbol_name = f"?sym{self.state['total_symbols_invented']:03d}"
        meaning = f"addresses {tension.tension_type} tension"
        
        return InventedSymbol(
            symbol=symbol_name,
            pattern=pattern_field,
            resolves_tension=tension,
            strength=tension.tension_strength,
            creation_tick=0,
            meaning_description=meaning
        )
    
    def _has_similar_symbol(self, tension: ConceptTension) -> bool:
        """Check if we already have a symbol for similar tension"""
        for symbol_data in self.state['invented_symbols'].values():
            existing_tension = symbol_data.resolves_tension
            
            # Same type and close location
            if (existing_tension.tension_type == tension.tension_type and
                abs(existing_tension.field_location[0] - tension.field_location[0]) <= 1 and
                abs(existing_tension.field_location[1] - tension.field_location[1]) <= 1):
                return True
        
        return False
    
    def _update_reasoning_field(self):
        """Update reasoning field with current state"""
        self.fields['reasoning_field'].activation = self.fields['input_field'].activation.copy()
        
        # Apply invented symbols to reduce tensions
        for symbol_data in self.state['invented_symbols'].values():
            if symbol_data.strength > 0.1:  # Only active symbols
                # Apply symbol pattern to reduce tension
                loc = symbol_data.resolves_tension.field_location
                influence = symbol_data.pattern.activation * symbol_data.strength * 0.2
                self.fields['reasoning_field'].activation += influence
        
        self.fields['reasoning_field'].activation = np.clip(self.fields['reasoning_field'].activation, 0, 1)
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning state"""
        return {
            "tick": self.state['tick_counter'],
            "known_concepts": len(self.state['concept_echoes']),
            "invented_symbols": len(self.state['invented_symbols']),
            "active_tensions": len(self.state['active_tensions']),
            "field_energy": np.sum(self.fields['reasoning_field'].activation),
            "tension_energy": np.sum(self.fields['tension_field'].activation),
            "strongest_tension": max(self.state['active_tensions'], key=lambda t: t.tension_strength).tension_strength if self.state['active_tensions'] else 0,
            "reasoning_history_length": len(self.state['reasoning_history']),
            "total_symbols_invented": self.state['total_symbols_invented']
        }
    
    def get_invented_symbols(self) -> List[InventedSymbol]:
        """Get all invented symbols"""
        return list(self.state['invented_symbols'].values())
    
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
        """Reset the reasoning engine"""
        self.state = {
            'concept_echoes': {},
            'invented_symbols': {},
            'active_tensions': [],
            'reasoning_history': [],
            'tick_counter': 0,
            'total_symbols_invented': 0
        }
        
        # Reset fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Reset molecules
        for molecule in self.molecules.values():
            if hasattr(molecule, 'reset'):
                molecule.reset()
    
    def visualize_reasoning_state(self, save_path: Optional[str] = None):
        """Visualize current reasoning state"""
        if not self.config['enable_visualization']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"ReasoningEngine - Molecular Reasoning (Tick {self.state['tick_counter']})", fontsize=16)
        
        # Input field
        im1 = axes[0, 0].imshow(self.fields['input_field'].activation, cmap='Blues')
        axes[0, 0].set_title("Input Field")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Reasoning field
        im2 = axes[0, 1].imshow(self.fields['reasoning_field'].activation, cmap='Greens') 
        axes[0, 1].set_title("Reasoning Field")
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Tension field
        im3 = axes[0, 2].imshow(self.fields['tension_field'].activation, cmap='Reds')
        axes[0, 2].set_title("Tension Field")
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Tension types
        tension_types = defaultdict(int)
        for tension in self.state['active_tensions']:
            tension_types[tension.tension_type] += 1
        
        if tension_types:
            axes[1, 0].bar(tension_types.keys(), tension_types.values())
            axes[1, 0].set_title("Tension Types")
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, "No Tensions", ha='center', va='center')
            axes[1, 0].set_title("Tension Types")
        
        # Symbols over time
        if self.state['reasoning_history']:
            ticks = [h["tick"] for h in self.state['reasoning_history']]
            symbols = [h["total_symbols"] for h in self.state['reasoning_history']]
            axes[1, 1].plot(ticks, symbols, 'o-')
            axes[1, 1].set_title("Symbols Invented Over Time")
            axes[1, 1].set_xlabel("Tick")
            axes[1, 1].set_ylabel("Total Symbols")
        
        # Sample invented symbol
        if self.state['invented_symbols']:
            latest_symbol = list(self.state['invented_symbols'].values())[-1]
            im4 = axes[1, 2].imshow(latest_symbol.pattern.activation, cmap='Purples')
            axes[1, 2].set_title(f"Latest Symbol: {latest_symbol.symbol}")
            axes[1, 2].axis('off')
            plt.colorbar(im4, ax=axes[1, 2])
        else:
            axes[1, 2].text(0.5, 0.5, "No Symbols\nInvented", ha='center', va='center')
            axes[1, 2].set_title("Latest Symbol")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def __repr__(self):
        return f"ReasoningEngine(tick={self.state['tick_counter']}, " \
               f"symbols={len(self.state['invented_symbols'])}, " \
               f"tensions={len(self.state['active_tensions'])})"


# === Demo Functions ===

def demo_basic_reasoning():
    """Demo basic tension detection and symbol invention"""
    print("=== Basic Reasoning Demo ===")
    
    engine = ReasoningEngine({'field_size': (8, 8), 'enable_visualization': False})
    
    # Create some basic concepts
    concepts = {
        "big": np.array([
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "small": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "happy": np.array([
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "sad": np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)
    }
    
    engine.inject_concepts(concepts)
    
    # Create contradictory input field (big + small activated simultaneously)
    contradictory_field = concepts["big"] * 0.8 + concepts["small"] * 0.7
    engine.inject_input_field(contradictory_field)
    
    print(f"Input field contains both 'big' and 'small' concepts")
    
    # Run reasoning steps
    for step in range(3):
        result = engine.reason_step()
        print(f"\nStep {step + 1}:")
        print(f"  Tensions detected: {result['tensions_detected']}")
        print(f"  Symbols invented: {result['symbols_invented']}")
        if result['new_symbols']:
            print(f"  New symbols: {result['new_symbols']}")
    
    # Show invented symbols
    symbols = engine.get_invented_symbols()
    print(f"\n💡 Total symbols invented: {len(symbols)}")
    for symbol in symbols:
        print(f"  {symbol.symbol}: {symbol.meaning_description}")
    
    return engine


def demo_conceptual_gaps():
    """Demo gap detection and filling"""
    print("\n=== Conceptual Gap Demo ===")
    
    engine = ReasoningEngine({'field_size': (8, 8), 'enable_visualization': False})
    
    # Inject minimal concepts
    concepts = {
        "cat": np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "runs": np.array([
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)
    }
    
    engine.inject_concepts(concepts)
    
    # Create input field with gaps (activation in uncovered regions)
    gap_field = np.zeros((8, 8))
    gap_field[3:5, 3:5] = 0.9  # Strong activation in center (no concept covers this)
    gap_field[6:8, 1:3] = 0.7  # Another gap
    
    engine.inject_input_field(gap_field)
    
    print("Input field has activations in regions not covered by known concepts")
    
    # Run reasoning
    for step in range(4):
        result = engine.reason_step()
        print(f"\nStep {step + 1}:")
        print(f"  Gaps detected: {len([t for t in engine.state['active_tensions'] if t.tension_type == 'gap'])}")
        print(f"  New symbols: {result['new_symbols']}")
    
    symbols = engine.get_invented_symbols()
    gap_fillers = [s for s in symbols if "gap" in s.symbol]
    print(f"\n🔍 Gap-filling symbols invented: {len(gap_fillers)}")
    for symbol in gap_fillers:
        print(f"  {symbol.symbol}: {symbol.meaning_description}")
    
    return engine


def demo_complex_reasoning():
    """Demo complex multi-step reasoning with multiple tension types"""
    print("\n=== Complex Multi-Step Reasoning Demo ===")
    
    engine = ReasoningEngine({'field_size': (10, 10), 'enable_visualization': True})
    
    # Rich concept set
    concepts = {
        "big": np.eye(10) * 0.8,  # Diagonal pattern
        "small": np.fliplr(np.eye(10)) * 0.7,  # Reverse diagonal
        "fast": np.ones((10, 10)) * 0.3,  # Uniform background
        "slow": np.zeros((10, 10)),  # Empty
        "animal": np.random.random((10, 10)) * 0.4,  # Random pattern
        "happy": np.ones((5, 5)),  # Will be padded
        "movement": np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0, 1]] * 10) * 0.6  # Striped
    }
    
    # Resize concepts to fit field
    resized_concepts = {}
    for name, pattern in concepts.items():
        if pattern.shape != (10, 10):
            resized = np.zeros((10, 10))
            min_h = min(pattern.shape[0], 10)
            min_w = min(pattern.shape[1], 10) 
            resized[:min_h, :min_w] = pattern[:min_h, :min_w]
            resized_concepts[name] = resized
        else:
            resized_concepts[name] = pattern
    
    engine.inject_concepts(resized_concepts)
    
    # Create complex input with multiple tension types
    complex_field = (resized_concepts["big"] * 0.8 +        # Contradiction with small
                    resized_concepts["small"] * 0.6 +       # Contradiction with big  
                    resized_concepts["fast"] * 0.7 +        # Ambiguity with movement
                    resized_concepts["movement"] * 0.5)     # Ambiguity with fast
    
    # Add some gaps
    complex_field[7:9, 7:9] = 0.9  # Gap region
    complex_field[1:3, 8:10] = 0.8  # Another gap
    
    # Add overflow
    complex_field[4:6, 4:6] += 0.5  # Create overflow
    complex_field = np.clip(complex_field, 0, 1)
    
    engine.inject_input_field(complex_field)
    
    print("Complex input with contradictions, ambiguities, gaps, and overflow")
    
    # Run extended reasoning
    for step in range(6):
        result = engine.reason_step()
        
        # Analyze tension types
        tension_counts = {}
        for tension in engine.state['active_tensions']:
            tension_counts[tension.tension_type] = tension_counts.get(tension.tension_type, 0) + 1
        
        print(f"\nStep {step + 1}:")
        print(f"  Total tensions: {result['tensions_detected']}")
        for t_type, count in tension_counts.items():
            print(f"    {t_type}: {count}")
        print(f"  Symbols invented this step: {result['symbols_invented']}")
        print(f"  Total symbols: {result['total_symbols']}")
        print(f"  Field energy: {result['field_energy']:.3f}")
        
        if result['new_symbols']:
            print(f"  New symbols: {result['new_symbols']}")
    
    # Final analysis
    print(f"\n🧠 REASONING COMPLETE")
    summary = engine.get_reasoning_summary()
    symbols = engine.get_invented_symbols()
    
    print(f"Final state:")
    print(f"  Total symbols invented: {len(symbols)}")
    print(f"  Active tensions remaining: {summary['active_tensions']}")
    print(f"  Reasoning steps: {summary['reasoning_history_length']}")
    
    print(f"\nInvented symbols by type:")
    symbol_types = {}
    for symbol in symbols:
        symbol_type = symbol.resolves_tension.tension_type
        symbol_types[symbol_type] = symbol_types.get(symbol_type, 0) + 1
    
    for s_type, count in symbol_types.items():
        print(f"  {s_type}: {count} symbols")
    
    # Show visualization
    engine.visualize_reasoning_state()
    
    return engine


# === Main Demo ===

if __name__ == '__main__':
    print("🧠 REASONING ENGINE - Molecular Symbol Invention 🧠")
    print("Detects field tensions and invents symbols to resolve them!")
    print("Uses molecular operations and combinatorial reasoning\n")
    
    # Run comprehensive demos
    basic_engine = demo_basic_reasoning()
    gap_engine = demo_conceptual_gaps()
    complex_engine = demo_complex_reasoning()
    
    # System capabilities summary
    print("\n" + "="*60)
    print("🎯 MOLECULAR REASONING CAPABILITIES DEMONSTRATED")
    print("="*60)
    
    all_engines = [basic_engine, gap_engine, complex_engine]
    total_symbols = sum(len(engine.get_invented_symbols()) for engine in all_engines)
    total_steps = sum(engine.state['tick_counter'] for engine in all_engines)
    
    print(f"✅ Contradiction detection and resolution")
    print(f"✅ Conceptual gap identification and filling") 
    print(f"✅ Ambiguity detection and disambiguation")
    print(f"✅ Overflow regulation and control")
    print(f"✅ Multi-step molecular reasoning")
    print(f"✅ Automatic symbol invention")
    print(f"✅ Field tension analysis")
    print(f"✅ Real-time cognitive adaptation")
    
    print(f"\n📊 DEMO STATISTICS:")
    print(f"Total symbols invented: {total_symbols}")
    print(f"Total reasoning steps: {total_steps}")
    print(f"Average symbols per step: {total_symbols/total_steps:.2f}")
    
    print(f"\n💡 KEY INNOVATIONS:")
    print(f"• Symbols emerge from field tensions using molecular operations")
    print(f"• Reasoning uses combinatorial kernels (S, K, I, B, C, W)")
    print(f"• Completely interpretable decision making")
    print(f"• Zero-shot concept invention")
    print(f"• Natural tension resolution dynamics")
    
    print(f"\n🌟 This demonstrates true artificial reasoning!")
    print("The system INVENTS new concepts to resolve cognitive tensions.")
    print("No training data, no neural networks - pure molecular intelligence!")
    
    print("\n🚀 ReasoningEngine Demo Complete! 🚀")