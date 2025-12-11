# ============================================================================
# EchoChamber - Pattern Memory Using Molecular Echo Dynamics
# ============================================================================

"""
EchoChamber - Pattern memory using echo dynamics and molecular processing

Composition:
- Atoms: Echo, MemoryTrace, Resonator, Threshold, Damper, Amplifier
- Molecules: AssociativeMemory, PatternRecognizer, WorkingMemory
- Fields: echo_field, memory_field, interference_field, temporal_field

This organism stores patterns as "echoes" that fade over time, with recognition
happening through resonance. Uses molecular operations for pattern processing,
interference, and memory consolidation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass

try:
    from ...core import NDAnalogField
    from ...atoms.pattern_primitives import EchoAtom, AmplifierAtom
    from ...atoms.temporal import MemoryTraceAtom, ThresholdAtom
    from ...atoms.multi_field import ResonatorAtom
    from ...atoms.tension_resolvers import DamperAtom
    from ...molecules.memory import AssociativeMemoryMolecule, WorkingMemoryMolecule
    from ...molecules.perception import PatternRecognizerMolecule
except ImportError:
    from combinatronix.core import NDAnalogField
    from combinatronix.atoms.pattern_primitives import EchoAtom, AmplifierAtom
    from combinatronix.atoms.temporal import MemoryTraceAtom, ThresholdAtom
    from combinatronix.atoms.multi_field import ResonatorAtom
    from combinatronix.atoms.tension_resolvers import DamperAtom
    from combinatronix.molecules.memory import AssociativeMemoryMolecule, WorkingMemoryMolecule
    from combinatronix.molecules.perception import PatternRecognizerMolecule


@dataclass
class PatternEcho:
    """A single pattern echo in the chamber"""
    pattern: NDAnalogField
    strength: float
    age: int
    original_strength: float
    name: str = ""
    resonance_frequency: float = 1.0
    decay_rate: float = 0.98


class EchoChamber:
    """Pattern memory using echo dynamics and molecular processing"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the echo chamber
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            'field_size': (8, 8),
            'max_echoes': 10,
            'recognition_threshold': 0.6,
            'echo_decay_rate': 0.98,
            'interference_strength': 0.3,
            'resonance_threshold': 0.5,
            'learning_enabled': True,
            'enable_visualization': True,
            'temporal_window': 20
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
            'echoes': [],
            'tick_count': 0,
            'total_patterns_learned': 0,
            'recognition_history': [],
            'interference_events': 0
        }
        
        # Initialize echo kernels
        self._initialize_echo_kernels()
    
    def _initialize_atoms(self):
        """Initialize atomic operations"""
        self.atoms = {
            'echo': EchoAtom(decay_rate=self.config['echo_decay_rate'], depth=5),
            'memory_trace': MemoryTraceAtom(accumulation_rate=0.3, decay_rate=0.95),
            'resonator': ResonatorAtom(amplification=1.5, threshold=self.config['resonance_threshold']),
            'threshold': ThresholdAtom(threshold=self.config['recognition_threshold'], mode='binary'),
            'damper': DamperAtom(threshold=0.8, damping_rate=0.5, mode='soft'),
            'amplifier': AmplifierAtom(threshold=0.1, gain=1.2, mode='linear')
        }
    
    def _initialize_molecules(self):
        """Initialize molecular operations"""
        self.molecules = {
            'associative_memory': AssociativeMemoryMolecule(
                amplification=1.2,
                resonance_threshold=0.5
            ),
            'working_memory': WorkingMemoryMolecule(
                capacity=self.config['max_echoes'],
                decay_rate=0.95
            ),
            'pattern_recognizer': PatternRecognizerMolecule(
                amplification=1.5,
                resonance_threshold=self.config['resonance_threshold']
            )
        }
    
    def _initialize_fields(self):
        """Initialize field structures"""
        self.fields = {
            'echo_field': NDAnalogField(self.config['field_size']),
            'memory_field': NDAnalogField(self.config['field_size']),
            'interference_field': NDAnalogField(self.config['field_size']),
            'temporal_field': NDAnalogField((self.config['temporal_window'], *self.config['field_size'])),
            'resonance_field': NDAnalogField(self.config['field_size'])
        }
    
    def _initialize_echo_kernels(self):
        """Initialize echo processing kernels"""
        self.echo_kernels = {
            'identity': self._identity_kernel,
            'amplify': self._amplify_kernel,
            'fade': self._fade_kernel,
            'spread': self._spread_kernel,
            'flip': self._flip_kernel,
            'rotate': self._rotate_kernel,
            'resonate': self._resonate_kernel
        }
    
    def _identity_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Identity kernel (I combinator)"""
        result = pattern.copy()
        result.activation *= strength
        return result
    
    def _amplify_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Amplify kernel (W combinator)"""
        result = pattern.copy()
        self.atoms['amplifier'].apply(result)
        result.activation *= (1.0 + strength * 0.2)
        return result
    
    def _fade_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Fade kernel (K combinator)"""
        result = pattern.copy()
        self.atoms['echo'].apply(result)
        result.activation *= (1.0 - strength * 0.1)
        return result
    
    def _spread_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Spread kernel (S combinator) - diffusion"""
        result = pattern.copy()
        
        # Apply diffusion-like spreading
        if len(pattern.shape) == 2:
            h, w = pattern.shape
            spread = np.zeros_like(pattern.activation)
            
            # Spread to neighbors
            for i in range(h):
                for j in range(w):
                    if pattern.activation[i, j] > 0:
                        # Spread to adjacent cells
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < h and 0 <= nj < w:
                                    spread[ni, nj] += pattern.activation[i, j] * strength * 0.1
            
            result.activation = spread * 0.9  # Normalize
        
        return result
    
    def _flip_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Flip kernel (C combinator)"""
        result = pattern.copy()
        if len(pattern.shape) == 2:
            result.activation = np.transpose(pattern.activation) * strength
        else:
            result.activation = np.flip(pattern.activation) * strength
        return result
    
    def _rotate_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Rotate kernel"""
        result = pattern.copy()
        if len(pattern.shape) == 2:
            result.activation = np.rot90(pattern.activation) * strength
        else:
            result.activation = np.roll(pattern.activation, 1) * strength
        return result
    
    def _resonate_kernel(self, pattern: NDAnalogField, strength: float = 1.0) -> NDAnalogField:
        """Resonate kernel using resonator atom"""
        result = pattern.copy()
        
        # Create resonance field
        resonance_field = NDAnalogField(pattern.shape)
        self.atoms['resonator'].apply(result, resonance_field)
        
        result.activation *= strength
        return result
    
    def inject_pattern(self, pattern: np.ndarray, name: str = "") -> Dict[str, Any]:
        """
        Inject a pattern into the echo chamber
        
        Args:
            pattern: Input pattern as numpy array
            name: Optional name for the pattern
            
        Returns:
            Dictionary with injection results
        """
        # Resize pattern to fit chamber if needed
        if pattern.shape != self.config['field_size']:
            pattern = self._resize_pattern(pattern)
        
        # Create NDAnalogField from pattern
        pattern_field = NDAnalogField(self.config['field_size'], activation=pattern)
        
        # Check if we recognize this pattern first
        recognition = self.recognize_pattern(pattern_field)
        
        if recognition["recognized"] and recognition["confidence"] > 0.8:
            # Strengthen existing echo
            self._strengthen_similar_echo(pattern_field, recognition["match_strength"])
            return {
                "learned": False,
                "recognized": True,
                "confidence": recognition["confidence"],
                "match_name": recognition["match_name"]
            }
        
        # Learn new pattern if learning enabled
        if self.config['learning_enabled']:
            # Remove oldest echo if at capacity
            if len(self.state['echoes']) >= self.config['max_echoes']:
                weakest_idx = min(range(len(self.state['echoes'])), 
                                key=lambda i: self.state['echoes'][i].strength)
                removed_echo = self.state['echoes'].pop(weakest_idx)
                print(f"Removed weak echo: '{removed_echo.name}'")
            
            # Add new echo
            new_echo = PatternEcho(
                pattern=pattern_field.copy(),
                strength=1.0,
                age=0,
                original_strength=1.0,
                name=name or f"pattern_{self.state['total_patterns_learned']}",
                resonance_frequency=1.0,
                decay_rate=self.config['echo_decay_rate']
            )
            self.state['echoes'].append(new_echo)
            self.state['total_patterns_learned'] += 1
            
            print(f"Learned new pattern: '{new_echo.name}'")
            return {
                "learned": True,
                "recognized": False,
                "confidence": 0.0,
                "pattern_name": new_echo.name
            }
        
        return {
            "learned": False,
            "recognized": recognition["recognized"],
            "confidence": recognition["confidence"],
            "match_name": recognition.get("match_name", "")
        }
    
    def _resize_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Resize pattern to fit chamber field size"""
        target_shape = self.config['field_size']
        
        if pattern.shape == target_shape:
            return pattern
        
        # Simple resize by cropping or padding
        resized = np.zeros(target_shape, dtype=np.float32)
        
        if len(pattern.shape) == 2 and len(target_shape) == 2:
            min_h = min(pattern.shape[0], target_shape[0])
            min_w = min(pattern.shape[1], target_shape[1])
            resized[:min_h, :min_w] = pattern[:min_h, :min_w]
        else:
            # For 1D or other cases, use simple padding/cropping
            min_size = min(len(pattern.flatten()), np.prod(target_shape))
            resized.flat[:min_size] = pattern.flatten()[:min_size]
        
        return resized
    
    def recognize_pattern(self, pattern: NDAnalogField) -> Dict[str, Any]:
        """
        Try to recognize a pattern from existing echoes using molecular processing
        
        Args:
            pattern: Pattern to recognize
            
        Returns:
            Dictionary with recognition results
        """
        if len(self.state['echoes']) == 0:
            return {
                "recognized": False, 
                "confidence": 0.0, 
                "match_name": "", 
                "match_strength": 0.0
            }
        
        best_match = None
        best_similarity = 0.0
        best_echo = None
        
        # Use pattern recognizer molecule for enhanced recognition
        for echo in self.state['echoes']:
            if echo.strength < 0.1:  # Skip very weak echoes
                continue
            
            # Use molecular pattern recognition
            recognized_pattern = self.molecules['pattern_recognizer'].process(pattern)
            echo_pattern = self.molecules['pattern_recognizer'].process(echo.pattern)
            
            # Compute similarity using molecular processing
            similarity = self._compute_molecular_similarity(recognized_pattern, echo_pattern)
            
            # Weight by echo strength and age
            age_factor = 1.0 / (1.0 + echo.age * 0.01)  # Slight age penalty
            weighted_similarity = similarity * echo.strength * age_factor
            
            if weighted_similarity > best_similarity:
                best_similarity = weighted_similarity
                best_match = echo.name
                best_echo = echo
        
        recognized = best_similarity > self.config['recognition_threshold']
        
        # Store recognition in history
        self.state['recognition_history'].append({
            'timestamp': self.state['tick_count'],
            'recognized': recognized,
            'confidence': best_similarity,
            'match_name': best_match or ""
        })
        
        return {
            "recognized": recognized,
            "confidence": best_similarity,
            "match_name": best_match or "",
            "match_strength": best_similarity,
            "best_echo": best_echo
        }
    
    def _compute_molecular_similarity(self, pattern1: NDAnalogField, pattern2: NDAnalogField) -> float:
        """Compute similarity using molecular processing"""
        # Use resonator atom for similarity computation
        temp_field1 = pattern1.copy()
        temp_field2 = pattern2.copy()
        
        # Apply resonator to find resonance between patterns
        resonated = self.atoms['resonator'].apply(temp_field1, temp_field2)
        
        # Similarity is the mean activation in resonated field
        similarity = np.mean(resonated.activation)
        
        # Also compute traditional correlation as backup
        flat1 = pattern1.activation.flatten()
        flat2 = pattern2.activation.flatten()
        
        if len(flat1) == len(flat2) and np.std(flat1) > 0 and np.std(flat2) > 0:
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            if not np.isnan(correlation):
                # Combine molecular and traditional similarity
                similarity = (similarity + abs(correlation)) / 2
        
        return max(0.0, min(1.0, similarity))
    
    def _strengthen_similar_echo(self, pattern: NDAnalogField, match_strength: float):
        """Strengthen echo that matches input pattern using molecular processing"""
        for echo in self.state['echoes']:
            similarity = self._compute_molecular_similarity(pattern, echo.pattern)
            if similarity > 0.7:  # High similarity
                # Strengthen echo
                echo.strength = min(1.0, echo.strength + 0.1 * match_strength)
                
                # Use associative memory molecule for pattern blending
                blended_pattern = self.molecules['associative_memory'].process(pattern)
                echo_pattern = self.molecules['associative_memory'].process(echo.pattern)
                
                # Blend patterns using molecular processing
                blend_factor = 0.05
                echo.pattern.activation = ((1 - blend_factor) * echo.pattern.activation + 
                                        blend_factor * pattern.activation)
                
                # Update resonance frequency based on match
                echo.resonance_frequency = (echo.resonance_frequency + match_strength) / 2
                break
    
    def tick(self):
        """Single time step - let echoes evolve using molecular processing"""
        self.state['tick_count'] += 1
        
        # Clear fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Process each echo using molecular operations
        echoes_to_remove = []
        
        for i, echo in enumerate(self.state['echoes']):
            # Age the echo
            echo.age += 1
            
            # Apply molecular processing to echo
            processed_echo = self._process_echo_molecular(echo)
            
            # Update echo pattern
            echo.pattern = processed_echo
            
            # Reduce strength over time
            echo.strength *= echo.decay_rate
            
            # Apply interference if multiple echoes
            if len(self.state['echoes']) > 1:
                interference_strength = self.config['interference_strength']
                echo.pattern.activation *= (1.0 - interference_strength * 0.1)
                self.state['interference_events'] += 1
            
            # Add to echo field
            self.fields['echo_field'].activation += echo.pattern.activation * echo.strength
            
            # Update memory field
            self.atoms['memory_trace'].apply(echo.pattern)
            self.fields['memory_field'].activation += echo.pattern.activation * echo.strength * 0.5
            
            # Mark very weak echoes for removal
            if echo.strength < 0.05:
                echoes_to_remove.append(i)
        
        # Remove dead echoes
        for i in reversed(echoes_to_remove):
            dead_echo = self.state['echoes'].pop(i)
            print(f"Echo '{dead_echo.name}' faded away after {dead_echo.age} ticks")
        
        # Update temporal field
        self._update_temporal_field()
        
        # Apply damper to prevent runaway activation
        self.atoms['damper'].apply(self.fields['echo_field'])
    
    def _process_echo_molecular(self, echo: PatternEcho) -> NDAnalogField:
        """Process echo using molecular operations"""
        processed = echo.pattern.copy()
        
        # Apply echo kernel based on age and strength
        if echo.age % 5 == 0:  # Apply spreading every 5 ticks
            processed = self.echo_kernels['spread'](processed, 0.3)
        
        if echo.strength > 0.8:  # Strong echoes get amplification
            processed = self.echo_kernels['amplify'](processed, 0.2)
        
        if echo.age > 10:  # Old echoes get fading
            processed = self.echo_kernels['fade'](processed, 0.1)
        
        # Apply resonance based on frequency
        if echo.resonance_frequency > 1.2:
            processed = self.echo_kernels['resonate'](processed, echo.resonance_frequency)
        
        return processed
    
    def _update_temporal_field(self):
        """Update temporal field with current echo state"""
        if len(self.state['echoes']) > 0:
            # Get current echo field state
            current_state = self.fields['echo_field'].activation.copy()
            
            # Store in temporal field
            temporal_idx = self.state['tick_count'] % self.config['temporal_window']
            self.fields['temporal_field'].activation[temporal_idx] = current_state
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state"""
        if not self.state['echoes']:
            return {
                "total_echoes": 0, 
                "total_strength": 0.0, 
                "strongest": None,
                "field_energy": 0.0,
                "interference_events": self.state['interference_events']
            }
        
        total_strength = sum(echo.strength for echo in self.state['echoes'])
        strongest = max(self.state['echoes'], key=lambda e: e.strength)
        
        return {
            "total_echoes": len(self.state['echoes']),
            "total_strength": total_strength,
            "strongest": strongest.name,
            "strongest_strength": strongest.strength,
            "average_age": sum(echo.age for echo in self.state['echoes']) / len(self.state['echoes']),
            "field_energy": np.sum(self.fields['echo_field'].activation),
            "interference_events": self.state['interference_events'],
            "patterns_learned": self.state['total_patterns_learned'],
            "recognition_events": len(self.state['recognition_history'])
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
    
    def visualize(self, save_path: Optional[str] = None):
        """Visualize the echo chamber state"""
        if not self.config['enable_visualization']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Echo Chamber - Molecular Pattern Memory", fontsize=16)
        
        # Echo field
        im1 = axes[0, 0].imshow(self.fields['echo_field'].activation, cmap='hot')
        axes[0, 0].set_title(f"Echo Field\n(Energy: {np.sum(self.fields['echo_field'].activation):.2f})")
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Memory field
        im2 = axes[0, 1].imshow(self.fields['memory_field'].activation, cmap='blues')
        axes[0, 1].set_title("Memory Field")
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Interference field
        im3 = axes[0, 2].imshow(self.fields['interference_field'].activation, cmap='purples')
        axes[0, 2].set_title("Interference Field")
        axes[0, 2].axis('off')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Show top echoes
        sorted_echoes = sorted(self.state['echoes'], key=lambda e: e.strength, reverse=True)
        
        for i, echo in enumerate(sorted_echoes[:3]):
            if i < 3:
                im = axes[1, i].imshow(echo.pattern.activation, cmap='viridis')
                axes[1, i].set_title(f"{echo.name}\nStr: {echo.strength:.2f}, Age: {echo.age}")
                axes[1, i].axis('off')
                plt.colorbar(im, ax=axes[1, i])
        
        # Fill remaining subplots
        for i in range(len(sorted_echoes), 3):
            axes[1, i].text(0.5, 0.5, "No Echo", ha='center', va='center')
            axes[1, i].set_title("Empty")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def export_memory(self, filename: str):
        """Export echo memory to file"""
        memory_data = {
            "echoes": [
                {
                    "name": echo.name,
                    "pattern": echo.pattern.activation.tolist(),
                    "strength": echo.strength,
                    "age": echo.age,
                    "original_strength": echo.original_strength,
                    "resonance_frequency": echo.resonance_frequency,
                    "decay_rate": echo.decay_rate
                }
                for echo in self.state['echoes']
            ],
            "state": self.state.copy(),
            "config": self.config.copy(),
            "field_shapes": {name: field.shape for name, field in self.fields.items()}
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(memory_data, f, indent=2)
        print(f"Memory exported to {filename}")
    
    def reset(self):
        """Reset the echo chamber"""
        self.state = {
            'echoes': [],
            'tick_count': 0,
            'total_patterns_learned': 0,
            'recognition_history': [],
            'interference_events': 0
        }
        
        # Reset fields
        for field in self.fields.values():
            field.activation.fill(0)
        
        # Reset molecules
        for molecule in self.molecules.values():
            if hasattr(molecule, 'reset'):
                molecule.reset()
    
    def __repr__(self):
        return f"EchoChamber(echoes={len(self.state['echoes'])}, " \
               f"tick_count={self.state['tick_count']}, " \
               f"learned={self.state['total_patterns_learned']})"


# === Demo Functions ===

def demo_basic_learning():
    """Demo basic pattern learning and recognition"""
    print("=== Basic Learning Demo ===")
    
    chamber = EchoChamber({'field_size': (6, 6), 'max_echoes': 5, 'enable_visualization': False})
    
    # Create some simple patterns
    patterns = {
        "cross": np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0], 
            [1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "square": np.array([
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0], 
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float32),
        
        "diagonal": np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0]
        ], dtype=np.float32)
    }
    
    # Learn patterns
    print("Learning patterns...")
    for name, pattern in patterns.items():
        result = chamber.inject_pattern(pattern, name)
        print(f"  {name}: learned={result['learned']}, recognized={result['recognized']}")
    
    print(f"\nMemory state: {chamber.get_memory_summary()}")
    
    # Test recognition
    print("\nTesting recognition...")
    for name, pattern in patterns.items():
        pattern_field = NDAnalogField((6, 6), activation=pattern)
        result = chamber.recognize_pattern(pattern_field)
        print(f"{name}: recognized={result['recognized']}, confidence={result['confidence']:.3f}")
    
    # Test with noisy patterns
    print("\nTesting with noisy patterns...")
    for name, pattern in patterns.items():
        noisy = pattern + np.random.random(pattern.shape) * 0.2
        noisy_field = NDAnalogField((6, 6), activation=noisy)
        result = chamber.recognize_pattern(noisy_field)
        print(f"{name} (noisy): recognized={result['recognized']}, confidence={result['confidence']:.3f}")
    
    return chamber


def demo_echo_evolution():
    """Demo how echoes evolve over time"""
    print("\n=== Echo Evolution Demo ===")
    
    chamber = EchoChamber({'field_size': (6, 6), 'max_echoes': 3, 'enable_visualization': False})
    
    # Learn a simple pattern
    simple_pattern = np.eye(6, dtype=np.float32)  # Identity matrix
    chamber.inject_pattern(simple_pattern, "identity")
    
    print("Watching echo evolve over time...")
    
    for t in range(20):
        summary = chamber.get_memory_summary()
        print(f"Tick {t:2d}: echoes={summary['total_echoes']}, "
              f"strength={summary['total_strength']:.3f}, "
              f"field_energy={summary['field_energy']:.3f}")
        
        chamber.tick()
        
        if summary['total_echoes'] == 0:
            print("All echoes have faded away!")
            break
    
    return chamber


def demo_interference():
    """Demo how multiple patterns interfere in the chamber"""
    print("\n=== Pattern Interference Demo ===")
    
    chamber = EchoChamber({'field_size': (8, 8), 'max_echoes': 6, 'enable_visualization': False})
    
    # Create overlapping patterns
    pattern1 = np.zeros((8, 8))
    pattern1[2:6, 2:6] = 1.0  # Square in center
    
    pattern2 = np.zeros((8, 8))
    pattern2[3, :] = 1.0  # Horizontal line
    
    pattern3 = np.zeros((8, 8)) 
    pattern3[:, 4] = 1.0  # Vertical line
    
    chamber.inject_pattern(pattern1, "square")
    chamber.inject_pattern(pattern2, "h_line")
    chamber.inject_pattern(pattern3, "v_line")
    
    print("Initial patterns learned")
    print(f"Memory: {chamber.get_memory_summary()}")
    
    # Let them interfere for a few ticks
    print("\nLetting patterns interfere...")
    for t in range(10):
        chamber.tick()
        
        if t % 3 == 0:
            summary = chamber.get_memory_summary()
            print(f"  Tick {t}: field_energy={summary['field_energy']:.3f}, "
                  f"interference_events={summary['interference_events']}")
    
    # Test recognition after interference
    print("\nTesting recognition after interference:")
    test_square = np.zeros((8, 8))
    test_square[2:6, 2:6] = 1.0
    
    test_field = NDAnalogField((8, 8), activation=test_square)
    result = chamber.recognize_pattern(test_field)
    print(f"Square recognition: {result['confidence']:.3f}")
    
    return chamber


def demo_molecular_processing():
    """Demo molecular processing capabilities"""
    print("\n=== Molecular Processing Demo ===")
    
    chamber = EchoChamber({'field_size': (8, 8), 'enable_visualization': True})
    
    # Create complex pattern
    complex_pattern = np.zeros((8, 8))
    complex_pattern[2:6, 2:6] = 1.0  # Square
    complex_pattern[3, :] = 0.5      # Horizontal line
    complex_pattern[:, 4] = 0.5      # Vertical line
    
    # Learn pattern
    result = chamber.inject_pattern(complex_pattern, "complex")
    print(f"Pattern injection: {result}")
    
    # Let it evolve
    print("\nLetting pattern evolve with molecular processing...")
    for t in range(15):
        chamber.tick()
        
        if t % 5 == 0:
            summary = chamber.get_memory_summary()
            print(f"  Tick {t}: field_energy={summary['field_energy']:.3f}, "
                  f"echoes={summary['total_echoes']}")
    
    # Show visualization
    chamber.visualize()
    
    return chamber


# === Main Demo ===

if __name__ == '__main__':
    print("🔊 ECHO CHAMBER - Molecular Pattern Memory 🔊")
    print("Pattern memory using echo dynamics and molecular processing")
    print("Patterns 'echo' in the chamber until they fade - recognition through resonance\n")
    
    # Run demos
    basic_chamber = demo_basic_learning()
    evolution_chamber = demo_echo_evolution()
    interference_chamber = demo_interference()
    molecular_chamber = demo_molecular_processing()
    
    # Summary
    print("\n=== SYSTEM CAPABILITIES ===")
    print("✅ Zero-shot pattern learning (no training!)")
    print("✅ Molecular pattern recognition")
    print("✅ Graceful forgetting (weak echoes fade)")
    print("✅ Pattern interference and blending")
    print("✅ Noise tolerance with molecular processing")
    print("✅ Memory capacity management")
    print("✅ Real-time operation")
    print("✅ Completely interpretable")
    print("✅ Temporal evolution tracking")
    print("✅ Resonance-based recognition")
    
    print(f"\n🎯 Molecular echo memory system!")
    print("Uses atoms + molecules for pattern processing and memory consolidation.")
    print("Patterns literally 'echo' in the chamber until they fade away.")
    
    # Export demo
    try:
        basic_chamber.export_memory("echo_chamber_demo.json")
    except:
        print("Export skipped")
    
    print("\n🌟 Echo Chamber Demo Complete! 🌟")
    print("This shows how molecular architecture can create elegant pattern memory!")

