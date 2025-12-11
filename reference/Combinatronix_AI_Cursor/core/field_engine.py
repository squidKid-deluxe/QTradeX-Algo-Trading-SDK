
# ============================================================================
# FIELD ENGINE - field_engine.py
# ============================================================================

"""
N-Dimensional Analog Field Engine

Multi-dimensional fields with configurable propagation dynamics.
Supports diffusion, wave, combustion, and other propagation modes.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Callable
import numpy as np


class PropagationMode(Enum):
    """Field propagation modes"""
    DIFFUSION = "diffusion"
    WAVE = "wave"
    COMBUSTION = "combustion"
    FIELD_TENSION = "field_tension"
    PROBABILISTIC = "probabilistic"
    TURBULENT = "turbulent"
    BLOCK_TRANSFER = "block_transfer"
    RESONANCE = "resonance"
    ATTENTION = "attention"


@dataclass
class FieldConfig:
    """Configuration for field behavior"""
    base_resistance: float = 1.0
    base_capacitance: float = 1.0
    spread_rate: float = 0.1
    decay_rate: float = 0.95
    threshold: float = 0.5


class NDAnalogField:
    """N-Dimensional Analog Field with configurable dynamics"""
    
    def __init__(self, shape: Tuple[int, ...], config: Optional[FieldConfig] = None):
        self.shape = shape
        self.dim = len(shape)
        self.config = config or FieldConfig()
        
        # Core field properties
        self.activation = np.zeros(shape, dtype=np.float32)
        self.memory = np.zeros(shape, dtype=np.float32)
        self.phase = np.zeros(shape, dtype=np.complex64)
        self.resistance = np.full(shape, self.config.base_resistance, dtype=np.float32)
        self.capacitance = np.full(shape, self.config.base_capacitance, dtype=np.float32)
        
        # Metadata
        self.glyph_layer = np.full(shape, '', dtype=object)
        self.mode = PropagationMode.DIFFUSION
        
        # For wave propagation
        self._prev_activation = None
        
    def inject_signal(self, coord: Tuple[int, ...], strength: float = 1.0):
        """Inject signal at specific coordinate"""
        if self._valid_coord(coord):
            self.activation[coord] += strength
    
    def inject_pattern(self, pattern: np.ndarray, position: Optional[Tuple[int, ...]] = None):
        """Inject a pattern into the field"""
        if position is None:
            position = tuple(0 for _ in self.shape)
        
        # Handle broadcasting
        end_pos = tuple(min(position[i] + pattern.shape[i], self.shape[i]) 
                       for i in range(min(len(position), len(pattern.shape))))
        
        slices = tuple(slice(position[i], end_pos[i]) for i in range(len(end_pos)))
        pattern_slices = tuple(slice(0, end_pos[i] - position[i]) for i in range(len(end_pos)))
        
        self.activation[slices] += pattern[pattern_slices]
    
    def _valid_coord(self, coord: Tuple[int, ...]) -> bool:
        """Check if coordinate is valid"""
        return all(0 <= coord[i] < self.shape[i] for i in range(len(coord)))
    
    def neighbors(self, index: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get valid neighbors of a coordinate"""
        neighbors = []
        for d in range(self.dim):
            for offset in [-1, 1]:
                neighbor = list(index)
                neighbor[d] += offset
                if 0 <= neighbor[d] < self.shape[d]:
                    neighbors.append(tuple(neighbor))
        return neighbors
    
    def propagate(self, steps: int = 1, **kwargs):
        """Propagate field dynamics for given steps"""
        for _ in range(steps):
            if self.mode == PropagationMode.DIFFUSION:
                self._propagate_diffusion()
            elif self.mode == PropagationMode.WAVE:
                self._propagate_wave(kwargs.get('frequency', 1.0))
            elif self.mode == PropagationMode.COMBUSTION:
                self._propagate_combustion(kwargs.get('threshold', self.config.threshold))
            elif self.mode == PropagationMode.FIELD_TENSION:
                self._propagate_field_tension(kwargs.get('strength', 0.1))
            elif self.mode == PropagationMode.PROBABILISTIC:
                self._propagate_probabilistic(kwargs.get('uncertainty', 0.2))
            elif self.mode == PropagationMode.RESONANCE:
                self._propagate_resonance(kwargs.get('frequency', 1.0))
            elif self.mode == PropagationMode.ATTENTION:
                self._propagate_attention()
    
    def _propagate_diffusion(self):
        """Standard diffusion propagation"""
        new_activation = np.copy(self.activation)
        
        # Iterate through all coordinates
        it = np.nditer(self.activation, flags=['multi_index'])
        for value in it:
            idx = it.multi_index
            v_current = self.activation[idx]
            
            if v_current <= 0:
                continue
            
            # Transfer to neighbors
            for n_idx in self.neighbors(idx):
                r = self.resistance[n_idx]
                transfer = (v_current / r) * self.config.spread_rate
                new_activation[n_idx] += transfer
                new_activation[idx] -= transfer
            
            # Apply decay
            new_activation[idx] *= self.config.decay_rate
        
        # Update memory
        self.memory += new_activation * 0.1
        self.activation = new_activation
    
    def _propagate_wave(self, frequency: float = 1.0):
        """Wave equation propagation"""
        dt = 0.1
        c_squared = 1.0  # Wave speed squared
        
        # Compute Laplacian
        laplacian = np.zeros_like(self.activation)
        it = np.nditer(self.activation, flags=['multi_index'])
        
        for value in it:
            idx = it.multi_index
            neighbors_sum = sum(self.activation[n_idx] for n_idx in self.neighbors(idx))
            n_count = len(self.neighbors(idx))
            laplacian[idx] = neighbors_sum - n_count * self.activation[idx]
        
        # Wave equation: ∂²u/∂t² = c²∇²u
        if self._prev_activation is None:
            self._prev_activation = self.activation.copy()
        
        new_activation = (2 * self.activation - self._prev_activation + 
                         c_squared * laplacian * dt**2)
        
        self._prev_activation = self.activation.copy()
        self.activation = new_activation * self.config.decay_rate
    
    def _propagate_combustion(self, threshold: float):
        """Combustion/cascade propagation"""
        new_activation = np.copy(self.activation)
        
        it = np.nditer(self.activation, flags=['multi_index'])
        for value in it:
            idx = it.multi_index
            
            if self.activation[idx] > threshold:
                # Ignite neighbors
                for n_idx in self.neighbors(idx):
                    if self.activation[n_idx] > threshold * 0.5:
                        new_activation[n_idx] *= 2.0  # Amplify
                        new_activation[idx] *= 0.5   # Consume
        
        self.activation = np.clip(new_activation, 0, 2.0)
    
    def _propagate_field_tension(self, strength: float):
        """Gradient-based attraction/repulsion"""
        # Compute gradient
        gradients = np.gradient(self.activation)
        
        # Move along gradient
        for i, grad in enumerate(gradients):
            shift = np.roll(self.activation, 1, axis=i) - self.activation
            self.activation += shift * strength
        
        self.activation *= self.config.decay_rate
    
    def _propagate_probabilistic(self, uncertainty: float):
        """Probabilistic/stochastic propagation"""
        new_activation = np.copy(self.activation)
        
        it = np.nditer(self.activation, flags=['multi_index'])
        for value in it:
            idx = it.multi_index
            
            if self.activation[idx] > 0:
                neighbors = self.neighbors(idx)
                if neighbors:
                    # Random distribution to neighbors
                    probs = np.random.exponential(uncertainty, len(neighbors))
                    probs /= probs.sum() if probs.sum() > 0 else 1
                    
                    transfer = self.activation[idx] * 0.1
                    for prob, n_idx in zip(probs, neighbors):
                        new_activation[n_idx] += transfer * prob
                        new_activation[idx] -= transfer * prob
        
        self.activation = new_activation
    
    def _propagate_resonance(self, frequency: float):
        """Resonance amplification"""
        # Update phase
        self.phase *= np.exp(1j * frequency * 0.1)
        
        # Amplify regions with high phase coherence
        phase_coherence = np.abs(self.phase)
        self.activation *= (1.0 + phase_coherence * 0.1)
        self.activation *= self.config.decay_rate
    
    def _propagate_attention(self):
        """Attention-weighted propagation"""
        # Find high-activation regions
        attention_map = self.activation / (np.max(self.activation) + 1e-8)
        
        # Amplify attended regions
        self.activation *= (1.0 + attention_map * 0.2)
        
        # Standard diffusion with attention weighting
        self._propagate_diffusion()
    
    def place_glyph(self, coord: Tuple[int, ...], glyph: str):
        """Place a symbolic glyph at coordinate"""
        if self._valid_coord(coord):
            self.glyph_layer[coord] = glyph
    
    def reset(self):
        """Reset field to initial state"""
        self.activation.fill(0)
        self.memory.fill(0)
        self.phase.fill(0)
        self.glyph_layer.fill('')
        self._prev_activation = None
    
    def get_energy(self) -> float:
        """Get total field energy"""
        return np.sum(np.abs(self.activation))
    
    def get_state(self) -> dict:
        """Get complete field state"""
        return {
            'activation': self.activation.copy(),
            'memory': self.memory.copy(),
            'phase': self.phase.copy(),
            'glyphs': self.glyph_layer.copy(),
            'energy': self.get_energy()
        }

