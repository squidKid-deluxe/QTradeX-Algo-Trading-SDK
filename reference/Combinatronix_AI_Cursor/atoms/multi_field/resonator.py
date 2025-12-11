# ============================================================================
# 1. RESONATOR - resonator.py
# ============================================================================

"""
The Resonator - Amplify Phase Matching

Archetype: Harmony, recognition, sympathy
Category: Multi-Field
Complexity: 22 lines

Amplifies activation when fields are in phase/alignment. Foundation of
pattern recognition, harmony, and sympathetic resonance.

Usage:
    >>> resonator = ResonatorAtom(amplification=2.0, threshold=0.7)
    >>> resonator.apply(field_a, field_b)
"""

import numpy as np
try:
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.core import NDAnalogField


class ResonatorAtom:
    """Amplify when fields align/resonate"""
    
    def __init__(self, amplification: float = 2.0, threshold: float = 0.5, 
                 mode: str = 'correlation'):
        """
        Args:
            amplification: How much to amplify resonant regions
            threshold: Minimum correlation for resonance
            mode: 'correlation', 'phase', 'product'
        """
        self.amplification = amplification
        self.threshold = threshold
        self.mode = mode
        self.resonance_history = []
    
    def apply(self, field_a: NDAnalogField, field_b: NDAnalogField = None):
        """Amplify regions where fields resonate"""
        if field_b is None:
            # Self-resonance using phase field
            if field_a.phase is not None and np.any(field_a.phase != 0):
                resonance_map = self._compute_phase_coherence(field_a)
            else:
                return field_a
        else:
            # Cross-field resonance
            resonance_map = self._compute_resonance(field_a, field_b)
        
        # Amplify resonant regions
        field_a.activation *= (1.0 + resonance_map * (self.amplification - 1.0))
        
        # Track resonance
        self.resonance_history.append(np.mean(resonance_map))
        
        return field_a
    
    def _compute_resonance(self, field_a: NDAnalogField, field_b: NDAnalogField) -> np.ndarray:
        """Compute resonance map between two fields"""
        # Ensure same shape
        if field_a.shape != field_b.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(field_a.shape, field_b.shape))
            act_a = field_a.activation[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else field_a.activation
            act_b = field_b.activation[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else field_b.activation
        else:
            act_a = field_a.activation
            act_b = field_b.activation
        
        if self.mode == 'correlation':
            # Pointwise correlation (normalized dot product)
            norm_a = act_a / (np.abs(act_a) + 1e-8)
            norm_b = act_b / (np.abs(act_b) + 1e-8)
            resonance = norm_a * norm_b
            
        elif self.mode == 'phase':
            # Phase alignment (using complex representation)
            phase_a = np.angle(field_a.phase) if hasattr(field_a, 'phase') else 0
            phase_b = np.angle(field_b.phase) if hasattr(field_b, 'phase') else 0
            phase_diff = np.abs(phase_a - phase_b)
            resonance = np.cos(phase_diff)
            
        elif self.mode == 'product':
            # Simple product (both must be active)
            resonance = act_a * act_b
            max_product = np.max(np.abs(resonance)) + 1e-8
            resonance = resonance / max_product
        
        # Apply threshold
        resonance = np.where(resonance > self.threshold, resonance, 0)
        
        return resonance
    
    def _compute_phase_coherence(self, field: NDAnalogField) -> np.ndarray:
        """Compute internal phase coherence"""
        if not hasattr(field, 'phase') or field.phase is None:
            return np.zeros_like(field.activation)
        
        # Measure local phase coherence
        phase_angles = np.angle(field.phase)
        
        # Compute phase gradient
        if len(field.shape) == 2:
            grad_y, grad_x = np.gradient(phase_angles)
            phase_gradient = np.sqrt(grad_y**2 + grad_x**2)
            
            # Low gradient = high coherence
            coherence = 1.0 / (1.0 + phase_gradient)
        else:
            coherence = np.ones_like(field.activation)
        
        return coherence
    
    def get_resonance_strength(self, field_a: NDAnalogField, field_b: NDAnalogField) -> float:
        """Get overall resonance strength between fields"""
        resonance_map = self._compute_resonance(field_a, field_b)
        return np.mean(resonance_map)
    
    def find_resonant_regions(self, field_a: NDAnalogField, field_b: NDAnalogField, 
                             min_size: int = 4) -> list:
        """Find connected regions of high resonance"""
        resonance_map = self._compute_resonance(field_a, field_b)
        high_resonance = resonance_map > self.threshold
        
        # Simple connected components (for 2D)
        if len(field_a.shape) == 2:
            regions = []
            visited = np.zeros_like(high_resonance, dtype=bool)
            
            for i in range(high_resonance.shape[0]):
                for j in range(high_resonance.shape[1]):
                    if high_resonance[i, j] and not visited[i, j]:
                        region = self._flood_fill(high_resonance, visited, i, j)
                        if len(region) >= min_size:
                            regions.append(region)
            
            return regions
        
        return []
    
    def _flood_fill(self, mask, visited, i, j) -> list:
        """Simple flood fill for connected regions"""
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return []
        if visited[i, j] or not mask[i, j]:
            return []
        
        visited[i, j] = True
        region = [(i, j)]
        
        # Check 4-connected neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            region.extend(self._flood_fill(mask, visited, i + di, j + dj))
        
        return region
    
    def __repr__(self):
        return f"ResonatorAtom(amp={self.amplification:.1f}, mode='{self.mode}')"
