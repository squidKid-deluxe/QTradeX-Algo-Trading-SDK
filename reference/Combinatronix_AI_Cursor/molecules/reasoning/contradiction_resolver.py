# ============================================================================
# ContradictionResolver - Resolve Logical Contradictions
# ============================================================================

"""
ContradictionResolver - Resolve contradictions using balancing and composition

Composition: Balancer + Composer
Category: Reasoning
Complexity: Molecule (50-200 lines)

Resolves logical contradictions by finding equilibrium between opposing
positions and composing them into coherent solutions. This enables
dialectical thinking, conflict resolution, and synthesis of opposing ideas.

Example:
    >>> resolver = ContradictionResolver(equilibrium_rate=0.3, composition_strength=0.8)
    >>> resolution = resolver.resolve_contradiction(claim_a, claim_b)
    >>> synthesis = resolver.create_synthesis(opposing_views)
    >>> contradictions = resolver.detect_contradictions(field_a, field_b)
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.tension_resolvers import BalancerAtom
    from ...atoms.combinatorial import ComposerAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.tension_resolvers import BalancerAtom
    from combinatronix.atoms.combinatorial import ComposerAtom
    from combinatronix.core import NDAnalogField


class ContradictionResolver:
    """Resolve contradictions using balancing and composition"""
    
    def __init__(self, equilibrium_rate: float = 0.3, composition_strength: float = 0.8,
                 min_tension: float = 0.1, resolution_threshold: float = 0.5):
        """
        Args:
            equilibrium_rate: How fast to move toward equilibrium
            composition_strength: Strength of composition operations
            min_tension: Minimum tension to trigger resolution
            resolution_threshold: Threshold for considering contradiction resolved
        """
        self.balancer = BalancerAtom(
            equilibrium_rate=equilibrium_rate,
            min_tension=min_tension
        )
        self.composer = ComposerAtom(preserve_intermediate=True)
        
        # Resolution state
        self.resolution_threshold = resolution_threshold
        self.contradictions = []  # List of contradiction records
        self.resolutions = []  # List of resolution records
        self.syntheses = {}  # synthesis_id -> synthesis_data
        self.resolution_count = 0
        self.total_processed = 0
        self.tension_history = []
    
    def resolve_contradiction(self, claim_a: NDAnalogField, claim_b: NDAnalogField,
                            context: str = None, priority: float = 1.0) -> dict:
        """Resolve contradiction between two claims
        
        Args:
            claim_a: First claim/position
            claim_b: Second claim/position
            context: Context for the contradiction
            priority: Priority of resolution (0.0-1.0)
            
        Returns:
            Dictionary with resolution information
        """
        self.total_processed += 1
        
        # Detect contradiction
        contradiction_points = self.balancer.detect_contradictions(claim_a, claim_b)
        tension_level = self.balancer.get_tension_level()
        
        if len(contradiction_points) == 0:
            return {
                'resolved': True,
                'tension_level': tension_level,
                'resolution': 'No contradiction detected'
            }
        
        # Record contradiction
        contradiction_id = f"contradiction_{len(self.contradictions) + 1}"
        contradiction_record = {
            'id': contradiction_id,
            'claim_a': claim_a.activation.copy(),
            'claim_b': claim_b.activation.copy(),
            'contradiction_points': contradiction_points,
            'tension_level': tension_level,
            'context': context,
            'priority': priority,
            'created_time': self.total_processed
        }
        self.contradictions.append(contradiction_record)
        
        # Apply balancing
        balanced_a = type('Field', (), {
            'activation': claim_a.activation.copy(),
            'shape': claim_a.shape
        })()
        balanced_b = type('Field', (), {
            'activation': claim_b.activation.copy(),
            'shape': claim_b.shape
        })()
        
        self.balancer.apply(balanced_a, balanced_b)
        
        # Compose resolution
        resolution_field = self._compose_resolution(balanced_a, balanced_b, priority)
        
        # Check if resolved
        new_tension = self._compute_tension(balanced_a, balanced_b)
        is_resolved = new_tension < self.resolution_threshold
        
        # Record resolution
        resolution_id = f"resolution_{self.resolution_count + 1}"
        resolution_record = {
            'id': resolution_id,
            'contradiction_id': contradiction_id,
            'resolution_field': resolution_field.activation.copy(),
            'tension_before': tension_level,
            'tension_after': new_tension,
            'is_resolved': is_resolved,
            'priority': priority,
            'created_time': self.total_processed
        }
        self.resolutions.append(resolution_record)
        self.resolution_count += 1
        
        # Update tension history
        self.tension_history.append(new_tension)
        
        return resolution_record
    
    def _compose_resolution(self, field_a: NDAnalogField, field_b: NDAnalogField, 
                          priority: float) -> NDAnalogField:
        """Compose resolution from balanced fields"""
        # Create resolution field
        resolution_field = NDAnalogField(field_a.shape)
        
        # Define composition operations
        operations = [
            lambda f: self._weighted_average(f, field_a, field_b, priority),
            lambda f: self._synthesize_opposites(f, field_a, field_b),
            lambda f: self._find_common_ground(f, field_a, field_b)
        ]
        
        # Apply composition
        self.composer.apply(resolution_field, operations)
        
        return resolution_field
    
    def _weighted_average(self, field: NDAnalogField, field_a: NDAnalogField, 
                         field_b: NDAnalogField, priority: float) -> NDAnalogField:
        """Create weighted average of two fields"""
        # Weight based on priority and field strength
        weight_a = priority * np.sum(np.abs(field_a.activation))
        weight_b = (1 - priority) * np.sum(np.abs(field_b.activation))
        
        total_weight = weight_a + weight_b
        if total_weight > 0:
            weight_a /= total_weight
            weight_b /= total_weight
        
        field.activation = (field_a.activation * weight_a + 
                           field_b.activation * weight_b)
        
        return field
    
    def _synthesize_opposites(self, field: NDAnalogField, field_a: NDAnalogField, 
                             field_b: NDAnalogField) -> NDAnalogField:
        """Synthesize opposing fields into higher-level concept"""
        # Find regions where both fields are active
        both_active = np.logical_and(field_a.activation > 0.1, field_b.activation > 0.1)
        
        # In active regions, create synthesis
        synthesis = np.zeros_like(field.activation)
        synthesis[both_active] = (field_a.activation[both_active] + 
                                field_b.activation[both_active]) / 2
        
        # In single-active regions, preserve the stronger field
        only_a = np.logical_and(field_a.activation > 0.1, field_b.activation <= 0.1)
        only_b = np.logical_and(field_b.activation > 0.1, field_a.activation <= 0.1)
        
        synthesis[only_a] = field_a.activation[only_a]
        synthesis[only_b] = field_b.activation[only_b]
        
        field.activation = synthesis
        return field
    
    def _find_common_ground(self, field: NDAnalogField, field_a: NDAnalogField, 
                           field_b: NDAnalogField) -> NDAnalogField:
        """Find common ground between fields"""
        # Find regions where both fields have similar values
        difference = np.abs(field_a.activation - field_b.activation)
        common_mask = difference < np.mean(difference) * 0.5
        
        # Use average in common regions
        common_ground = np.zeros_like(field.activation)
        common_ground[common_mask] = (field_a.activation[common_mask] + 
                                    field_b.activation[common_mask]) / 2
        
        field.activation = common_ground
        return field
    
    def _compute_tension(self, field_a: NDAnalogField, field_b: NDAnalogField) -> float:
        """Compute tension between two fields"""
        tension = np.abs(field_a.activation - field_b.activation)
        return np.mean(tension)
    
    def create_synthesis(self, opposing_views: list, synthesis_name: str = None) -> str:
        """Create synthesis from multiple opposing views
        
        Args:
            opposing_views: List of opposing view fields
            synthesis_name: Name for the synthesis
            
        Returns:
            Synthesis ID
        """
        if len(opposing_views) < 2:
            return None
        
        if synthesis_name is None:
            synthesis_name = f"synthesis_{len(self.syntheses) + 1}"
        
        synthesis_id = f"synthesis_{id(opposing_views)}"
        
        # Create synthesis field
        synthesis_field = NDAnalogField(opposing_views[0].shape)
        
        # Apply multiple resolution steps
        current_field = opposing_views[0]
        for i in range(1, len(opposing_views)):
            resolution = self.resolve_contradiction(current_field, opposing_views[i])
            if resolution.get('is_resolved', False):
                current_field = type('Field', (), {
                    'activation': resolution['resolution_field'],
                    'shape': synthesis_field.shape
                })()
        
        synthesis_field.activation = current_field.activation
        
        # Store synthesis
        synthesis_data = {
            'id': synthesis_id,
            'name': synthesis_name,
            'synthesis_field': synthesis_field.activation.copy(),
            'input_views': [view.activation.copy() for view in opposing_views],
            'created_time': self.total_processed,
            'usage_count': 0
        }
        
        self.syntheses[synthesis_id] = synthesis_data
        
        return synthesis_id
    
    def detect_contradictions(self, field_a: NDAnalogField, field_b: NDAnalogField) -> list:
        """Detect contradictions between two fields
        
        Args:
            field_a: First field
            field_b: Second field
            
        Returns:
            List of contradiction locations
        """
        return self.balancer.detect_contradictions(field_a, field_b)
    
    def get_contradiction_strength(self, field_a: NDAnalogField, field_b: NDAnalogField) -> float:
        """Get strength of contradiction between fields
        
        Args:
            field_a: First field
            field_b: Second field
            
        Returns:
            Contradiction strength (0.0-1.0)
        """
        tension = np.abs(field_a.activation - field_b.activation)
        return np.mean(tension)
    
    def get_resolution_statistics(self) -> dict:
        """Get statistics about contradiction resolution"""
        if not self.resolutions:
            return {
                'total_contradictions': 0,
                'total_resolutions': 0,
                'resolution_rate': 0.0,
                'average_tension_reduction': 0.0
            }
        
        resolved_count = sum(1 for r in self.resolutions if r['is_resolved'])
        resolution_rate = resolved_count / len(self.resolutions)
        
        tension_reductions = [r['tension_before'] - r['tension_after'] 
                            for r in self.resolutions]
        avg_tension_reduction = np.mean(tension_reductions) if tension_reductions else 0.0
        
        return {
            'total_contradictions': len(self.contradictions),
            'total_resolutions': len(self.resolutions),
            'resolution_rate': resolution_rate,
            'average_tension_reduction': avg_tension_reduction,
            'current_tension_level': self.tension_history[-1] if self.tension_history else 0.0
        }
    
    def get_synthesis_by_name(self, name: str) -> dict:
        """Get synthesis by name
        
        Args:
            name: Name of synthesis
            
        Returns:
            Synthesis data or None if not found
        """
        for synthesis in self.syntheses.values():
            if synthesis['name'] == name:
                return synthesis
        return None
    
    def apply_synthesis(self, synthesis_id: str, target_field: NDAnalogField, 
                       strength: float = 1.0) -> bool:
        """Apply existing synthesis to target field
        
        Args:
            synthesis_id: ID of synthesis to apply
            target_field: Field to apply synthesis to
            strength: Application strength
            
        Returns:
            True if synthesis applied successfully
        """
        if synthesis_id not in self.syntheses:
            return False
        
        synthesis = self.syntheses[synthesis_id]
        synthesis['usage_count'] += 1
        
        # Apply synthesis
        target_field.activation = (target_field.activation * (1 - strength) + 
                                 synthesis['synthesis_field'] * strength)
        
        return True
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'contradictions': [{
                'id': c['id'],
                'tension_level': c['tension_level'],
                'priority': c['priority'],
                'context': c['context']
            } for c in self.contradictions],
            'resolutions': [{
                'id': r['id'],
                'is_resolved': r['is_resolved'],
                'tension_reduction': r['tension_before'] - r['tension_after']
            } for r in self.resolutions],
            'syntheses': [{
                'id': s['id'],
                'name': s['name'],
                'usage_count': s['usage_count']
            } for s in self.syntheses.values()],
            'resolution_count': self.resolution_count,
            'total_processed': self.total_processed
        }
    
    def reset(self):
        """Reset contradiction resolver"""
        self.contradictions.clear()
        self.resolutions.clear()
        self.syntheses.clear()
        self.resolution_count = 0
        self.total_processed = 0
        self.tension_history.clear()
        self.balancer.tension_history.clear()
        self.composer.intermediate_results.clear()
    
    def __repr__(self):
        return f"ContradictionResolver(contradictions={len(self.contradictions)}, resolutions={len(self.resolutions)})"