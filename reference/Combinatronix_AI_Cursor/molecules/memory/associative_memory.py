# ============================================================================
# AssociativeMemory - Create and Retrieve Associative Links
# ============================================================================

"""
AssociativeMemory - Create and retrieve associative links between patterns

Composition: Binder + Resonator
Category: Memory
Complexity: Molecule (50-200 lines)

Creates and manages associative links between memories using binding and
resonance. This enables pattern completion, associative retrieval, and
the formation of complex memory networks through resonance-based activation.

Example:
    >>> am = AssociativeMemory(binding_strength=0.8, resonance_threshold=0.6)
    >>> am.associate(pattern_a, pattern_b, strength=0.9)
    >>> retrieved = am.retrieve_by_resonance(query_pattern)
    >>> network = am.get_association_network()
"""

import numpy as np
from collections import defaultdict
try:
    from ...atoms.multi_field import BinderAtom, ResonatorAtom
    from ...core import NDAnalogField
except ImportError:
    from combinatronix.atoms.multi_field import BinderAtom, ResonatorAtom
    from combinatronix.core import NDAnalogField


class AssociativeMemory:
    """Associative memory using binding and resonance for pattern completion"""
    
    def __init__(self, binding_strength: float = 0.8, resonance_threshold: float = 0.6,
                 amplification: float = 2.0, decay_rate: float = 0.99):
        """
        Args:
            binding_strength: Strength of new associations
            resonance_threshold: Minimum resonance for activation
            amplification: How much to amplify resonant patterns
            decay_rate: How fast associations decay
        """
        self.binder = BinderAtom(binding_strength=binding_strength, decay_rate=decay_rate)
        self.resonator = ResonatorAtom(
            amplification=amplification,
            threshold=resonance_threshold,
            mode='correlation'
        )
        
        # Memory network
        self.associations = {}  # pattern_id -> association_data
        self.patterns = {}  # pattern_id -> pattern_data
        self.pattern_counter = 0
        self.association_counter = 0
        self.activation_history = []
    
    def store_pattern(self, pattern: np.ndarray, name: str = None, 
                     metadata: dict = None) -> str:
        """Store a pattern in associative memory
        
        Args:
            pattern: Pattern to store
            name: Optional name for the pattern
            metadata: Additional metadata
            
        Returns:
            Pattern ID
        """
        self.pattern_counter += 1
        pattern_id = f"pattern_{self.pattern_counter}"
        
        self.patterns[pattern_id] = {
            'id': pattern_id,
            'pattern': pattern.copy(),
            'shape': pattern.shape,
            'name': name or f"Pattern_{self.pattern_counter}",
            'metadata': metadata or {},
            'creation_time': self.pattern_counter,
            'activation_count': 0,
            'last_activated': 0
        }
        
        return pattern_id
    
    def associate(self, pattern_id_a: str, pattern_id_b: str, 
                 strength: float = None, bidirectional: bool = True) -> bool:
        """Create association between two patterns
        
        Args:
            pattern_id_a: First pattern ID
            pattern_id_b: Second pattern ID
            strength: Association strength (uses default if None)
            bidirectional: Whether to create bidirectional association
            
        Returns:
            True if association created successfully
        """
        if pattern_id_a not in self.patterns or pattern_id_b not in self.patterns:
            return False
        
        self.association_counter += 1
        association_id = f"assoc_{self.association_counter}"
        
        # Create association data
        association_data = {
            'id': association_id,
            'pattern_a': pattern_id_a,
            'pattern_b': pattern_id_b,
            'strength': strength or self.binder.binding_strength,
            'creation_time': self.association_counter,
            'activation_count': 0,
            'bidirectional': bidirectional
        }
        
        # Store association
        self.associations[association_id] = association_data
        
        # Update pattern metadata
        if 'associations' not in self.patterns[pattern_id_a]['metadata']:
            self.patterns[pattern_id_a]['metadata']['associations'] = []
        self.patterns[pattern_id_a]['metadata']['associations'].append(association_id)
        
        if bidirectional:
            if 'associations' not in self.patterns[pattern_id_b]['metadata']:
                self.patterns[pattern_id_b]['metadata']['associations'] = []
            self.patterns[pattern_id_b]['metadata']['associations'].append(association_id)
        
        return True
    
    def retrieve_by_resonance(self, query_pattern: np.ndarray, 
                             max_results: int = 5) -> list:
        """Retrieve patterns by resonance with query
        
        Args:
            query_pattern: Pattern to match against
            max_results: Maximum number of results
            
        Returns:
            List of (pattern_id, field, resonance_strength) tuples
        """
        # Create query field
        query_field = type('Field', (), {
            'activation': query_pattern,
            'shape': query_pattern.shape
        })()
        
        # Test resonance with all stored patterns
        resonance_scores = []
        
        for pattern_id, pattern_data in self.patterns.items():
            # Create field from stored pattern
            stored_field = type('Field', (), {
                'activation': pattern_data['pattern'],
                'shape': pattern_data['shape']
            })()
            
            # Compute resonance
            resonance_strength = self.resonator.get_resonance_strength(
                query_field, stored_field
            )
            
            if resonance_strength >= self.resonator.threshold:
                resonance_scores.append((resonance_strength, pattern_id, stored_field))
        
        # Sort by resonance strength
        resonance_scores.sort(reverse=True)
        
        # Return top results
        results = []
        for resonance_strength, pattern_id, field in resonance_scores[:max_results]:
            # Update activation count
            self.patterns[pattern_id]['activation_count'] += 1
            self.patterns[pattern_id]['last_activated'] = self.pattern_counter
            
            results.append((pattern_id, field, resonance_strength))
        
        # Record activation
        self.activation_history.append({
            'time': self.pattern_counter,
            'query_shape': query_pattern.shape,
            'results_count': len(results),
            'max_resonance': max([r[2] for r in results]) if results else 0
        })
        
        return results
    
    def pattern_completion(self, partial_pattern: np.ndarray, 
                          completion_strength: float = 0.8) -> NDAnalogField:
        """Complete a partial pattern using associations
        
        Args:
            partial_pattern: Incomplete pattern
            completion_strength: How strongly to apply completion
            
        Returns:
            Completed pattern field
        """
        # Find most resonant stored pattern
        results = self.retrieve_by_resonance(partial_pattern, max_results=1)
        
        if not results:
            # No match found, return original
            field = NDAnalogField(partial_pattern.shape)
            field.activation = partial_pattern
            return field
        
        pattern_id, stored_field, resonance_strength = results[0]
        
        # Create completion field
        completion_field = NDAnalogField(partial_pattern.shape)
        completion_field.activation = partial_pattern.copy()
        
        # Apply resonance-based completion
        if stored_field.activation.shape == partial_pattern.shape:
            # Direct completion
            completion_field.activation += stored_field.activation * completion_strength
        else:
            # Resize and complete
            # Simple resizing (could be improved)
            if len(stored_field.activation.shape) == len(partial_pattern.shape):
                min_shape = tuple(min(s1, s2) for s1, s2 in 
                                zip(stored_field.activation.shape, partial_pattern.shape))
                resized = stored_field.activation[:min_shape[0], :min_shape[1]]
                completion_field.activation[:min_shape[0], :min_shape[1]] += resized * completion_strength
        
        # Apply resonator amplification
        self.resonator.apply(completion_field, stored_field)
        
        return completion_field
    
    def get_association_network(self) -> dict:
        """Get the complete association network
        
        Returns:
            Dictionary representing the association network
        """
        network = {
            'patterns': {},
            'associations': {},
            'statistics': self.get_network_statistics()
        }
        
        # Add pattern information
        for pattern_id, pattern_data in self.patterns.items():
            network['patterns'][pattern_id] = {
                'name': pattern_data['name'],
                'shape': pattern_data['shape'],
                'activation_count': pattern_data['activation_count'],
                'associations': pattern_data['metadata'].get('associations', [])
            }
        
        # Add association information
        for assoc_id, assoc_data in self.associations.items():
            network['associations'][assoc_id] = {
                'pattern_a': assoc_data['pattern_a'],
                'pattern_b': assoc_data['pattern_b'],
                'strength': assoc_data['strength'],
                'activation_count': assoc_data['activation_count'],
                'bidirectional': assoc_data['bidirectional']
            }
        
        return network
    
    def get_network_statistics(self) -> dict:
        """Get statistics about the association network"""
        if not self.patterns:
            return {
                'pattern_count': 0,
                'association_count': 0,
                'average_activation': 0.0,
                'network_density': 0.0
            }
        
        activation_counts = [p['activation_count'] for p in self.patterns.values()]
        association_counts = [len(p['metadata'].get('associations', [])) 
                            for p in self.patterns.values()]
        
        # Calculate network density
        max_possible_associations = len(self.patterns) * (len(self.patterns) - 1) // 2
        actual_associations = len(self.associations)
        network_density = actual_associations / max_possible_associations if max_possible_associations > 0 else 0
        
        return {
            'pattern_count': len(self.patterns),
            'association_count': len(self.associations),
            'average_activation': np.mean(activation_counts),
            'average_associations_per_pattern': np.mean(association_counts),
            'network_density': network_density,
            'resonance_threshold': self.resonator.threshold,
            'binding_strength': self.binder.binding_strength
        }
    
    def find_strongly_connected_patterns(self, min_connections: int = 3) -> list:
        """Find patterns with many associations
        
        Args:
            min_connections: Minimum number of associations
            
        Returns:
            List of pattern IDs with many connections
        """
        strongly_connected = []
        
        for pattern_id, pattern_data in self.patterns.items():
            association_count = len(pattern_data['metadata'].get('associations', []))
            if association_count >= min_connections:
                strongly_connected.append({
                    'pattern_id': pattern_id,
                    'name': pattern_data['name'],
                    'connection_count': association_count,
                    'activation_count': pattern_data['activation_count']
                })
        
        # Sort by connection count
        strongly_connected.sort(key=lambda x: x['connection_count'], reverse=True)
        return strongly_connected
    
    def decay_associations(self):
        """Apply decay to all associations"""
        self.binder.decay_bindings()
        
        # Update association strengths
        for assoc_data in self.associations.values():
            assoc_data['strength'] *= self.binder.decay_rate
    
    def remove_weak_associations(self, strength_threshold: float = 0.1) -> int:
        """Remove associations below strength threshold
        
        Args:
            strength_threshold: Minimum strength to keep
            
        Returns:
            Number of associations removed
        """
        associations_to_remove = []
        
        for assoc_id, assoc_data in self.associations.items():
            if assoc_data['strength'] < strength_threshold:
                associations_to_remove.append(assoc_id)
        
        # Remove weak associations
        for assoc_id in associations_to_remove:
            self._remove_association(assoc_id)
        
        return len(associations_to_remove)
    
    def _remove_association(self, assoc_id: str):
        """Remove association and update pattern metadata"""
        if assoc_id not in self.associations:
            return
        
        assoc_data = self.associations[assoc_id]
        
        # Remove from pattern metadata
        for pattern_id in [assoc_data['pattern_a'], assoc_data['pattern_b']]:
            if pattern_id in self.patterns:
                associations = self.patterns[pattern_id]['metadata'].get('associations', [])
                if assoc_id in associations:
                    associations.remove(assoc_id)
        
        # Remove association
        del self.associations[assoc_id]
    
    def get_state(self) -> dict:
        """Return internal state for inspection"""
        return {
            'patterns': {pid: {
                'name': p['name'],
                'shape': p['shape'],
                'activation_count': p['activation_count']
            } for pid, p in self.patterns.items()},
            'associations': {aid: {
                'pattern_a': a['pattern_a'],
                'pattern_b': a['pattern_b'],
                'strength': a['strength']
            } for aid, a in self.associations.items()},
            'pattern_counter': self.pattern_counter,
            'association_counter': self.association_counter,
            'network_statistics': self.get_network_statistics()
        }
    
    def reset(self):
        """Reset associative memory"""
        self.patterns.clear()
        self.associations.clear()
        self.pattern_counter = 0
        self.association_counter = 0
        self.activation_history.clear()
        self.binder.clear_bindings()
        self.resonator.resonance_history.clear()
    
    def __repr__(self):
        return f"AssociativeMemory(patterns={len(self.patterns)}, associations={len(self.associations)})"