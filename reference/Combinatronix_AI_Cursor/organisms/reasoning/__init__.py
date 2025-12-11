"""
Reasoning Organisms

Organisms for advanced reasoning and symbol invention, built from atoms and molecules.
These organisms handle cognitive tension detection, symbol invention, complex
reasoning, cross-domain analogy making, semantic network construction, and
self-modeling using the Combinatronix molecular architecture.
"""

from .reasoning_engine import ReasoningEngine, ConceptTension, InventedSymbol
from .analogy_maker import AnalogyMaker, AnalogyMapping, DomainPattern
from .semantic_network import SemanticNetwork, ConceptNode, Relation, ActivationPath
from .self_model import SelfModel, SelfSnapshot, CapabilityAssessment, SelfImprovementGoal

__all__ = [
    'ReasoningEngine',
    'ConceptTension',
    'InventedSymbol',
    'AnalogyMaker',
    'AnalogyMapping',
    'DomainPattern',
    'SemanticNetwork',
    'ConceptNode',
    'Relation',
    'ActivationPath',
    'SelfModel',
    'SelfSnapshot',
    'CapabilityAssessment',
    'SelfImprovementGoal'
]
