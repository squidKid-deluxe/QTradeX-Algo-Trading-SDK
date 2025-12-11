"""
Integration Organisms

Organisms for cross-subsystem integration and correlation analysis, built from atoms and molecules.
These organisms handle field correlation, system coordination, multi-modal integration,
and complete cognitive architecture using the Combinatronix molecular architecture.
"""

from .field_correlator import FieldCorrelator, SubsystemType, CategoryDimension, SubsystemField
from .cognitive_architecture import CognitiveArchitecture, CognitiveState, ActionPlan, ThoughtProcess

__all__ = [
    'FieldCorrelator',
    'SubsystemType',
    'CategoryDimension', 
    'SubsystemField',
    'CognitiveArchitecture',
    'CognitiveState',
    'ActionPlan',
    'ThoughtProcess'
]
