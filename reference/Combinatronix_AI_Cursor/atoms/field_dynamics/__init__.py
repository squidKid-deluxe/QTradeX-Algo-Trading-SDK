# combinatronix/atoms/field_dynamics/__init__.py
"""
Field Dynamics - Spatial Operation Atoms

These 5 atoms manipulate spatial relationships and movement in fields:

1. Vortex - Circular flow/rotation (recursion, cycles)
2. Attractor - Pull toward point (goals, magnetism)
3. Barrier - Block propagation (boundaries, constraints)
4. Bridge - Connect distant regions (analogy, metaphor)
5. Void - Absorb activation (forgetting, death)

Each operates on field topology and spatial structure.
"""

from .vortex import VortexAtom
from .attractor import AttractorAtom
from .barrier import BarrierAtom
from .bridge import BridgeAtom
from .void import VoidAtom

__all__ = ['VortexAtom', 'AttractorAtom', 'BarrierAtom', 'BridgeAtom', 'VoidAtom']




