# combinatronix/atoms/pattern_primitives/__init__.py
"""
Pattern Primitives - The Most Fundamental Atoms

These 5 atoms represent the basic patterns that all other operations build upon:
1. Pulse - Oscillating activation (time itself)
2. Seed - Point source that spreads (origin)
3. Echo - Decaying repetition (memory)
4. Mirror - Reflection/symmetry (duality)
5. Gradient - Flow from high to low (attraction)

Each atom is 10-50 lines and has zero dependencies on other atoms.
"""

from .pulse import PulseAtom
from .seed import SeedAtom
from .echo import EchoAtom
from .mirror import MirrorAtom
from .gradient import GradientAtom

__all__ = ['PulseAtom', 'SeedAtom', 'EchoAtom', 'MirrorAtom', 'GradientAtom']

