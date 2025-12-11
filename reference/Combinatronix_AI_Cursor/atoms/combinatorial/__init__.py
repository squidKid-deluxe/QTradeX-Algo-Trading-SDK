# combinatronix/atoms/combinatorial/__init__.py
"""
Combinatorial Atoms - Pure Logic Operations

These 5 atoms are the fundamental combinators (S, K, I, B, C) wrapped
as field operations. They represent pure logical/cognitive operations:

1. Witness - I combinator (identity/observation)
2. Selector - K combinator (choice/constant)
3. Weaver - S combinator (synthesis/application)
4. Composer - B combinator (composition/sequence)
5. Swapper - C combinator (perspective shift/flip)

Each operates on fields while preserving combinator semantics.
"""

from .witness import WitnessAtom
from .selector import SelectorAtom
from .weaver import WeaverAtom
from .composer import ComposerAtom
from .swapper import SwapperAtom

__all__ = ['WitnessAtom', 'SelectorAtom', 'WeaverAtom', 'ComposerAtom', 'SwapperAtom']
