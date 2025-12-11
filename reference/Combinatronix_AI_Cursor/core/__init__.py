# combinatronix/core/__init__.py
"""
Combinatronix Core Library

The foundational layer for all Combinatronix AI systems.
Includes: Combinator VM, Field Engine, Kernel Algebra
"""

from .combinator_vm import (
    Comb, Val, App, Thunk,
    app, reduce_whnf, reduce_step,
    show, serialize, deserialize
)

from .field_engine import (
    NDAnalogField,
    PropagationMode,
    FieldConfig
)

from .kernel_algebra import (
    CognitiveKernel,
    KernelLibrary
)

__version__ = "0.1.0"
__all__ = [
    # VM
    'Comb', 'Val', 'App', 'Thunk', 'app',
    'reduce_whnf', 'reduce_step', 'show',
    'serialize', 'deserialize',
    # Field Engine
    'NDAnalogField', 'PropagationMode', 'FieldConfig',
    # Kernels
    'CognitiveKernel', 'KernelLibrary'
]

