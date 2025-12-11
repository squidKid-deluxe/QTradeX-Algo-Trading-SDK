# 🧠 Combinatronix AI

Revolutionary AI framework using combinators and field dynamics instead of neural networks.

## Quick Start

```python
from combinatronix.atoms.pattern_primitives import pulse
from combinatronix.core import NDAnalogField

field = NDAnalogField((8, 8))
pulse_atom = pulse.PulseAtom(frequency=1.0)
pulse_atom.apply(field)
```

## Philosophy

- **No backpropagation** - uses field dynamics and combinatorial reasoning
- **Interpretable** - every operation is understandable
- **Compositional** - build complex intelligence from 30 atomic operations
- **Zero-shot** - works immediately without training

## Structure

- `atoms/` - 30 atomic operations (the periodic table)
- `molecules/` - Intermediate programs (2-5 atoms combined)
- `organisms/` - Complete systems (full AI capabilities)
- `core/` - VM and field engine
- `examples/` - Tutorials and demos

## Installation

```bash
pip install -e .
```

## License

MIT License - Build the future of AI!
