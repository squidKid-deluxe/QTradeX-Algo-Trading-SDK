# 🧬 COMBINATRONIX MOLECULES - BUILD PLAN

## Overview
Molecules are 2-5 atoms composed together to create intermediate-complexity cognitive operations.
Each molecule should be 50-200 lines and solve a specific cognitive task.

---

## 📁 FOLDER STRUCTURE
```
combinatronix/molecules/
├── __init__.py
├── perception/
│   ├── __init__.py
│   ├── edge_detector.py          # Gradient + Threshold
│   ├── motion_detector.py        # MemoryTrace + Comparator
│   ├── pattern_recognizer.py    # Resonator + Binder
│   └── object_tracker.py         # Seed + Gradient + MemoryTrace
├── memory/
│   ├── __init__.py
│   ├── short_term_memory.py     # Echo + Decay
│   ├── long_term_memory.py      # MemoryTrace + Threshold + Binder
│   ├── associative_memory.py    # Binder + Resonator
│   └── working_memory.py        # Echo + Attractor + Damper
├── reasoning/
│   ├── __init__.py
│   ├── contradiction_resolver.py # Balancer + Composer
│   ├── gap_filler.py             # Filler + Anticipator
│   ├── analogizer.py             # Translator + Comparator + Bridge
│   └── pattern_completer.py     # Anticipator + Resonator
├── attention/
│   ├── __init__.py
│   ├── focus.py                  # Attractor + Damper
│   ├── saliency.py              # Amplifier + Threshold
│   ├── novelty_detector.py      # Comparator + MemoryTrace
│   └── attention_shift.py       # Gradient + Vortex
└── emotion/
    ├── __init__.py
    ├── arousal.py               # Amplifier + Rhythm
    ├── valence.py               # Attractor + Repeller
    ├── motivation.py            # Gradient + Anticipator
    └── homeostasis.py           # Balancer + Damper
```

---

## 🔧 MOLECULE TEMPLATE
```python
# combinatronix/molecules/[category]/[name].py
"""
[Name] - [One-line description]

Composition: [Atom1] + [Atom2] + [Atom3]
Category: [Category]
Complexity: Molecule (50-200 lines)

[Detailed description of what this molecule does and why]

Example:
    >>> molecule = [Name]()
    >>> result = molecule.process(field)
"""

from combinatronix.atoms.pattern_primitives import [atoms]
from combinatronix.atoms.combinatorial import [atoms]
from combinatronix.atoms.field_dynamics import [atoms]
from combinatronix.atoms.temporal import [atoms]
from combinatronix.core import NDAnalogField

class [Name]Molecule:
    """[Description]"""
    
    def __init__(self, **config):
        # Initialize constituent atoms
        self.atom1 = Atom1(**config1)
        self.atom2 = Atom2(**config2)
        self.atom3 = Atom3(**config3)
        
        # Molecule-specific state
        self.state = {}
    
    def process(self, field: NDAnalogField, **kwargs):
        """Main processing method
        
        Args:
            field: Input field
            **kwargs: Additional parameters
            
        Returns:
            Processed field or result
        """
        # Compose atoms into functional unit
        result = self._apply_pipeline(field, **kwargs)
        return result
    
    def _apply_pipeline(self, field, **kwargs):
        # Step-by-step application of atoms
        self.atom1.apply(field)
        self.atom2.apply(field)
        self.atom3.apply(field)
        return field
    
    def get_state(self):
        """Return internal state for inspection"""
        return self.state.copy()
    
    def reset(self):
        """Reset molecule state"""
        self.state.clear()
```

---

## 📋 PRIORITY BUILD ORDER

### Week 1: Perception (Foundation)
1. **edge_detector.py** - Gradient + Threshold
2. **motion_detector.py** - MemoryTrace + Comparator  
3. **pattern_recognizer.py** - Resonator + Binder
4. **object_tracker.py** - Seed + Gradient + MemoryTrace

### Week 2: Memory (Storage)
5. **short_term_memory.py** - Echo + Decay
6. **long_term_memory.py** - MemoryTrace + Threshold
7. **associative_memory.py** - Binder + Resonator
8. **working_memory.py** - Echo + Attractor + Damper

### Week 3: Attention (Selection)
9. **focus.py** - Attractor + Damper
10. **saliency.py** - Amplifier + Threshold
11. **novelty_detector.py** - Comparator + MemoryTrace
12. **attention_shift.py** - Gradient + Vortex

### Week 4: Reasoning & Emotion
13. **analogizer.py** - Bridge + Comparator
14. **pattern_completer.py** - Anticipator + Resonator
15. **arousal.py** - Amplifier + Rhythm
16. **motivation.py** - Gradient + Anticipator

---

## 🎯 DETAILED SPECIFICATIONS

### 1. EdgeDetector (Perception)
**Atoms:** Gradient + Threshold
**Purpose:** Detect edges/boundaries in visual fields
```python
class EdgeDetector:
    def __init__(self, threshold=0.3, strength=1.0):
        self.gradient = GradientAtom(strength=strength)
        self.threshold = ThresholdAtom(threshold=threshold, mode='rectify')
    
    def process(self, field):
        # Compute gradient magnitude
        grad_field = self.gradient.get_gradient_field(field)
        field.activation = grad_field
        
        # Threshold to get clean edges
        self.threshold.apply(field)
        return field
```

### 2. MotionDetector (Perception)
**Atoms:** MemoryTrace + Comparator (from multi_field)
**Purpose:** Detect movement by comparing current to past
```python
class MotionDetector:
    def __init__(self, sensitivity=0.2):
        self.memory = MemoryTraceAtom(accumulation_rate=0.5, decay_rate=0.9)
        self.threshold = ThresholdAtom(threshold=sensitivity)
        
    def process(self, field):
        # Compare current to memory
        if self.memory.trace is not None:
            motion = np.abs(field.activation - self.memory.trace)
            field.activation = motion
        
        self.memory.apply(field)
        self.threshold.apply(field)
        return field
```

### 3. ShortTermMemory (Memory)
**Atoms:** Echo + Decay
**Purpose:** Hold recent information with natural forgetting
```python
class ShortTermMemory:
    def __init__(self, capacity=5, decay_rate=0.9):
        self.echo = EchoAtom(decay_rate=decay_rate, depth=capacity)
        self.decay = DecayAtom(decay_rate=decay_rate)
        
    def store(self, field):
        self.echo.apply(field)
        return field
    
    def recall(self, field):
        # Retrieve echo pattern
        if self.echo.history:
            field.activation = self.echo.history[-1].copy()
        self.decay.apply(field)
        return field
```

### 4. Focus (Attention)
**Atoms:** Attractor + Damper (Amplifier inverse)
**Purpose:** Focus attention on specific region
```python
class Focus:
    def __init__(self, location=None, strength=0.3):
        self.attractor = AttractorAtom(location=location, strength=strength)
        self.damper = DecayAtom(decay_rate=0.8, selective=True)
        
    def apply(self, field, focus_location=None):
        if focus_location:
            self.attractor.set_location(focus_location)
        
        # Pull toward focus
        self.attractor.apply(field)
        
        # Dampen background
        self.damper.apply(field)
        return field
```

### 5. Analogizer (Reasoning)
**Atoms:** Bridge + Comparator + Translator
**Purpose:** Find analogies between patterns
```python
class Analogizer:
    def __init__(self):
        self.bridge = BridgeAtom(bidirectional=True)
        self.comparator = ComparatorAtom()  # from multi_field
        
    def find_analogy(self, field_a, field_b, threshold=0.6):
        # Compare fields
        similarity = self.comparator.apply(field_a, field_b)
        
        if similarity > threshold:
            # Create bridge between analogous regions
            self.bridge.connect(field_a, region_a, region_b, strength=similarity)
        
        return similarity
```

---

## 🧪 TESTING STRATEGY

### Unit Tests (per molecule)
```python
def test_edge_detector():
    detector = EdgeDetector(threshold=0.3)
    field = create_test_field_with_edges()
    result = detector.process(field)
    assert has_detected_edges(result)

def test_motion_detector():
    detector = MotionDetector()
    field1 = create_static_field()
    field2 = create_moved_field()
    detector.process(field1)
    result = detector.process(field2)
    assert has_detected_motion(result)
```

### Integration Tests (molecule combinations)
```python
def test_attention_with_memory():
    focus = Focus()
    stm = ShortTermMemory()
    
    # Focus stores pattern
    field = create_test_pattern()
    focus.apply(field, focus_location=(8, 8))
    stm.store(field)
    
    # Recall from memory
    retrieved = stm.recall(empty_field())
    assert patterns_match(field, retrieved, tolerance=0.2)
```

---

## 📊 SUCCESS METRICS

Each molecule should:
- [ ] Combine 2-5 atoms effectively
- [ ] Solve a specific cognitive task
- [ ] Be 50-200 lines
- [ ] Have comprehensive docstring
- [ ] Include usage example
- [ ] Pass unit tests
- [ ] Work with other molecules

---

## 🚀 NEXT STEPS AFTER MOLECULES

1. **Organisms** - Port existing systems:

2. **Examples** - Create tutorials showing:
   - How to build custom molecules
   - How to compose molecules into organisms
   - Real-world applications

3. **Documentation** - Complete API docs and guides

---

## 💡 TIPS FOR IMPLEMENTATION

1. **Start simple** - Build core functionality first, add features later
2. **Test early** - Write tests as you build
3. **Compose don't duplicate** - Reuse atoms, don't reimplement
4. **Document behavior** - Explain what each molecule does and why
5. **Think cognitive** - Each molecule should map to a real cognitive operation
6. **Profile performance** - Ensure molecules run efficiently

---

