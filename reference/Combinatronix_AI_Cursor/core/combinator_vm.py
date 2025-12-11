
# ============================================================================
# COMBINATOR VM - combinator_vm.py
# ============================================================================

"""
Combinator Virtual Machine

Implements combinator calculus with S, K, I, B, C, W, Y combinators.
Uses graph reduction for efficient evaluation.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import json
import numpy as np


@dataclass(frozen=True)
class Comb:
    """Combinator node"""
    tag: str  # 'S', 'K', 'I', 'B', 'C', 'W', 'Y'


@dataclass(frozen=True)
class Val:
    """Value node - wraps any Python value"""
    v: Any


@dataclass
class App:
    """Application node - function applied to argument"""
    f: Any
    x: Any


@dataclass
class Thunk:
    """Lazy evaluation thunk"""
    node: Any
    value: Optional[Any] = None
    under_eval: bool = False


Node = Union[Comb, Val, App, Thunk]


def app(f: Node, x: Node) -> App:
    """Create application node"""
    return App(f, x)


def is_comb(n: Node, tag: Optional[str] = None) -> bool:
    """Check if node is a combinator (optionally with specific tag)"""
    return isinstance(n, Comb) and (tag is None or n.tag == tag)


def unwind(node: Node) -> Tuple[Node, List[Node]]:
    """Unwind application spine to get head and arguments"""
    args: List[Node] = []
    n = node
    while isinstance(n, App):
        args.append(n.x)
        n = n.f
    args.reverse()
    return n, args


def reduce_step(node: Node) -> Tuple[Node, bool]:
    """Single step of normal-order reduction"""
    
    # Handle thunks
    if isinstance(node, Thunk):
        if node.value is not None:
            return node.value, True
        if node.under_eval:
            return node, False
        node.under_eval = True
        try:
            result, changed = reduce_step(node.node)
            if changed:
                node.value = result
                return result, True
            return node, False
        finally:
            node.under_eval = False
    
    head, args = unwind(node)
    
    # S combinator: S f g x = f x (g x)
    if is_comb(head, 'S') and len(args) >= 3:
        f, g, x = args[:3]
        result = app(app(f, x), app(g, x))
        # Apply to remaining args if any
        for arg in args[3:]:
            result = app(result, arg)
        return result, True
    
    # K combinator: K x y = x
    if is_comb(head, 'K') and len(args) >= 2:
        x = args[0]
        # Apply to remaining args if any
        result = x
        for arg in args[2:]:
            result = app(result, arg)
        return result, True
    
    # I combinator: I x = x
    if is_comb(head, 'I') and len(args) >= 1:
        x = args[0]
        result = x
        for arg in args[1:]:
            result = app(result, arg)
        return result, True
    
    # B combinator: B f g x = f (g x)
    if is_comb(head, 'B') and len(args) >= 3:
        f, g, x = args[:3]
        result = app(f, app(g, x))
        for arg in args[3:]:
            result = app(result, arg)
        return result, True
    
    # C combinator: C f x y = f y x
    if is_comb(head, 'C') and len(args) >= 3:
        f, x, y = args[:3]
        result = app(app(f, y), x)
        for arg in args[3:]:
            result = app(result, arg)
        return result, True
    
    # W combinator: W f x = f x x
    if is_comb(head, 'W') and len(args) >= 2:
        f, x = args[:2]
        result = app(app(f, x), x)
        for arg in args[2:]:
            result = app(result, arg)
        return result, True
    
    # Y combinator: Y f = f (Y f) - fixed point
    if is_comb(head, 'Y') and len(args) >= 1:
        f = args[0]
        result = app(f, app(Comb('Y'), f))
        for arg in args[1:]:
            result = app(result, arg)
        return result, True
    
    # Try to reduce left, then right
    if isinstance(node, App):
        f_red, did = reduce_step(node.f)
        if did:
            return App(f_red, node.x), True
        x_red, did2 = reduce_step(node.x)
        if did2:
            return App(node.f, x_red), True
    
    return node, False


def reduce_whnf(node: Node, max_steps: int = 10000) -> Node:
    """Reduce to weak head normal form"""
    cur = node
    for _ in range(max_steps):
        cur, did = reduce_step(cur)
        if not did:
            return cur
    return cur  # Return partial result if limit exceeded


def show(node: Node) -> str:
    """Pretty print a node"""
    if isinstance(node, Comb):
        return node.tag
    if isinstance(node, Val):
        return repr(node.v)
    if isinstance(node, App):
        return f"({show(node.f)} {show(node.x)})"
    if isinstance(node, Thunk):
        return "<thunk>"
    return str(node)


def serialize(node: Node) -> dict:
    """Serialize node to JSON-compatible dict"""
    if isinstance(node, Comb):
        return {'type': 'comb', 'tag': node.tag}
    elif isinstance(node, Val):
        value = node.v
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            return {'type': 'val', 'value': value.tolist(), 'dtype': 'ndarray'}
        # Handle complex numbers
        elif isinstance(value, complex):
            return {'type': 'val', 'value': {'real': value.real, 'imag': value.imag}, 'dtype': 'complex'}
        else:
            return {'type': 'val', 'value': value}
    elif isinstance(node, App):
        return {'type': 'app', 'f': serialize(node.f), 'x': serialize(node.x)}
    elif isinstance(node, Thunk):
        return {'type': 'thunk', 'node': serialize(node.node)}
    else:
        return {'type': 'unknown', 'value': str(node)}


def deserialize(data: dict) -> Node:
    """Deserialize dict back to node"""
    if data['type'] == 'comb':
        return Comb(data['tag'])
    elif data['type'] == 'val':
        if 'dtype' in data:
            if data['dtype'] == 'ndarray':
                return Val(np.array(data['value']))
            elif data['dtype'] == 'complex':
                return Val(complex(data['value']['real'], data['value']['imag']))
        return Val(data['value'])
    elif data['type'] == 'app':
        return App(deserialize(data['f']), deserialize(data['x']))
    elif data['type'] == 'thunk':
        return Thunk(deserialize(data['node']))
    else:
        return Val(data['value'])

