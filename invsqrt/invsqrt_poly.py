"""Benchmark I (phasing by 1/sqrt(x)) with a table-driven piecewise polynomial instead of Newton-Raphson.

Pipeline, for x in [1, max_x] held in a 22-qubit register:

1. Normalize: count leading zero pairs s (unary flags) and shift x in place to x_norm = x 4^s, so
   M = x_norm / 2^20 lies in [1, 4) and 1/sqrt(x) = 2^(s - 10) f(M) with f(M) = M^(-1/2).
2. Look up, from the top `k` bits of x_norm, the coefficient magnitudes a_0..a_d of a degree-d
   polynomial for f on that segment (QROM; uncomputed later by measurement, which is cheap).
3. Horner in subtract form, in place: r_{d-1} = a_{d-1} - T(a_d delta), ..., y = a_0 - T(r_1 delta),
   where delta is the offset of M inside its segment (the low bits of x_norm, truncated) and T drops
   partial-product bits below each register's last bit. Signs alternate because f's derivatives do.
4. De-normalize with unary-controlled rotations (y1 = y 2^s) and phase by the bits of y1.
5. Undo 4-1.

`model` reproduces the circuit's integer arithmetic exactly, so the worst phase error over all x is
known before building or simulating anything.
"""
from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np

N_P = 10
WIDTH = 2 * N_P + 2          # 22 bits of x
FRAC = WIDTH - 2             # M = x_norm / 2^20
MAX_X = 3 * (2**N_P - 2)**2
MASK64 = (1 << 64) - 1


@dataclasses.dataclass(frozen=True)
class Design:
    k: int = 9                      # address bits (2 integer bits of M + k-2 fractional bits)
    degree: int = 2
    widths: tuple = (30, 22, 13)    # fractional bits of a_0, a_1, ..., a_d (a_0 also gets 1 integer bit)
    delta_bits: tuple = (20, 20)    # last delta bit (as 2^-m) used by each Horner step, from the top step down
    drop: int = 0                   # low bits of y1 left out of the phase
    c0_bias: int = 0                # added to a_0 (in its ulps) to centre the truncation error
    controlled_add: bool = False    # controlled adders instead of AND-into-scratch rows
    binary_shift: bool = False      # s as 4 binary bits (barrel shifts) instead of 10 unary flags
    bitnorm: bool = False           # normalize by single bits; the parity of the shift selects a table half

    @property
    def seg_bits(self) -> int:      # fractional bits of M resolved by the address
        return self.k - 2

    @property
    def out_bits(self) -> int:      # fractional bits of y1, in turns
        return self.widths[0] + N_P

    @property
    def frac(self) -> int:          # fractional bits of the normalized mantissa held by x
        return WIDTH - 1 if self.bitnorm else FRAC

    def delta_ms(self, step: int) -> list[int]:
        """Exponents m (weight 2^-m) of the delta bits used by Horner step `step` (0 = top)."""
        first = self.k if self.bitnorm else self.seg_bits + 1
        return list(range(first, min(self.delta_bits[step], self.frac) + 1))


# Coefficients.


def segments(design: Design):
    """(address, segment start, segment width, scale) for every used address; f = scale / sqrt(M0 + delta)."""
    if design.bitnorm:
        h = 2.0**-(design.k - 1)                                   # M'' in [1, 2), k-1 fractional bits
        for p in (0, 1):
            for frac in range(2**(design.k - 1)):
                yield p * 2**(design.k - 1) + frac, 1 + frac * h, h, 2.0**((p - 1) / 2)
    else:
        h = 2.0**-design.seg_bits
        for addr in range(2**design.seg_bits, 2**design.k):       # M >= 1
            yield addr, addr * h, h, 1.0


def coefficient_table(design: Design) -> list[list[int]]:
    """table[j][address] = round(|c_j| 2^w_j) for the segment of each address (0 for unused ones)."""
    tables = [[0] * 2**design.k for _ in range(design.degree + 1)]
    for addr, m0, h, scale in segments(design):
        # Near-minimax fit: Chebyshev interpolation on the segment, converted to powers of delta.
        nodes = (np.cos((2 * np.arange(64) + 1) * np.pi / 128) + 1) * h / 2
        cheb = np.polynomial.Chebyshev.fit(nodes, scale / np.sqrt(m0 + nodes), design.degree, domain=[0, h])
        coefs = cheb.convert(kind=np.polynomial.Polynomial, domain=[0, h], window=[0, h]).coef
        for j, c in enumerate(coefs):
            if (c < 0) != (j % 2 == 1):
                raise ValueError(f'unexpected sign of c_{j} at segment {addr}')
            value = round(abs(c) * 2**design.widths[j]) + (design.c0_bias if j == 0 else 0)
            tables[j][addr] = value
    return tables


# Exact classical model of the circuit.


def normalize(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = (WIDTH - (np.floor(np.log2(xs)).astype(np.int64) + 1)) // 2
    return s, xs << (2 * s)


def model(design: Design, xs: np.ndarray) -> np.ndarray:
    """y1 (integer, out_bits fractional bits of a turn) for each x, as the circuit computes it."""
    xs = np.asarray(xs, dtype=np.int64)
    if design.bitnorm:
        t = WIDTH - (np.floor(np.log2(xs)).astype(np.int64) + 1)
        x_norm, s = xs << t, t >> 1
        addr = (t & 1) * 2**(design.k - 1) + ((x_norm >> (WIDTH - design.k)) & (2**(design.k - 1) - 1))
    else:
        s, x_norm = normalize(xs)
        addr = x_norm >> (WIDTH - design.k)
    tables = [np.array(t, dtype=object) for t in coefficient_table(design)]
    regs = [t[addr].astype(np.int64) for t in tables]
    w = design.widths
    acc = regs[design.degree]
    for step, j in enumerate(range(design.degree - 1, -1, -1)):
        target = regs[j].copy()
        for m in design.delta_ms(step):
            bit = (x_norm >> (design.frac - m)) & 1
            shift = w[j + 1] + m - w[j]                       # acc 2^-w[j+1] 2^-m in units of 2^-w[j]
            term = acc >> shift if shift >= 0 else acc << -shift
            target = target - bit * term
        acc = target
    y1 = acc << s
    return y1


def phase_units(design: Design, y1: np.ndarray) -> np.ndarray:
    """The applied phase in units of 2^-64 turns (mod 1), with the dropped low bits removed."""
    y = (y1 >> design.drop) << design.drop
    return (y.astype(np.uint64) << np.uint64(64 - design.out_bits)) & np.uint64(MASK64)


def reference(xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target phase and tolerance in units of 2^-64 turns, exactly as the official verifier."""
    rs = [math.isqrt((1 << 130) // int(x)) for x in xs]
    target = np.array([((r + 1) >> 1) & MASK64 for r in rs], dtype=np.uint64)
    tol = np.array([min(r >> 25, 1 << 63) for r in rs], dtype=np.uint64)
    return target, tol


_REF = None


def worst_ratio(design: Design) -> float:
    """max |phase error| / tol(x) over every x in [1, max_x] (must stay below 1, with margin for synthesis)."""
    global _REF
    xs = np.arange(1, MAX_X + 1, dtype=np.int64)
    if _REF is None:
        _REF = reference(xs)
    target, tol = _REF
    y1 = model(design, xs)
    if (y1 < 0).any() or (y1 >= 1 << (design.out_bits + 1)).any():
        return math.inf
    err = (phase_units(design, y1) - target).view(np.int64).astype(np.float64)
    return float(np.max(np.abs(err) / tol.astype(np.float64)))


# Circuit.


def build(design: Design):
    import kickmix as km

    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name='x')
    flags = b.alloc_qubits(5 if design.bitnorm else 4 if design.binary_shift else N_P)
    w = design.widths
    y1 = b.alloc_qubits(design.out_bits + 1)                  # a_0 lives in its low w0 + 1 bits
    regs = [y1[:w[0] + 1]] + [b.alloc_qubits(w[j]) for j in range(1, design.degree + 1)]
    scratch = None if design.controlled_add else b.alloc_qubits(max(len(r) for r in regs))
    tables = coefficient_table(design)
    packed = [sum(tables[j][a] << sum(len(regs[i]) for i in range(j)) for j in range(design.degree + 1))
              for a in range(2**design.k)]
    lookup_target = km.array([q for r in regs for q in r])
    if design.bitnorm:
        address = km.array(list(x[WIDTH - design.k:WIDTH - 1]) + [flags[0]])   # k-1 mantissa bits, then parity
    else:
        address = x[WIDTH - design.k:]

    from invsqrt import nr_reference_circuit as nr

    def horner(inverse: bool) -> None:
        steps = list(enumerate(range(design.degree - 1, -1, -1)))
        for step, j in (reversed(steps) if inverse else steps):
            acc, target = regs[j + 1], regs[j]
            for m in design.delta_ms(step):
                ctl = x[design.frac - m]
                shift = w[j + 1] + m - w[j]
                terms = [(acc[i], i - shift) for i in range(max(0, shift), len(acc)) if i - shift < len(target)]
                if not terms:
                    continue
                if design.controlled_add:
                    lo = terms[0][1]
                    src = km.array([q for q, _ in terms])
                    (b.iadd if inverse else b.isub)(src, target=target[lo:], control=ctl)
                else:
                    nr._add_row(b, inverse, ctl, terms, target, scratch, subtract=True)

    def compute(inverse: bool) -> None:
        if design.bitnorm:
            norm = lambda inv: _normalize_binary(b, inv, x, flags, unit=1)
            denorm = lambda inv: _denormalize_binary(b, inv, flags[1:], y1, w[0] + 1)
        elif design.binary_shift:
            norm = lambda inv: _normalize_binary(b, inv, x, flags)
            denorm = lambda inv: _denormalize_binary(b, inv, flags, y1, w[0] + 1)
        else:
            norm = lambda inv: nr._normalize(b, inv, x, flags)
            denorm = lambda inv: nr._denormalize(b, inv, flags, y1)
        if not inverse:
            norm(False)
            b.init_lookup(table=packed, address=address, target=lookup_target)
            horner(False)
            denorm(False)
        else:
            denorm(True)
            horner(True)
            b.del_lookup(table=packed, address=address, target=lookup_target)
            norm(True)

    compute(False)
    f = design.out_bits
    for j in range(design.drop, len(y1)):
        if j - f + 1 < 1:                                     # weights of a turn or more are identities
            b.z_pow(y1[j], Fraction(2) ** (j - f + 1))
    compute(True)
    b.free(flags)
    b.free(y1)
    for r in regs[1:]:
        b.free(r)
    if scratch is not None:
        b.free(scratch)
    return b.finish_circuit()


def _zero_flag(b, inverse: bool, bits, flag) -> None:
    """`flag := all(bits == 0)` with an AND chain whose intermediates are measured away."""
    import kickmix as km
    bits = list(bits)
    b.x(km.array(bits))
    if len(bits) == 1:
        b.cx(bits[0], flag)
    else:
        chain = b.alloc_qubits(max(len(bits) - 2, 0))
        prev = bits[0]
        for i, q in enumerate(bits[1:-1]):
            b.init_and(prev, q, chain[i])
            prev = chain[i]
        (b.del_and if inverse else b.init_and)(prev, bits[-1], flag)
        for i in reversed(range(len(bits) - 2)):
            b.del_and(chain[i - 1] if i else bits[0], bits[i + 1], chain[i])
        b.free(chain)
    b.x(km.array(bits))


def _normalize_binary(b, inverse: bool, x, sbits, unit: int = 2) -> None:
    """x := x unit^s with s = sum 2^i sbits[i]: stage i shifts by unit 2^i bits if the top ones are zero."""
    stages = list(reversed(range(len(sbits))))              # largest shift first
    for i in (reversed(stages) if inverse else stages):
        shift = unit * 2 ** i
        if inverse:
            b.right_rotate(x, shift, control=sbits[i])
            _zero_flag(b, True, x[len(x) - shift:], sbits[i])
        else:
            _zero_flag(b, False, x[len(x) - shift:], sbits[i])
            b.left_rotate(x, shift, control=sbits[i])


def _denormalize_binary(b, inverse: bool, sbits, y1, base: int) -> None:
    """y1 := y1 2^s, smallest shift first so each lane only spans the bits that can be occupied."""
    stages = list(range(len(sbits)))
    reach = base
    lanes = []
    for i in stages:
        reach = min(len(y1), reach + 2 ** i)
        lanes.append(y1[:reach])
    for i in (reversed(stages) if inverse else stages):
        (b.right_rotate if inverse else b.left_rotate)(lanes[i], 2 ** i, control=sbits[i])


def score(design: Design, ratio: float | None = None) -> dict:
    """Qubits, Toffolis and the verifier's cost estimate, with the rotation price from the error margin."""
    c = build(design)
    stats = c.operation_statistics()
    ratio = worst_ratio(design) if ratio is None else ratio
    f = design.out_bits
    n_rot = sum(1 for j in range(design.drop, f) if j - f + 1 < -2)       # finer than a T gate
    n_t = sum(1 for j in range(design.drop, f) if j - f + 1 == -2)
    margin = (1 - ratio) * 2.0**-24 / math.sqrt(MAX_X)                      # turns, at the worst x
    eps = math.pi * margin / max(n_rot, 1)
    t = n_t + n_rot * (0.56 * math.log2(1 / eps) + 5.3) if ratio < 1 else math.inf
    cost = stats.num_toffoli_like + t / 4
    return dict(ratio=ratio, qubits=c.num_qubits, toffolis=stats.num_toffoli_like, rotations=n_rot,
                cost=cost, score=c.num_qubits * cost, circuit=c)
