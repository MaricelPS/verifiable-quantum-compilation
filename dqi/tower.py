"""Tower-field GF(2^12) = GF(((2^3)^2)^2) arithmetic as bilinear AND networks for kickmix.

Every nonlinear step is a bilinear map out ^= B(X, Y) realised as a list of AND terms
(xform, yform, outvec): the AND of a linear form of X with a linear form of Y is XORed
into the output bits in outvec. Each term costs exactly one Toffoli to compute and zero
to uncompute (X-basis measurement plus CZ phase fix-ups), so the costs are:

    GF(2^3) multiply      6 ANDs        GF(2^3) inverse        6 ANDs
    GF(2^6) multiply     18 ANDs        GF(2^6) inverse       24 ANDs
    GF(2^12) multiply    54 ANDs        GF(2^12) inverse      78 ANDs

against 63 (multiply) and 315 (inverse with scaffold) for the kickmix built-ins.

Registers may store values in any GF(2)-basis: the term forms are expressed in tower
coordinates and converted to the register's own coordinates with the basis matrices.
"""
from __future__ import annotations

import functools
import random

import _paths  # noqa: F401  (competition kickmix)
import kickmix as km

DEG = 12


# ----------------------------------------------------------------------------------------
# GF(2) linear algebra on bit-vectors stored as Python ints.
# A matrix is a list of column vectors: M[j] is the image of basis vector e_j.

def mat_apply(cols, v):
    out = 0
    j = 0
    while v:
        if v & 1:
            out ^= cols[j]
        v >>= 1
        j += 1
    return out


def mat_inverse(cols, n):
    """Inverse of an n x n GF(2) matrix given as columns."""
    # Gaussian elimination on rows of [M | I].
    rows = []
    for i in range(n):
        r = 0
        for j in range(n):
            if (cols[j] >> i) & 1:
                r |= 1 << j
        rows.append((r, 1 << i))
    for c in range(n):
        p = next(i for i in range(c, n) if (rows[i][0] >> c) & 1)
        rows[c], rows[p] = rows[p], rows[c]
        for i in range(n):
            if i != c and (rows[i][0] >> c) & 1:
                rows[i] = (rows[i][0] ^ rows[c][0], rows[i][1] ^ rows[c][1])
    # rows[i][1] is row i of the inverse.
    inv_cols = [0] * n
    for i in range(n):
        for j in range(n):
            if (rows[i][1] >> j) & 1:
                inv_cols[j] |= 1 << i
    return inv_cols


def row_times_mat(row, cols):
    """Row vector times matrix: (row . M)_j = parity(row & M[j])."""
    out = 0
    for j, c in enumerate(cols):
        if bin(row & c).count('1') & 1:
            out |= 1 << j
    return out


def identity(n):
    return [1 << j for j in range(n)]


# ----------------------------------------------------------------------------------------
# Classical tower field.

def _mul_poly_mod(a, b, mod, deg):
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if (a >> deg) & 1:
            a ^= mod
    return r


GF8_MOD = 0b1011  # u^3 + u + 1


def mul8(a, b):
    return _mul_poly_mod(a, b, GF8_MOD, 3)


def _trace8(x):
    return (x ^ mul8(x, x) ^ mul8(mul8(x, x), mul8(x, x))) & 1  # trace lands in GF(2)


LAMBDA = next(l for l in range(1, 8) if _trace8(l) == 1)  # v^2 + v + LAMBDA irreducible


def mul64(x, y):
    x0, x1, y0, y1 = x & 7, x >> 3, y & 7, y >> 3
    p0, p1 = mul8(x0, y0), mul8(x1, y1)
    p2 = mul8(x0 ^ x1, y0 ^ y1)
    return (p0 ^ mul8(LAMBDA, p1)) | ((p2 ^ p0) << 3)


def _pow64(x, e):
    r = 1
    while e:
        if e & 1:
            r = mul64(r, x)
        x = mul64(x, x)
        e >>= 1
    return r


def _trace64(x):
    t, s = 0, x
    for _ in range(6):
        t ^= s
        s = mul64(s, s)
    return t


MU = next(m for m in range(1, 64) if _trace64(m) == 1)  # w^2 + w + MU irreducible


def mul4096(x, y):
    x0, x1, y0, y1 = x & 63, x >> 6, y & 63, y >> 6
    p0, p1 = mul64(x0, y0), mul64(x1, y1)
    p2 = mul64(x0 ^ x1, y0 ^ y1)
    return (p0 ^ mul64(MU, p1)) | ((p2 ^ p0) << 6)


def pow4096(x, e):
    r = 1
    while e:
        if e & 1:
            r = mul4096(r, x)
        x = mul4096(x, x)
        e >>= 1
    return r


@functools.lru_cache(maxsize=None)
def basis_matrices(degree: int = DEG):
    """Returns (T, Tinv): T maps tower coordinates to kickmix GF(2^12) coordinates."""
    gf = km.GF2Field(degree)
    order = (1 << degree) - 1
    # A primitive tower element g and its minimal polynomial over GF(2).
    g = next(c for c in range(2, 1 << degree)
             if all(pow4096(c, order // p) != 1 for p in (3, 5, 7, 13)))
    powers = [1]
    for _ in range(degree):
        powers.append(mul4096(powers[-1], g))
    # Find the GF(2) relation sum c_k g^k = 0 with c_12 = 1 by solving on bit vectors.
    cols = powers[:degree]
    inv = mat_inverse(cols, degree)
    coeffs = mat_apply(inv, powers[degree])  # g^12 = sum coeffs_k g^k
    # Find a root r of the same polynomial in the kickmix field.
    def is_root(r):
        acc, rk = 0, 1
        for k in range(degree):
            if (coeffs >> k) & 1:
                acc ^= rk
            rk = gf.mul(rk, r)
        return acc == rk
    r = next(c for c in range(2, 1 << degree) if is_root(c))
    # Discrete log of tower basis vectors, then map g^k -> r^k.
    log = {}
    acc = 1
    for k in range(order):
        log[acc] = k
        acc = mul4096(acc, g)
    T = [gf.pow(r, log[1 << j]) for j in range(degree)]
    Tinv = mat_inverse(T, degree)
    # Sanity: the map is a ring homomorphism.
    rng = random.Random(1)
    for _ in range(300):
        a, b = rng.randrange(1 << degree), rng.randrange(1 << degree)
        assert mat_apply(T, mul4096(a, b)) == gf.mul(mat_apply(T, a), mat_apply(T, b))
    return T, Tinv


# ----------------------------------------------------------------------------------------
# Bilinear term lists in tower coordinates: (xform, yform, outvec).

def _const_mul_matrix(mul, c, n):
    return [mul(c, 1 << j) for j in range(n)]


@functools.lru_cache(maxsize=None)
def terms_mul8():
    # Karatsuba-style 6-AND product of 3-bit polynomials, reduced modulo u^3 + u + 1.
    # Raw product coefficients d0..d4 from p_i = a_i b_i and p_ij = (a_i + a_j)(b_i + b_j).
    prods = {
        'p0': (0b001, 0b001), 'p1': (0b010, 0b010), 'p2': (0b100, 0b100),
        'p01': (0b011, 0b011), 'p02': (0b101, 0b101), 'p12': (0b110, 0b110),
    }
    # d0 = p0; d1 = p01+p0+p1; d2 = p02+p0+p1+p2; d3 = p12+p1+p2; d4 = p2.
    d = [0] * 5
    contrib = {
        'p0': [0, 1, 2], 'p1': [1, 2, 3], 'p2': [2, 3, 4],
        'p01': [1], 'p02': [2], 'p12': [3],
    }
    reduce = {0: 0b001, 1: 0b010, 2: 0b100, 3: 0b011, 4: 0b110}  # u^3 = u+1, u^4 = u^2+u
    out = []
    for name, (xf, yf) in prods.items():
        vec = 0
        for k in contrib[name]:
            vec ^= reduce[k]
        out.append((xf, yf, vec))
    _check_terms(out, 3, 3, lambda a, b: mul8(a, b))
    return tuple(out)


def _lift(terms, xsh, ysh, outmap):
    """Shifts forms and pushes outvec through the linear map `outmap` (callable)."""
    return [(xf << xsh, yf << ysh, outmap(ov)) for xf, yf, ov in terms]


def _ext_mul_terms(base, n, const_mul):
    """Karatsuba over a quadratic extension z^2 = z + c of a base field of n bits."""
    lo = lambda ov: ov
    hi = lambda ov: ov << n
    out = []
    # p0 = x0 y0 contributes to both the constant and z coefficients.
    out += [(xf, yf, ov | (ov << n)) for xf, yf, ov in base]
    # p1 = x1 y1 contributes c * p1 to the constant coefficient.
    cm = [const_mul(1 << j) for j in range(n)]
    out += [(xf << n, yf << n, mat_apply(cm, ov)) for xf, yf, ov in base]
    # p2 = (x0 + x1)(y0 + y1) contributes to the z coefficient.
    out += [(xf | (xf << n), yf | (yf << n), ov << n) for xf, yf, ov in base]
    return out


@functools.lru_cache(maxsize=None)
def terms_mul64():
    t = _ext_mul_terms(terms_mul8(), 3, lambda v: mul8(LAMBDA, v))
    _check_terms(t, 6, 6, mul64)
    return tuple(t)


@functools.lru_cache(maxsize=None)
def terms_mul4096():
    t = _ext_mul_terms(terms_mul64(), 6, lambda v: mul64(MU, v))
    _check_terms(t, 12, 12, mul4096)
    return tuple(t)


def _eval_terms(terms, x, y):
    out = 0
    for xf, yf, ov in terms:
        if (bin(xf & x).count('1') & 1) and (bin(yf & y).count('1') & 1):
            out ^= ov
    return out


def _check_terms(terms, nx, ny, fn, samples=400):
    rng = random.Random(7)
    pairs = [(rng.randrange(1 << nx), rng.randrange(1 << ny)) for _ in range(samples)]
    if nx + ny <= 12:
        pairs = [(a, b) for a in range(1 << nx) for b in range(1 << ny)]
    for a, b in pairs:
        assert _eval_terms(terms, a, b) == fn(a, b), (a, b)


def precompose(terms, xmap=None, ymap=None, outmap=None):
    """Rewrites tower-coordinate terms for registers stored in other bases.

    xmap/ymap: row-transform matrices (columns) R such that tower = R . register,
    i.e. a tower form l becomes l . R on register bits. outmap: matrix (columns)
    taking tower output vectors to register output vectors.
    """
    out = []
    for xf, yf, ov in terms:
        if xmap is not None:
            xf = row_times_mat(xf, xmap)
        if ymap is not None:
            yf = row_times_mat(yf, ymap)
        if outmap is not None:
            ov = mat_apply(outmap, ov)
        out.append((xf, yf, ov))
    return out


# ----------------------------------------------------------------------------------------
# Quantum emitters.

def _prep(builder, reg, form):
    """CNOT-folds the linear form onto its lowest qubit; returns (pivot, undo-list)."""
    bits = [j for j in range(len(reg)) if (form >> j) & 1]
    p = bits[0]
    for j in bits[1:]:
        builder.cx(reg[j], reg[p])
    return p, bits[1:]


def _unprep(builder, reg, p, rest):
    for j in reversed(rest):
        builder.cx(reg[j], reg[p])


def emit_bilinear(builder, X, Y, out, terms, scratch):
    """out ^= B(X, Y). X, Y, out must be disjoint qubit arrays. One Toffoli per term."""
    for xf, yf, ov in terms:
        if not ov or not xf or not yf:
            continue
        px, rx = _prep(builder, X, xf)
        py, ry = _prep(builder, Y, yf)
        targets = [out[k] for k in range(len(out)) if (ov >> k) & 1]
        if len(targets) == 1:
            builder.ccx(X[px], Y[py], targets[0])
        else:
            builder.init_and(X[px], Y[py], scratch)
            builder.cx(scratch, km.array(targets))
            builder.del_and(X[px], Y[py], scratch)
        _unprep(builder, Y, py, ry)
        _unprep(builder, X, px, rx)


def emit_bilinear_mbuc_clear(builder, X, Y, out, terms):
    """Clears `out` (holding exactly B(X, Y)) by X-measurement and CZ phase fix-ups."""
    bits = builder.alloc_bits(len(out))
    builder.hmr(out, bits)
    for k in range(len(out)):
        sel = [(xf, yf) for xf, yf, ov in terms if (ov >> k) & 1 and xf and yf]
        if not sel:
            continue
        builder.push_condition(bits[k])
        for xf, yf in sel:
            px, rx = _prep(builder, X, xf)
            py, ry = _prep(builder, Y, yf)
            builder.cz(X[px], Y[py])
            _unprep(builder, Y, py, ry)
            _unprep(builder, X, px, rx)
        builder.pop_condition()
    builder.free(bits)


def emit_linear(builder, src, dst, cols):
    """dst ^= M . src, M given by columns (cols[j] = image of src bit j)."""
    for j, c in enumerate(cols):
        for k in range(len(dst)):
            if (c >> k) & 1:
                builder.cx(src[j], dst[k])


def _pair_program(xf, yf, n):
    """CNOT program putting forms xf, yf (independent) on qubits 0 and 1 of an n-qubit register."""
    rows = [xf, yf]
    for e in range(n):
        cand = 1 << e
        # Complete to a basis greedily (GF(2) rank check).
        test = rows + [cand]
        if _rank(test, n) == len(test):
            rows.append(cand)
        if len(rows) == n:
            break
    cols = [sum(((rows[i] >> j) & 1) << i for i in range(n)) for j in range(n)]
    return _cnot_program(cols)


def _rank(rows, n):
    rows, r = list(rows), 0
    for c in range(n):
        p = next((i for i in range(r, len(rows)) if (rows[i] >> c) & 1), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> c) & 1:
                rows[i] ^= rows[r]
        r += 1
    return r


def emit_quadratic_inplace(builder, X, out, terms, scratch):
    """out ^= Q(X) for a quadratic form given as terms (xf, yf, ov) over the same register X."""
    for xf, yf, ov in terms:
        if not ov or not xf or not yf:
            continue
        targets = [out[k] for k in range(len(out)) if (ov >> k) & 1]
        if xf == yf:                                  # (l.x)^2 = l.x over GF(2)
            p, rest = _prep(builder, X, xf)
            builder.cx(X[p], km.array(targets))
            _unprep(builder, X, p, rest)
            continue
        prog = _pair_program(xf, yf, len(X))
        _run_program(builder, X, prog)
        if len(targets) == 1:
            builder.ccx(X[0], X[1], targets[0])
        else:
            builder.init_and(X[0], X[1], scratch)
            builder.cx(scratch, km.array(targets))
            builder.del_and(X[0], X[1], scratch)
        _run_program(builder, X, _invert_program(prog))


def emit_quadratic_mbuc_clear(builder, X, out, terms):
    """Clears `out` (holding exactly Q(X)) by X-measurement and CZ / Z phase fix-ups."""
    bits = builder.alloc_bits(len(out))
    builder.hmr(out, bits)
    for k in range(len(out)):
        sel = [(xf, yf) for xf, yf, ov in terms if (ov >> k) & 1 and xf and yf]
        if not sel:
            continue
        builder.push_condition(bits[k])
        for xf, yf in sel:
            if xf == yf:
                p, rest = _prep(builder, X, xf)
                builder.z(X[p])
                _unprep(builder, X, p, rest)
                continue
            prog = _pair_program(xf, yf, len(X))
            _run_program(builder, X, prog)
            builder.cz(X[0], X[1])
            _run_program(builder, X, _invert_program(prog))
        builder.pop_condition()
    builder.free(bits)


# ----------------------------------------------------------------------------------------
# Tower inverse.

def _sq_matrix(mul, n):
    return [mul(1 << j, 1 << j) for j in range(n)]


@functools.lru_cache(maxsize=None)
def _inverse_tables():
    T, Tinv = basis_matrices()
    m64 = terms_mul64()
    m8 = terms_mul8()
    # N = a*b + a^2 + MU*b^2 (6 bits), x = a + b w.
    N_bil = [(xf, yf, ov) for xf, yf, ov in m64]  # X = a (6), Y = b (6)
    sq64 = _sq_matrix(mul64, 6)
    N_lin_a = sq64
    N_lin_b = [mul64(MU, c) for c in sq64]
    # n = c*d + c^2 + LAMBDA*d^2 (3 bits), N = c + d v.
    sq8 = _sq_matrix(mul8, 3)
    n_lin_c = sq8
    n_lin_d = [mul8(LAMBDA, c) for c in sq8]
    # ninv = n^4 * n^2, written as bilinear(n, ncopy) with forms precomposed by the squarings.
    sq8_2 = [mat_apply(sq8, c) for c in sq8]  # x -> x^4
    # terms over (z1 = x^4, z2 = x^2): l . z1 = l . (S4 x) -> row l times S4.
    ninv_terms = [(row_times_mat(xf, sq8_2), row_times_mat(yf, sq8), ov) for xf, yf, ov in m8]
    # Ninv = ((c + d) * ninv) + (d * ninv) v. X = N (6 bits), Y = ninv (3 bits).
    Ninv_terms = []
    for xf, yf, ov in m8:
        Ninv_terms.append((xf | (xf << 3), yf, ov))        # (c + d) * ninv -> low
        Ninv_terms.append((xf << 3, yf, ov << 3))          # d * ninv -> high
    # y = ((a + b) * Ninv) + (b * Ninv) w. X = x (12 bits), Y = Ninv (6 bits).
    y_terms = []
    for xf, yf, ov in m64:
        y_terms.append((xf | (xf << 6), yf, ov))
        y_terms.append((xf << 6, yf, ov << 6))
    return dict(N_bil=N_bil, N_lin_a=N_lin_a, N_lin_b=N_lin_b, n_lin_c=n_lin_c,
                n_lin_d=n_lin_d, ninv=ninv_terms, Ninv=Ninv_terms, y=y_terms)


class TowerInverse:
    """y = gate ? x^{-1} : 0 for x in a kickmix-basis register; 78 (+3 if gated) Toffolis."""

    def __init__(self, builder, x, *, gate=None, out_basis='kickmix', lean=False):
        self.b = builder
        self.x = x
        self.gate = gate
        # lean: 3 fewer qubits. Gated: the gate goes into the copy of n (no separate gated n).
        # Ungated: n^6 is computed from n alone with in-place linear changes (no copy of n).
        self.lean = lean
        T, Tinv = basis_matrices()
        tb = _inverse_tables()
        self.tb = tb
        self.out_map = T if out_basis == 'kickmix' else None
        self.scratch = builder.alloc_qubits(1)[0]
        # Express x in tower coordinates in place (free), restored by `release`.
        self._to_tower = _cnot_program(Tinv)
        _run_program(builder, x, self._to_tower)

    def compute(self):
        b, x, tb = self.b, self.x, self.tb
        a_q, b_q = x[0:6], x[6:12]
        self.N = b.alloc_qubits(6)
        emit_bilinear(b, a_q, b_q, self.N, tb['N_bil'], self.scratch)
        emit_linear(b, a_q, self.N, tb['N_lin_a'])
        emit_linear(b, b_q, self.N, tb['N_lin_b'])
        c_q, d_q = self.N[0:3], self.N[3:6]
        self.n = b.alloc_qubits(3)
        emit_bilinear(b, c_q, d_q, self.n, terms_mul8(), self.scratch)
        emit_linear(b, c_q, self.n, tb['n_lin_c'])
        emit_linear(b, d_q, self.n, tb['n_lin_d'])
        self.ninv = None
        if self.lean and self.gate is not None:
            self.ncopy = b.alloc_qubits(3)
            b.init_and(self.gate, self.n, self.ncopy)
            self.ninv = b.alloc_qubits(3)
            emit_bilinear(b, self.n, self.ncopy, self.ninv, tb['ninv'], self.scratch)
        elif self.lean:
            self.ninv = b.alloc_qubits(3)
            emit_quadratic_inplace(b, self.n, self.ninv, tb['ninv'], self.scratch)
        else:
            if self.gate is not None:
                self.ng = b.alloc_qubits(3)
                b.init_and(self.gate, self.n, self.ng)
            else:
                self.ng = self.n
            self.ncopy = b.alloc_qubits(3)
            b.cx(self.ng, self.ncopy)
            self.ninv = b.alloc_qubits(3)
            emit_bilinear(b, self.ng, self.ncopy, self.ninv, tb['ninv'], self.scratch)
        self.Ninv = b.alloc_qubits(6)
        emit_bilinear(b, self.N, self.ninv, self.Ninv, tb['Ninv'], self.scratch)
        self.y = b.alloc_qubits(12)
        y_terms = tb['y'] if self.out_map is None else precompose(tb['y'], outmap=self.out_map)
        self.y_terms = y_terms
        emit_bilinear(b, x, self.Ninv, self.y, y_terms, self.scratch)
        return self.y

    def uncompute(self):
        b, x, tb = self.b, self.x, self.tb
        emit_bilinear_mbuc_clear(b, x, self.Ninv, self.y, self.y_terms)
        b.free(self.y)
        emit_bilinear_mbuc_clear(b, self.N, self.ninv, self.Ninv, tb['Ninv'])
        b.free(self.Ninv)
        if self.lean and self.gate is not None:
            emit_bilinear_mbuc_clear(b, self.n, self.ncopy, self.ninv, tb['ninv'])
            b.free(self.ninv)
            b.del_and(self.gate, self.n, self.ncopy)
            b.free(self.ncopy)
        elif self.lean:
            emit_quadratic_mbuc_clear(b, self.n, self.ninv, tb['ninv'])
            b.free(self.ninv)
        else:
            emit_bilinear_mbuc_clear(b, self.ng, self.ncopy, self.ninv, tb['ninv'])
            b.free(self.ninv)
            b.cx(self.ng, self.ncopy)
            b.free(self.ncopy)
            if self.gate is not None:
                b.del_and(self.gate, self.n, self.ng)
                b.free(self.ng)
        c_q, d_q = self.N[0:3], self.N[3:6]
        emit_linear(b, d_q, self.n, tb['n_lin_d'])
        emit_linear(b, c_q, self.n, tb['n_lin_c'])
        emit_bilinear_mbuc_clear(b, c_q, d_q, self.n, terms_mul8())
        b.free(self.n)
        a_q, b_q = x[0:6], x[6:12]
        emit_linear(b, b_q, self.N, tb['N_lin_b'])
        emit_linear(b, a_q, self.N, tb['N_lin_a'])
        emit_bilinear_mbuc_clear(b, a_q, b_q, self.N, tb['N_bil'])
        b.free(self.N)

    def release(self):
        _run_program(self.b, self.x, _invert_program(self._to_tower))
        self.b.free(self.scratch)


# ----------------------------------------------------------------------------------------
# In-place invertible linear maps as CNOT programs.

def _cnot_program(cols):
    """CNOT list (control, target) realising v -> M v in place (M invertible, columns)."""
    n = len(cols)
    # Reduce M to I with row ops; row op "row_t ^= row_c" == CNOT(c -> t) acting on v.
    rows = []
    for i in range(n):
        r = 0
        for j in range(n):
            if (cols[j] >> i) & 1:
                r |= 1 << j
        rows.append(r)
    ops = []
    for c in range(n):
        if not (rows[c] >> c) & 1:
            p = next(i for i in range(c + 1, n) if (rows[i] >> c) & 1)
            rows[c] ^= rows[p]
            ops.append((p, c))
        for i in range(n):
            if i != c and (rows[i] >> c) & 1:
                rows[i] ^= rows[c]
                ops.append((c, i))
    # E_k ... E_1 M = I  =>  M = E_1 ... E_k (each self-inverse); apply E_k first.
    return list(reversed(ops))


def _invert_program(ops):
    return list(reversed(ops))


def _run_program(builder, reg, ops):
    for c, t in ops:
        builder.cx(reg[c], reg[t])


def emit_mul_tower(builder, X, Y, out, scratch, *, x_basis='kickmix', y_basis='kickmix',
                   out_basis='kickmix'):
    """out ^= X * Y with 54 Toffolis; each register may be in kickmix or tower basis."""
    T, Tinv = basis_matrices()
    terms = precompose(
        terms_mul4096(),
        xmap=Tinv if x_basis == 'kickmix' else None,
        ymap=Tinv if y_basis == 'kickmix' else None,
        outmap=T if out_basis == 'kickmix' else None,
    )
    emit_bilinear(builder, X, Y, out, terms, scratch)
    return terms
