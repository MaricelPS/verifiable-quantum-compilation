"""Benchmark II (DQI): constant-schedule reversible key-equation solver + tower-field Forney.

Replaces the reference's synchronized extended Euclidean algorithm (222 iterations, each paying
for every sub-step under quantum control) with two phases of exactly n = 70 steps whose shape is
known classically:

Phase 1 (Bernstein-Yang style divsteps on the reversed remainders f, g):
    swap_t = (delta > 0) and g(0) != 0;  if swap_t: (f, g) <- (g, f), delta <- -delta
    c_t = g(0) / f(0);  g <- (g + c_t f) / x;  delta <- delta + 1
  The division by x is a relabeling of g's registers, and the register freed by it stores c_t
  (the stash), so each step is reversible with zero bookkeeping cost.

Phase 2 (replay of the transition matrices from the right, popping the stash):
    v <- v T_t P_t   with T_t = [[x, 0], [c_t, 1]] and P_t the swap, starting from v = (0, 1).
  The new constant coefficient of v_f equals c_t v_g(0), so c_t is uncomputed from it.
  At the end v = (omega_rev, sigma_rev).

Forney: sigma_rev(X) = 0 marks an error at X = alpha^j, with value X^-70 omega_rev(X) / (X sigma_rev'(X)).

Register sizes per step come from classical simulation of the algorithm (bounds_k20.json):
shots with error weight below 20 (probability ~1e-5) may fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401  (competition kickmix + verifiers)
import kickmix as km

import tower

N_SYN = 70
DEG = 12
DELTA_BITS = 7
BOUNDS = json.loads((Path(__file__).with_name('bounds_k20.json')).read_text())

T_MAT, TINV_MAT = tower.basis_matrices()
TO_TOWER = tower._cnot_program(TINV_MAT)
FROM_TOWER = tower._invert_program(TO_TOWER)
MUL_TK_K = tower.precompose(tower.terms_mul4096(), ymap=TINV_MAT, outmap=T_MAT)  # tower x kickmix
MUL_KT_K = tower.precompose(tower.terms_mul4096(), xmap=TINV_MAT, outmap=T_MAT)  # kickmix x tower


def _to_tower(b, reg):
    tower._run_program(b, reg, TO_TOWER)


def _from_tower(b, reg):
    tower._run_program(b, reg, FROM_TOWER)


class ZeroTest:
    """flag = (reg == 0) by an AND tree, uncomputed by measurement."""

    def __init__(self, builder, reg):
        self.b, self.reg = builder, reg

    def compute(self):
        b = self.b
        b.x(self.reg)
        layer, self.steps = list(self.reg), []
        while len(layer) > 1:
            nxt = []
            for i in range(0, len(layer) - 1, 2):
                t = b.alloc_qubits(1)[0]
                b.init_and(layer[i], layer[i + 1], t)
                self.steps.append((layer[i], layer[i + 1], t))
                nxt.append(t)
            if len(layer) % 2:
                nxt.append(layer[-1])
            layer = nxt
        b.x(self.reg)
        return layer[0]

    def uncompute(self):
        b = self.b
        b.x(self.reg)
        for c1, c2, t in reversed(self.steps):
            b.del_and(c1, c2, t)
            b.free(t)
        b.x(self.reg)


class Pool:
    """LIFO pool of zeroed 12-qubit registers; keeps register identities stable under reversal."""

    def __init__(self, builder):
        self.b, self.free = builder, []

    def acquire(self):
        return self.free.pop() if self.free else self.b.alloc_qubits(DEG)

    def release(self, reg):
        self.free.append(reg)

    def resize(self, lst, n):
        while len(lst) < n:
            lst.append(self.acquire())
        while len(lst) > n:
            self.release(lst.pop())


class TwoPhaseDecoder:

    def __init__(self, builder, syndromes, error_mbuc):
        self.b = builder
        self.S = syndromes
        self.E = error_mbuc
        self.pool = Pool(builder)
        self.scratch = None

    # ------------------------------------------------------------------ helpers
    def _mul(self, x, y, out, terms):
        tower.emit_bilinear(self.b, x, y, out, terms, self.scratch)

    def _swap_flags(self):
        """Returns (pos, gz, cleanup) with pos = delta > 0 and gz = (g0 == 0)."""
        b = self.b
        ztd = ZeroTest(b, self.delta)
        dz = ztd.compute()
        ztg = ZeroTest(b, self.G[0])
        gz = ztg.compute()
        sign = self.delta[DELTA_BITS - 1]
        pos = b.alloc_qubits(1)[0]
        b.x(km.array([sign, dz]))
        b.init_and(sign, dz, pos)
        b.x(km.array([sign, dz]))

        def cleanup():
            b.x(km.array([sign, dz]))
            b.del_and(sign, dz, pos)
            b.x(km.array([sign, dz]))
            b.free(pos)
            ztg.uncompute()
            ztd.uncompute()
        return pos, gz, cleanup

    # ------------------------------------------------------------------ phase 1
    def _p1(self, t):
        b, B = self.b, BOUNDS
        self.p1_sizes.append((len(self.F), len(self.G)))
        self.pool.resize(self.F, B['AF'][t])
        self.pool.resize(self.G, B['AG'][t])
        sb = b.alloc_qubits(1)[0]
        pos, gz, cleanup = self._swap_flags()
        b.x(gz)
        b.init_and(pos, gz, sb)
        b.x(gz)
        cleanup()
        for i in range(B['W'][t]):
            b.cswap(sb, self.F[i], self.G[i])
        b.cx(sb, self.delta)
        b.iadd(1, target=self.delta)
        b.iadd(1, target=self.delta, control=sb)
        c = b.alloc_qubits(DEG)
        inv = tower.TowerInverse(b, self.F[0], out_basis='tower')
        y = inv.compute()
        self._mul(self.G[0], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        _to_tower(b, c)
        for i in range(B['R'][t]):
            self._mul(c, self.F[i], self.G[i], MUL_TK_K)
        _from_tower(b, c)
        slot = self.G.pop(0)
        b.cx(c, slot)
        b.cx(slot, c)
        b.free(c)
        self.stash[t] = slot
        self.swapbits[t] = sb

    def _p1_rev(self, t):
        b, B = self.b, BOUNDS
        slot, sb = self.stash.pop(t), self.swapbits[t]
        c = b.alloc_qubits(DEG)
        b.cx(slot, c)
        b.cx(c, slot)
        self.G.insert(0, slot)
        _to_tower(b, c)
        for i in reversed(range(B['R'][t])):
            self._mul(c, self.F[i], self.G[i], MUL_TK_K)
        _from_tower(b, c)
        inv = tower.TowerInverse(b, self.F[0], out_basis='tower')
        y = inv.compute()
        self._mul(self.G[0], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        b.free(c)
        b.isub(1, target=self.delta, control=sb)
        b.isub(1, target=self.delta)
        b.cx(sb, self.delta)
        for i in reversed(range(B['W'][t])):
            b.cswap(sb, self.F[i], self.G[i])
        pos, gz, cleanup = self._swap_flags()
        b.x(gz)
        b.del_and(pos, gz, sb)
        b.x(gz)
        cleanup()
        b.free(sb)
        nf, ng = self.p1_sizes.pop()
        self.pool.resize(self.G, ng)
        self.pool.resize(self.F, nf)

    # ------------------------------------------------------------------ phase 2
    def _p2(self, j):
        b, B = self.b, BOUNDS
        t = N_SYN - 1 - j
        self.VF.insert(0, self.pool.acquire())
        self.p2_sizes.append((len(self.VF), len(self.VG)))
        self.pool.resize(self.VF, B['LVF'][j])
        self.pool.resize(self.VG, B['LVG'][j])
        c = self.stash.pop(t)
        _to_tower(b, c)
        for i in range(B['M2'][j]):
            self._mul(c, self.VG[i], self.VF[i], MUL_TK_K)
        _from_tower(b, c)
        inv = tower.TowerInverse(b, self.VG[0], out_basis='tower')
        y = inv.compute()
        self._mul(self.VF[0], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        self.pool.release(c)
        for i in range(B['W2'][j]):
            b.cswap(self.swapbits[t], self.VF[i], self.VG[i])

    def _p2_rev(self, j):
        b, B = self.b, BOUNDS
        t = N_SYN - 1 - j
        for i in reversed(range(B['W2'][j])):
            b.cswap(self.swapbits[t], self.VF[i], self.VG[i])
        c = self.pool.acquire()
        inv = tower.TowerInverse(b, self.VG[0], out_basis='tower')
        y = inv.compute()
        self._mul(self.VF[0], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        _to_tower(b, c)
        for i in reversed(range(B['M2'][j])):
            self._mul(c, self.VG[i], self.VF[i], MUL_TK_K)
        _from_tower(b, c)
        self.stash[t] = c
        nvf, nvg = self.p2_sizes.pop()
        self.pool.resize(self.VG, nvg)
        self.pool.resize(self.VF, nvf)
        self.pool.release(self.VF.pop(0))

    # ------------------------------------------------------------------ Forney
    def _forney(self):
        b = self.b
        gf = km.GF2Field(DEG)
        alpha = gf.primitive_element
        order = (1 << DEG) - 1
        vg, vf = self.VG, self.VF
        m = len(self.E)
        mult = [(vg[i], gf.pow(alpha, i % order)) for i in range(1, len(vg))]
        mult += [(vf[i], gf.pow(alpha, (i - 70) % order)) for i in range(len(vf))
                 if (i - 70) % order]
        pairs = ([(vg[i], vg[1]) for i in range(3, len(vg), 2)]
                 + [(vg[1], vg[0])]
                 + [(vg[i], vg[0]) for i in range(2, len(vg), 2)]
                 + [(vf[i], vf[0]) for i in range(1, len(vf))])
        src = km.array([q for s, _ in pairs for q in s])
        dst = km.array([q for _, d in pairs for q in d])
        rsrc = km.array([q for s, _ in reversed(pairs) for q in s])
        rdst = km.array([q for _, d in reversed(pairs) for q in d])
        is_error = b.alloc_bits(1)
        forney_step = b.mark('forney_step')
        for j in range(m):
            with forney_step:
                b.bit_store(is_error[0], True, control=self.E[j])
                b.push_condition(is_error[0])
                b.cx(src, dst)
                zt = ZeroTest(b, vg[0])
                is_zero = zt.compute()
                inv = tower.TowerInverse(b, vg[1], gate=is_zero)
                y = inv.compute()
                b.gf2_phase_by_product(vf[0], y, mask=self.E[j], field=gf)
                inv.uncompute()
                inv.release()
                zt.uncompute()
                b.cx(rsrc, rdst)
                b.pop_condition()
                b.bit_store(is_error, False)
                if j + 1 < m:
                    for reg, step in mult:
                        b.gf2_imul(step, target=reg, field=gf)
        if m > 1:
            for reg, step in mult:
                b.gf2_idiv(gf.pow(step, m - 1), target=reg, field=gf)
        b.free(is_error)

    # ------------------------------------------------------------------ driver
    def build(self):
        b = self.b
        self.scratch = b.alloc_qubits(1)[0]
        self.delta = b.alloc_qubits(DELTA_BITS)
        b.x(self.delta[0])                                   # delta = 1
        f0 = b.alloc_qubits(DEG)
        b.x(f0[0])                                           # f = 1
        self.F = [f0]
        self.G = [self.S[N_SYN - 1 - i] for i in range(N_SYN)]
        self.stash, self.swapbits, self.p1_sizes, self.p2_sizes = {}, {}, [], []
        with b.mark('phase1'):
            for t in range(N_SYN):
                self._p1(t)
        one = self.pool.acquire()
        b.x(one[0])
        self.VF, self.VG = [], [one]
        with b.mark('phase2'):
            for j in range(N_SYN):
                self._p2(j)
        with b.mark('forney'):
            self._forney()
        with b.mark('phase2_rev'):
            for j in reversed(range(N_SYN)):
                self._p2_rev(j)
        assert not self.VF and len(self.VG) == 1
        b.x(self.VG[0][0])
        self.pool.release(self.VG.pop())
        with b.mark('phase1_rev'):
            for t in reversed(range(N_SYN)):
                self._p1_rev(t)
        assert len(self.F) == 1 and self.F[0] is f0
        assert all(a is b for a, b in zip(self.G, [self.S[N_SYN - 1 - i] for i in range(N_SYN)]))
        b.x(f0[0])
        b.free(f0)
        b.x(self.delta[0])
        b.free(self.delta)
        for reg in self.pool.free:
            b.free(reg)
        b.free(self.scratch)


def build_circuit(m=4095, n=N_SYN, degree=DEG):
    assert n == N_SYN and degree == DEG
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    TwoPhaseDecoder(builder, syndromes, error_mbuc).build()
    return builder.finish_circuit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', nargs='?', default='dqi.kmb')
    p.add_argument('--m', type=int, default=4095)
    a = p.parse_args()
    c = build_circuit(a.m)
    c.to_file(a.output, format='kmb')
    print(f'{a.output}: {c.operation_statistics().num_toffoli_like:,} Toffolis, {c.num_qubits} qubits')


if __name__ == '__main__':
    main()
