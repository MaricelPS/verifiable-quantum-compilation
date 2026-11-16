"""Benchmark II (DQI), submitted decoder (v3): constant-schedule key-equation solver + tower-field Forney.

Phase 1 (Bernstein-Yang style divsteps on the reversed remainders f, g):
    swap_t = (delta > 0) and g(0) != 0;  if swap_t: (f, g) <- (g, f), delta <- -delta
    c_t = g(0) / f(0);  g <- (g + c_t f) / x;  delta <- delta + 1
  f and g share one register array: f from the bottom, g reversed from the top. Dividing g by x
  pops the array's top register, which then stores c_t (the stash), so array + stash have a
  constant size.

Phase 2 (replay of the transition matrices from the right, popping the stash):
    v <- v T_t P_t   with T_t = [[x, 0], [c_t, 1]] and P_t the swap, starting from v = (0, 1).
  The new constant coefficient of v_f equals c_t v_g(0), so c_t is uncomputed from it.
  At the end v = (omega_rev, sigma_rev).

Swap bits are not stored. Phase 1 recomputes swap_t after the step as
(delta' <= 0) and (c_t != 0), which is exact except after a non-swap elimination with delta < 0;
the step index of such a (rare) step is kept in a 7-qubit exception register X. Phase 2 runs delta
backwards with the same rule, and clears swap_t with (delta_t >= 1) and (v_f(0) != 0).

Forney: sigma_rev(X) = 0 marks an error at X = alpha^j, with value X^-70 omega_rev(X) / (X sigma_rev'(X)).

Register sizes per step come from classical simulation of the algorithm (bounds_k20.json).
Shots that fall outside them (error weight below 20, two exception steps, ~9 in 100,000) may fail.
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
PACKED_LEN = 143                # registers of the packed f/g array at the start of phase 1
X_BITS = 7                      # exception slot: step index + 1 of a non-swap elimination with delta < 0
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
    """Zeroed 12-qubit registers for the polynomial arrays.

    Syndrome registers must end where they started and are allocated anyway, so an idle one waits on a
    stack and is reused before any new register is allocated; other registers go back to the builder.
    The forward pass logs every choice, and the reverse pass (`reverse = True`) replays the log in the
    mirrored order, so it hands back exactly the registers the forward pass used.
    """

    def __init__(self, builder, keep):
        self.b, self.stack, self.keep, self.log, self.reverse = builder, [], keep, [], False

    def acquire(self):
        if self.reverse:
            kind, reg = self.log.pop()
            assert kind in ('rel-syn', 'rel-free')
            if kind == 'rel-syn':
                assert self.stack.pop() is reg
                return reg
            return self.b.alloc_qubits(DEG)
        if self.stack:
            reg = self.stack.pop()
            self.log.append(('acq-syn', reg))
            return reg
        self.log.append(('acq-new', None))
        return self.b.alloc_qubits(DEG)

    def release(self, reg):
        syn = id(reg) in self.keep
        if self.reverse:
            kind, old = self.log.pop()
            assert kind in ('acq-syn', 'acq-new') and (kind == 'acq-syn') == syn
        else:
            self.log.append(('rel-syn' if syn else 'rel-free', reg if syn else None))
        if syn:
            self.stack.append(reg)
        else:
            self.b.free(reg)

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
        self.pool = Pool(builder, {id(r) for r in syndromes})
        self.scratch = None

    # ------------------------------------------------------------------ helpers
    def _mul(self, x, y, out, terms):
        tower.emit_bilinear(self.b, x, y, out, terms, self.scratch)

    # Flags. Each returns the flag qubit(s) and a cleanup that uncomputes them by measurement.
    def _delta_ge1(self):
        """p = (delta >= 1) = not sign and delta != 0."""
        b = self.b
        ztd = ZeroTest(b, self.delta)
        dz = ztd.compute()
        sign = self.delta[DELTA_BITS - 1]
        p = b.alloc_qubits(1)[0]
        b.x(km.array([sign, dz]))
        b.init_and(sign, dz, p)
        b.x(km.array([sign, dz]))

        def cleanup():
            b.x(km.array([sign, dz]))
            b.del_and(sign, dz, p)
            b.x(km.array([sign, dz]))
            b.free(p)
            ztd.uncompute()
        return p, sign, cleanup

    def _post_u(self, c):
        """u = (delta <= 0) and (c != 0)."""
        b = self.b
        p, _, clp = self._delta_ge1()
        ztc = ZeroTest(b, c)
        cz = ztc.compute()
        u = b.alloc_qubits(1)[0]
        b.x(km.array([p, cz]))
        b.init_and(p, cz, u)
        b.x(km.array([p, cz]))

        def cleanup():
            b.x(km.array([p, cz]))
            b.del_and(p, cz, u)
            b.x(km.array([p, cz]))
            b.free(u)
            ztc.uncompute()
            clp()
        return u, cleanup

    def _x_eq(self, t):
        """eq = (X == t + 1)."""
        b = self.b
        flips = km.array([self.X[i] for i in range(X_BITS) if not ((t + 1) >> i) & 1])
        b.x(flips)
        b.x(self.X)                      # ZeroTest wants all-zero; X == K  <=>  (X ^ K) == 0
        zt = ZeroTest(b, self.X)
        eq = zt.compute()

        def cleanup():
            zt.uncompute()
            b.x(self.X)
            b.x(flips)
        return eq, cleanup

    def _write_x(self, ctrl, t):
        for i in range(X_BITS):
            if ((t + 1) >> i) & 1:
                self.b.cx(ctrl, self.X[i])

    def _sb_post_compute(self, c, t, sb):
        """sb ^= (delta <= 0) and (c != 0) and not (X == t + 1)   (state after step t)."""
        b = self.b
        u, clu = self._post_u(c)
        eq, cleq = self._x_eq(t)
        b.x(eq)
        b.init_and(u, eq, sb)
        b.x(eq)
        cleq()
        clu()

    def _sb_post_uncompute(self, c, t, sb):
        b = self.b
        u, clu = self._post_u(c)
        eq, cleq = self._x_eq(t)
        b.x(eq)
        b.del_and(u, eq, sb)
        b.x(eq)
        cleq()
        clu()

    def _pre_flags(self):
        """(p, nz, sign, cleanup) with p = (delta >= 1), nz = (g0 != 0) (as negated zero flag gz)."""
        b = self.b
        p, sign, clp = self._delta_ge1()
        ztg = ZeroTest(b, self.A[-1])
        gz = ztg.compute()

        def cleanup():
            ztg.uncompute()
            clp()
        return p, gz, sign, cleanup

    def _exception_write(self, sign, gz, t):
        """X ^= (t+1) if this step is a non-swap with delta < 0 and g0 != 0 (pre-state flags)."""
        b = self.b
        ex = b.alloc_qubits(1)[0]
        b.x(gz)
        b.init_and(sign, gz, ex)
        self._write_x(ex, t)
        b.del_and(sign, gz, ex)
        b.x(gz)
        b.free(ex)

    # ------------------------------------------------------------------ phase 1
    # f and g share one array A: f from the bottom (f_i = A[i]), g reversed from the top
    # (g_i = A[len - 1 - i]). Dividing g by x pops A's top register into the stash, so A shrinks
    # by one register per step while the stash grows by one.
    def _p1(self, t):
        b, B, A = self.b, BOUNDS, self.A
        L = len(A)
        sb = b.alloc_qubits(1)[0]
        p, gz, sign, cleanup = self._pre_flags()
        b.x(gz)
        b.init_and(p, gz, sb)
        b.x(gz)
        self._exception_write(sign, gz, t)
        cleanup()
        for i in range(min(B['W'][t], L // 2)):
            b.cswap(sb, A[i], A[L - 1 - i])
        b.cx(sb, self.delta)
        b.iadd(1, target=self.delta)
        b.iadd(1, target=self.delta, control=sb)
        c = b.alloc_qubits(DEG)
        inv = tower.TowerInverse(b, A[0], out_basis='tower', lean=True)
        y = inv.compute()
        self._mul(A[L - 1], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        _to_tower(b, c)
        for i in range(min(B['R'][t], L // 2)):
            self._mul(c, A[i], A[L - 1 - i], MUL_TK_K)
        _from_tower(b, c)
        slot = A.pop()
        b.cx(c, slot)
        b.cx(slot, c)
        b.free(c)
        self.stash[t] = slot
        self._sb_post_uncompute(slot, t, sb)
        b.free(sb)

    def _p1_rev(self, t):
        b, B, A = self.b, BOUNDS, self.A
        slot = self.stash.pop(t)
        sb = b.alloc_qubits(1)[0]
        self._sb_post_compute(slot, t, sb)
        c = b.alloc_qubits(DEG)
        b.cx(slot, c)
        b.cx(c, slot)
        A.append(slot)
        L = len(A)
        _to_tower(b, c)
        for i in reversed(range(min(B['R'][t], L // 2))):
            self._mul(c, A[i], A[L - 1 - i], MUL_TK_K)
        _from_tower(b, c)
        inv = tower.TowerInverse(b, A[0], out_basis='tower', lean=True)
        y = inv.compute()
        self._mul(A[L - 1], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        b.free(c)
        b.isub(1, target=self.delta, control=sb)
        b.isub(1, target=self.delta)
        b.cx(sb, self.delta)
        for i in reversed(range(min(B['W'][t], L // 2))):
            b.cswap(sb, A[i], A[L - 1 - i])
        p, gz, sign, cleanup = self._pre_flags()
        self._exception_write(sign, gz, t)
        b.x(gz)
        b.del_and(p, gz, sb)
        b.x(gz)
        cleanup()
        b.free(sb)

    # ------------------------------------------------------------------ phase 2
    def _sb_exact(self, sb, compute):
        """sb = (delta >= 1) and (VF[0] != 0), valid after the swap of phase-2 step t (delta = delta_t)."""
        b = self.b
        p, _, clp = self._delta_ge1()
        ztv = ZeroTest(b, self.VF[0])
        vz = ztv.compute()
        b.x(vz)
        (b.init_and if compute else b.del_and)(p, vz, sb)
        b.x(vz)
        ztv.uncompute()
        clp()

    def _consume_x(self, c, t, sb):
        """X ^= (t+1) when step t was an exception: u and not sb (delta = delta_{t+1})."""
        b = self.b
        u, clu = self._post_u(c)
        ex = b.alloc_qubits(1)[0]
        b.x(sb)
        b.init_and(u, sb, ex)
        self._write_x(ex, t)
        b.del_and(u, sb, ex)
        b.x(sb)
        b.free(ex)
        clu()

    def _p2(self, j):
        b, B = self.b, BOUNDS
        t = N_SYN - 1 - j
        self.VF.insert(0, self.pool.acquire())
        self.p2_sizes.append((len(self.VF), len(self.VG)))
        self.pool.resize(self.VF, B['LVF'][j])
        self.pool.resize(self.VG, B['LVG'][j])
        c = self.stash.pop(t)
        sb = b.alloc_qubits(1)[0]
        self._sb_post_compute(c, t, sb)
        self._consume_x(c, t, sb)
        b.isub(1, target=self.delta, control=sb)          # delta_{t+1} -> delta_t
        b.isub(1, target=self.delta)
        b.cx(sb, self.delta)
        _to_tower(b, c)
        for i in range(B['M2'][j]):
            self._mul(c, self.VG[i], self.VF[i], MUL_TK_K)
        _from_tower(b, c)
        inv = tower.TowerInverse(b, self.VG[0], out_basis='tower', lean=True)
        y = inv.compute()
        self._mul(self.VF[0], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        self.pool.release(c)
        for i in range(B['W2'][j]):
            b.cswap(sb, self.VF[i], self.VG[i])
        self._sb_exact(sb, compute=False)
        b.free(sb)

    def _p2_rev(self, j):
        b, B = self.b, BOUNDS
        t = N_SYN - 1 - j
        sb = b.alloc_qubits(1)[0]
        self._sb_exact(sb, compute=True)
        for i in reversed(range(B['W2'][j])):
            b.cswap(sb, self.VF[i], self.VG[i])
        c = self.pool.acquire()
        inv = tower.TowerInverse(b, self.VG[0], out_basis='tower', lean=True)
        y = inv.compute()
        self._mul(self.VF[0], y, c, MUL_KT_K)
        inv.uncompute()
        inv.release()
        _to_tower(b, c)
        for i in reversed(range(B['M2'][j])):
            self._mul(c, self.VG[i], self.VF[i], MUL_TK_K)
        _from_tower(b, c)
        b.cx(sb, self.delta)                              # delta_t -> delta_{t+1}
        b.iadd(1, target=self.delta)
        b.iadd(1, target=self.delta, control=sb)
        self._consume_x(c, t, sb)
        self._sb_post_uncompute(c, t, sb)
        b.free(sb)
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
                inv = tower.TowerInverse(b, vg[1], gate=is_zero, lean=True)
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
        self.X = b.alloc_qubits(X_BITS)
        b.x(self.delta[0])                                   # delta = 1
        f0 = b.alloc_qubits(DEG)
        b.x(f0[0])                                           # f = 1
        middle = [b.alloc_qubits(DEG) for _ in range(PACKED_LEN - 1 - N_SYN)]
        self.A = [f0] + middle + list(self.S)                # g_i = S_{69-i} sits at A[len-1-i]
        self.stash, self.p2_sizes = {}, []
        with b.mark('phase1'):
            for t in range(N_SYN):
                self._p1(t)
        one = self.pool.acquire()
        b.x(one[0])
        self.VF, self.VG = [], [one]
        with b.mark('phase2'):
            for j in range(N_SYN):
                self._p2(j)
        trimmed = len(self.VF) > 35                # omega_rev has at most 35 coefficients
        if trimmed:
            self.pool.release(self.VF.pop())
        # Idle syndrome registers (allocated anyway) take over the contents of F's top registers,
        # which are then freed for the duration of Forney.
        idle = list(self.pool.stack)
        self.pool.stack.clear()
        moved = []
        movable = [i for i, r in enumerate(self.A) if id(r) not in self.pool.keep and r is not self.A[0]]
        for reg, idx in zip(idle, movable):
            b.swap(reg, self.A[idx])
            b.free(self.A[idx])
            self.A[idx] = reg
            moved.append(idx)
        with b.mark('forney'):
            self._forney()
        assert len(moved) == len(idle)
        for idx in reversed(moved):
            fresh = b.alloc_qubits(DEG)
            b.swap(self.A[idx], fresh)
            self.A[idx] = fresh
        self.pool.stack.extend(idle)
        self.pool.reverse = True                  # from here on, every pool call mirrors a forward one
        if trimmed:
            self.VF.append(self.pool.acquire())
        with b.mark('phase2_rev'):
            for j in reversed(range(N_SYN)):
                self._p2_rev(j)
        assert not self.VF and len(self.VG) == 1
        b.x(self.VG[0][0])
        self.pool.release(self.VG.pop())
        with b.mark('phase1_rev'):
            for t in reversed(range(N_SYN)):
                self._p1_rev(t)
        assert len(self.A) == PACKED_LEN and self.A[0] is f0
        assert all(x is y for x, y in zip(self.A[-N_SYN:], self.S))
        b.x(f0[0])
        b.free(f0)
        for reg in self.A[1:-N_SYN]:
            b.free(reg)
        b.x(self.delta[0])
        b.free(self.delta)
        b.free(self.X)
        assert not self.pool.stack and not self.pool.log
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
