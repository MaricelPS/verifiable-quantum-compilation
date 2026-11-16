"""Benchmark II (DQI): Reed-Solomon MBUC decoder with tower-field GF(2^12) arithmetic.

Built on the organizers' reference circuit (`dqi/dqi_reference_circuit.py`, synchronized extended
Euclidean algorithm after Proos and Zalka), which it subclasses.

Changes against the reference circuit:
  * Forney/Chien: the per-position inverse sigma'^{-1} (315 Toffolis) becomes a
    tower-field inverse (78 Toffolis) gated by the sigma==0 flag (3 Toffolis), which
    also replaces the 12-Toffoli `located = is_zero AND omega` product. The zero test
    is an AND tree uncomputed by measurement (11 Toffolis instead of 2 x 12).
  * EEA quotient: b/a via gated tower inverse + 54-Toffoli tower multiply
    (about 135 Toffolis instead of 402).
  * EEA sweep: the controlled multiply (75 Toffolis) becomes a gated copy of q
    (12 Toffolis, measurement-uncomputed) and a 54-Toffoli tower multiply.
"""
from __future__ import annotations

import argparse

import _paths  # noqa: F401  (competition kickmix + verifiers)
import kickmix as km
from dqi import dqi_reference_circuit as ref  # noqa: E402

import tower  # noqa: E402

T_MAT, TINV_MAT = tower.basis_matrices()
TO_TOWER = tower._cnot_program(TINV_MAT)
FROM_TOWER = tower._invert_program(TO_TOWER)
MUL_TT_K = tower.precompose(tower.terms_mul4096(), outmap=T_MAT)  # tower x tower -> kickmix
MUL_KT_K = tower.precompose(tower.terms_mul4096(), xmap=TINV_MAT, outmap=T_MAT)


def _to_tower(builder, reg):
    tower._run_program(builder, reg, TO_TOWER)


def _from_tower(builder, reg):
    tower._run_program(builder, reg, FROM_TOWER)


class ZeroTest:
    """flag = (reg == 0) through an AND tree; uncomputed by measurement (0 Toffolis)."""

    def __init__(self, builder, reg):
        self.b, self.reg = builder, reg

    def compute(self):
        b = self.b
        b.x(self.reg)
        layer = list(self.reg)
        self.steps = []
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


class TowerDecoder(ref.ProosAndZalkaDecoder):

    # ------------------------------------------------------------------ EEA multiply step
    def _multiply_and_add_or_sub(self, builder, reg_a, reg_b, n_A, n_a, n_q, ctrl, it, *,
                                 is_adjoint):
        c0_hc, _, _, c3_hc = ctrl
        m = len(reg_b)

        def accumulate_quotient(index, control):
            with builder.mark('quotient_div'):
                inv = tower.TowerInverse(builder, reg_a[index], gate=control,
                                         out_basis='tower')
                y = inv.compute()
                scratch = builder.alloc_qubits(1)[0]
                tower.emit_bilinear(builder, reg_b[index], y, q_i, MUL_KT_K, scratch)
                builder.free(scratch)
                inv.uncompute()
                inv.release()

        def gated_mul(i, control):
            # reg_b[i] ^= (control * q_i) * reg_a[i]
            qg = builder.alloc_qubits(self.field.degree)
            builder.init_and(control, q_i, qg)
            _to_tower(builder, qg)
            _to_tower(builder, reg_a[i])
            scratch = builder.alloc_qubits(1)[0]
            tower.emit_bilinear(builder, qg, reg_a[i], reg_b[i], MUL_TT_K, scratch)
            builder.free(scratch)
            _from_tower(builder, reg_a[i])
            _from_tower(builder, qg)
            builder.del_and(control, q_i, qg)
            builder.free(qg)

        def sweep_positions():
            with builder.mark('sweep_positions'):
                lim_q = min(m - 1, 1 + it)
                lim_q1 = min(m, 1 + it)
                lim_a = min(m - 2, it // 2 + 1)
                lim_A = m - 2
                order = range(m) if is_adjoint else range(m - 1, -1, -1)
                _active, _xor_threshold = ref._active, ref._xor_threshold

                with (
                    builder.unary_iteration(n_q, control=c0_hc) as eq_i,
                    builder.unary_iteration(n_q, control=c3_hc) as eq_ip1,
                    builder.unary_iteration(n_a, control=c3_hc) as cur_a,
                    builder.unary_iteration(n_A, control=c0_hc) as cur_A,
                ):
                    mul = builder.alloc_qubits(1)[0]

                    def endpoint(i):
                        _xor_threshold(builder, cur_a, lim_a, mul, c3_hc, i + 1,
                                       at_least=True, lowest=1)
                        _xor_threshold(builder, cur_A, lim_A, mul, c0_hc, m - i,
                                       at_least=True, lowest=1)

                    prev = None
                    for i in order:
                        if prev is None:
                            endpoint(i)
                        else:
                            builder.cx(
                                km.array([
                                    _active(cur_a, min(prev, i) + 1, lim_a),
                                    _active(cur_A, m - max(prev, i), lim_A),
                                ]),
                                mul,
                            )
                        prev = i

                        swaps = [a for a in (_active(eq_i, i, lim_q),
                                             _active(eq_ip1, i + 1, lim_q1))
                                 if a is not False]
                        multiplies = i < lim_a or i >= 2

                        if is_adjoint and multiplies:
                            gated_mul(i, mul)
                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[i])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[i])
                            builder.cx(km.array(swaps), both)
                            builder.free(both)
                        if not is_adjoint and multiplies:
                            gated_mul(i, mul)

                    endpoint(prev)
                    builder.free(mul)

        with builder.mark('multiply_and_add_or_sub'):
            q_i = builder.alloc_qubits(self.field.degree)
            leading, trailing = (0, m - 1) if is_adjoint else (m - 1, 0)
            accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc)
            builder.iadd(n_a, target=n_q)
            sweep_positions()
            builder.isub(n_a, target=n_q)
            accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc)
            builder.free(q_i)

    # ------------------------------------------------------------------ Chien + Forney
    def _chien_search_and_forney_mbuc(self, builder, reg_b, error_mbuc, n_b, n_B):
        gf = self.field
        degree = gf.degree
        m = len(reg_b)
        t = self.n_syndromes // 2
        block_length = (1 << degree) - 1
        primitive = gf.primitive_element

        with builder.mark('pre_align_reg_b'):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=False)

        sigma = reg_b[0]
        sigma_prime = reg_b[1]
        omega = reg_b[m // 2 + t]
        is_error = builder.alloc_bits(1)

        omega_indices = range(m // 2 + 1, m // 2 + t + 1)
        step_multipliers = [
            (k, gf.pow(primitive, k)) for k in range(1, t + 1)
        ] + [
            (k, gf.pow(primitive, (m - 2 - k) % block_length))
            for k in omega_indices
            if (m - 2 - k) % block_length != 0
        ]
        sum_pairs = (
            [(reg_b[k], omega) for k in range(m // 2 + 1, m // 2 + t)]
            + [(reg_b[k], sigma_prime) for k in range(3, t + 1, 2)]
            + [(reg_b[k], sigma) for k in range(2, t + 1, 2)]
            + [(sigma_prime, sigma)]
        )
        fwd_src = km.array([q for src, _ in sum_pairs for q in src])
        fwd_dst = km.array([q for _, dst in sum_pairs for q in dst])
        rev_src = km.array([q for src, _ in reversed(sum_pairs) for q in src])
        rev_dst = km.array([q for _, dst in reversed(sum_pairs) for q in dst])

        num_positions = min(block_length, len(error_mbuc))
        forney_step = builder.mark('forney_step')
        for j in range(num_positions):
            with forney_step:
                builder.bit_store(is_error[0], True, control=error_mbuc[j])
                builder.push_condition(is_error[0])

                builder.cx(fwd_src, fwd_dst)
                zt = ZeroTest(builder, sigma)
                is_zero = zt.compute()
                inv = tower.TowerInverse(builder, sigma_prime, gate=is_zero)
                y = inv.compute()  # sigma'^{-1} where sigma == 0, else 0 (kickmix basis)
                builder.gf2_phase_by_product(omega, y, mask=error_mbuc[j], field=gf)
                inv.uncompute()
                inv.release()
                zt.uncompute()
                builder.cx(rev_src, rev_dst)

                builder.pop_condition()
                builder.bit_store(is_error, False)

                if j + 1 < num_positions:
                    for k, step in step_multipliers:
                        builder.gf2_imul(step, target=reg_b[k], field=gf)

        if num_positions > 1:
            for k, step in step_multipliers:
                builder.gf2_idiv(gf.pow(step, num_positions - 1), target=reg_b[k], field=gf)

        builder.free(is_error)

        with builder.mark('unalign_reg_b'):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=True)


def build_circuit(m, n, degree, decoder_cls=TowerDecoder):
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    decoder_cls(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc,
        n_iters=ref.default_n_iters(n))
    return builder.finish_circuit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', nargs='?', default='dqi.kmb')
    p.add_argument('--m', type=int, default=4095)
    p.add_argument('--n', type=int, default=70)
    p.add_argument('--degree', type=int, default=12)
    a = p.parse_args()
    c = build_circuit(a.m, a.n, a.degree)
    c.to_file(a.output, format='kmb')
    print(f'{a.output}: {c.operation_statistics().num_toffoli_like:,} Toffolis, '
          f'{c.num_qubits} qubits')


if __name__ == '__main__':
    main()
