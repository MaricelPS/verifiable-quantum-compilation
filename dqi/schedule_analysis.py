"""Records the EEA degree registers after every forward iteration and measures how input-dependent the
sub-step schedule is (the reason the circuit cannot simply skip sub-steps by iteration number).

    COMP_DIR=... python dqi/schedule_analysis.py 2560
"""
import sys

import numpy as np

import _paths  # noqa: F401
import kickmix as km
from dqi import dqi_reference_circuit as ref
from dqi import verify_submission_dqi as V

M, N, DEG = 4095, 70, 12
NAMES = ['n_A', 'n_B', 'n_a', 'n_b', 'n_q', 'counter', 'flag', 'halting']
OBS = []


class Recorder(ref.ProosAndZalkaDecoder):
    """Forward EEA only, copying the bookkeeping registers to fresh observation registers each iteration."""

    def _synchronized_eea(self, builder, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter,
                          halting_counter, *, n_iters, is_adjoint):
        if is_adjoint:
            return
        for it in range(n_iters):
            self._iteration(builder, it, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter,
                            halting_counter)
            regs = [n_A, n_B, n_a, n_b, n_q, counter, km.array([flag]), halting_counter]
            obs = [builder.create_quantum_register(len(r), name=f'o{it}_{k}') for k, r in enumerate(regs)]
            for o, r in zip(obs, regs):
                builder.cx(r, o)
            OBS.append(obs)

    def _iteration(self, builder, it, reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter,
                   halting_counter):
        # Same body as the reference forward iteration.
        is_c3 = builder.alloc_qubits(1)[0]
        builder.init_and(counter[0], counter[1], is_c3)
        is_hc_zero = builder.alloc_qubits(1)[0]
        builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
        ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)
        self._update_flag(builder, reg_b, n_A, n_B, n_a, n_b, n_q, flag, ctrl)
        self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a, n_q, ctrl, it, is_adjoint=False)
        self._shift_b_array_right(builder, reg_b, n_B, n_b, flag, ctrl, it, is_adjoint=False)
        self._update_metadata(builder, n_B, n_b, n_q, flag, ctrl, is_adjoint=False)
        self._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b, flag, ctrl)
        self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)
        builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
        builder.free(is_hc_zero)
        self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter, flag, is_c3,
                                     is_adjoint=False)
        builder.del_and(counter[0], counter[1], is_c3)
        builder.free(is_c3)
        self._update_counter(builder, counter, flag, is_adjoint=False)

    def _chien_search_and_forney_mbuc(self, *args, **kwargs):
        return


def main(num_shots):
    b = km.CircuitBuilder()
    syn = [b.create_quantum_register(DEG, name=f'S{i}') for i in range(N)]
    err = [b.create_classical_register(DEG, name=f'E{j}') for j in range(M)]
    Recorder(N, km.GF2Field(DEG)).build_syndrome_decoding_mbuc(
        b, syndromes=syn, error_mbuc=err, n_iters=ref.default_n_iters(N))
    circuit = b.finish_circuit()
    spec = V.ProblemSpecDQI.benchmark_instance()
    tables = V.FieldTables(km.GF2Field(DEG))
    probs = V.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
    rng = np.random.default_rng(12345)
    batches = []
    for _ in range(num_shots // 256):
        syndromes, _, _ = V.sample_shots(spec, tables, rng, 256, probs)
        sim = km.Simulator(batch_size=256, debug_print=False, seed=1)
        sim.use_same_registers_as(circuit)
        sim.write_across_shots(km.array([q for s in syn for q in s]), V.to_bits(syndromes, DEG))
        sim.do(circuit)
        arr = np.zeros((len(OBS), len(NAMES), 256), dtype=np.int64)
        for it, obs in enumerate(OBS):
            for k, o in enumerate(obs):
                bits = sim.read_across_shots(o).astype(np.int64)
                arr[it, k] = (bits * (1 << np.arange(len(o)))[:, None]).sum(0)
        batches.append(arr)
    arr = np.concatenate(batches, axis=2)
    counter = np.concatenate([np.zeros((1, arr.shape[2]), int), arr[:-1, 5]], 0)  # value during iteration
    halted = np.concatenate([np.zeros((1, arr.shape[2]), int), arr[:-1, 7]], 0) > 0
    for v in range(4):
        per_shot = ((counter == v) & ~halted).sum(0)
        print(f'sub-step {v}: {per_shot.mean():.1f} iterations per shot (min {per_shot.min()}, max {per_shot.max()})')
    first_done = np.argmax(arr[:, 7] > 0, axis=0)
    print(f'EEA finishes at iteration {first_done.min()}-{first_done.max()} of {ref.default_n_iters(N)}')
    off = [int((counter[it] != np.bincount(counter[it]).argmax()).sum()) for it in range(len(OBS))]
    for it in (10, 50, 100, 130, 150):
        print(f'iteration {it:3d}: {off[it]:4d} of {arr.shape[2]} shots off the majority sub-step '
              f'({off[it] / arr.shape[2]:.2%})')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2560)
