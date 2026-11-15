"""Simulates the tower-field circuits with kickmix and checks values, clean ancillas and phases."""
import numpy as np
import pytest

import tower
import kickmix as km

GF = km.GF2Field(12)


def _bits(vals, n):
    return ((np.asarray(vals)[None, :] >> np.arange(n)[:, None]) & 1).astype(bool)


def _ints(bits):
    return (bits.astype(np.int64) * (1 << np.arange(bits.shape[0]))[:, None]).sum(0)


def _run(build, inputs, widths):
    """Runs build(builder, regs) on a batch; returns register values, dirty-ancilla mask, phases, Toffolis."""
    b = km.CircuitBuilder()
    regs = [b.create_quantum_register(w, name=f'r{i}') for i, w in enumerate(widths)]
    build(b, regs)
    c = b.finish_circuit()
    sim = km.Simulator(batch_size=len(inputs[0]), debug_print=False, seed=3)
    sim.use_same_registers_as(c)
    for r, v, w in zip(regs, inputs, widths):
        sim.write_across_shots(r, _bits(v, w))
    sim.do(c)
    named = {q for r in regs for q in r}
    anc = km.array([km.q(k) for k in range(c.num_qubits) if km.q(k) not in named])
    dirty = sim.read_across_shots(anc).any(0) if len(anc) else np.zeros(len(inputs[0]), bool)
    return ([_ints(sim.read_across_shots(r)) for r in regs], dirty, sim.read_phase_flipped_across_shots(),
            c.operation_statistics().num_toffoli_like)


def test_basis_change_is_a_field_isomorphism():
    T, _ = tower.basis_matrices()
    rng = np.random.default_rng(0)
    for a, b in rng.integers(0, 4096, size=(2000, 2)):
        a, b = int(a), int(b)
        assert tower.mat_apply(T, tower.mul4096(a, b)) == GF.mul(tower.mat_apply(T, a), tower.mat_apply(T, b))


def test_multiply_54_toffolis():
    rng = np.random.default_rng(1)
    x, y, z = (rng.integers(0, 4096, 4096) for _ in range(3))

    def build(b, regs):
        s = b.alloc_qubits(1)[0]
        tower.emit_mul_tower(b, *regs, s)
        b.free(s)

    (xo, yo, zo), dirty, phase, tof = _run(build, [x, y, z], [12, 12, 12])
    expected = np.array([c ^ GF.mul(int(a), int(b)) for a, b, c in zip(x, y, z)])
    assert (zo == expected).all() and (xo == x).all() and (yo == y).all()
    assert not dirty.any() and not phase.any()
    assert tof == 54


@pytest.mark.parametrize('gated', [False, True])
def test_inverse_all_inputs(gated):
    x = np.arange(4096)
    gate = np.random.default_rng(2).integers(0, 2, 4096) if gated else np.ones(4096, int)

    def build(b, regs):
        inv = tower.TowerInverse(b, regs[0], gate=regs[2][0] if gated else None)
        b.cx(inv.compute(), regs[1])   # copy the inverse out, then uncompute by measurement
        inv.uncompute()
        inv.release()

    (xo, yo, _), dirty, phase, tof = _run(build, [x, np.zeros(4096, int), gate], [12, 12, 1])
    expected = np.array([GF.invert(int(a)) if a and g else 0 for a, g in zip(x, gate)])
    assert (yo == expected).all() and (xo == x).all()
    assert not dirty.any() and not phase.any()
    assert tof == (81 if gated else 78)
