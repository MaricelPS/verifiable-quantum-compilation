"""The submitted two-phase decoder (v3): classical model on random shots, and the full circuit on verifier shots."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import numpy as np  # noqa: E402

import gf  # noqa: E402
import two_phase  # noqa: E402


def test_model_decodes_random_shots():
    from dqi import verify_submission_dqi as V
    probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    rng = np.random.default_rng(5)
    for _ in range(100):
        S, _, pos, vals = gf.sample(rng, probs=probs)
        _, _, stash, swaps, _ = two_phase.phase1(S)
        vf, vg, rec = two_phase.phase2(stash, swaps)
        assert all(r['recoverable'] for r in rec)
        for p, v in zip(pos, vals):
            assert two_phase.forney_value(vf, vg, int(p)) == int(v)


@pytest.mark.slow
def test_full_circuit_on_verifier_shots():
    from dqi import verify_submission_dqi as V
    import decoder
    circuit = decoder.build_circuit()                       # ~30 s
    spec = V.ProblemSpecDQI.benchmark_instance(num_shots=2560)   # fewer shots fail the 2^-128 bound
    out = V.verify(circuit, spec, seed=3, max_workers=1)
    assert out['status'] == 'PASS'
    # By design about 1 shot in 10,000 fails (two exception steps, or error weight below 20).
    assert out['value_errors'] + out['phase_errors'] <= 2
    assert circuit.num_qubits == 1811
