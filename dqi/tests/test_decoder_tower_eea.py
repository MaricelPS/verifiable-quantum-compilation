"""End-to-end check of the decoder on a small instance with the official verifier."""
import kickmix as km
from dqi import verify_submission_dqi as V

import decoder_tower_eea as decoder


def test_small_instance_passes():
    m, n, degree = 300, 10, 12
    spec = V.ProblemSpecDQI.benchmark_instance(m=m, n=n, degree=degree, num_shots=2560)
    out = V.verify(decoder.build_circuit(m, n, degree), spec, seed=11, max_workers=1)
    assert out['status'] == 'PASS'
    assert out['value_errors'] == 0 and out['phase_errors'] == 0
