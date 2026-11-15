"""Toffolis and qubits added by each stage of the reference invsqrt circuit (forward direction)."""
import sys
import _paths  # noqa: F401,E402  (competition kickmix + verifiers)
from invsqrt import nr_reference_circuit as nr  # noqa: E402

STAGES = ['_normalize', '_select_interval', '_load_coefficients', '_horner', '_square', '_multiply_m',
          '_correct', '_multiply_y0', '_denormalize']
orig = {s: getattr(nr, s) for s in STAGES}
prev = 0
for k, stage in enumerate(STAGES):
    for j, s in enumerate(STAGES):
        setattr(nr, s, orig[s] if j <= k else (lambda *a, **kw: None))
    c = nr.build_nr_circuit(10, compute_only=True)
    tof = c.operation_statistics().num_toffoli_like
    print(f'{stage:20s} +{tof - prev:6d} Toffolis   (cumulative {tof:6d}, qubits {c.num_qubits})')
    prev = tof
