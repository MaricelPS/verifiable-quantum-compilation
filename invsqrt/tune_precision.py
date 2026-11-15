"""Greedy search for the smallest register widths of the reference invsqrt circuit.

For each candidate Precision: the exact fixed-point model gives the worst relative error over all
x in [1, max_x]; the built circuit gives qubits and Toffolis; the score estimate adds the phase
rotations priced as in the verifier. Keeps a width reduction only if the error stays below
`margin * 2^-24`.
"""
import dataclasses
import math
import sys

import numpy as np

import _paths  # noqa: F401,E402  (competition kickmix + verifiers)
from invsqrt import nr_reference_circuit as nr  # noqa: E402

N_P = 10
MAX_X = 3 * (2**N_P - 2)**2
XS = np.arange(1, MAX_X + 1)
TOL = 2.0**-24


def evaluate(prec):
    try:
        err = nr.relative_error(N_P, prec, XS).max()
    except OverflowError:
        return None
    if err >= TOL:
        return dict(err=err / TOL, score=math.inf)
    c = nr.build_nr_circuit(N_P, prec=prec)
    q, tof = c.num_qubits, c.operation_statistics().num_toffoli_like
    f = prec.output_bits(N_P)
    n_rot = max(f - 3, 0)                               # bits of weight below 1/8 turn are rotations
    # Margin left for synthesis, in turns, at the worst x (x = max_x has the smallest tolerance).
    margin = (TOL - err) / math.sqrt(MAX_X)
    eps = math.pi * margin / n_rot
    t = 1 + n_rot * (0.56 * math.log2(1 / eps) + 5.3)
    cost = tof + t / 4
    return dict(err=err / TOL, q=q, tof=tof, n_rot=n_rot, cost=cost, score=q * cost)


def main(margin=0.95):
    prec = nr.Precision()
    best = evaluate(prec)
    print('start', dataclasses.astuple(prec), {k: round(v, 3) for k, v in best.items()}, flush=True)
    improved = True
    while improved:
        improved = False
        for field in ('result_bits', 'product_bits', 'square_bits', 'coef_bits', 'delta_bits'):
            while True:
                cand = dataclasses.replace(prec, **{field: getattr(prec, field) - 1})
                r = evaluate(cand)
                if r is None or r['err'] >= margin or r['score'] >= best['score']:
                    break
                prec, best, improved = cand, r, True
                print(field, dataclasses.astuple(prec), {k: round(v, 3) for k, v in best.items()}, flush=True)
    print('final', prec, best)


if __name__ == '__main__':
    main()
