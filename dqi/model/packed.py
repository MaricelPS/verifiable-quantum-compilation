"""Classical checks for the two v3 changes: the packed f/g array and the swap-bit rule.

    COMP_DIR=... python dqi/model/packed.py 30000

1. Runs phase 1 on one array of PACKED_LEN registers exactly as the circuit does (f from the bottom,
   g reversed from the top, swap and sweep ranges from bounds_k20.json, capped at half the array)
   and compares every stashed quotient with the plain model.
2. Counts the steps where swap_t = (delta_{t+1} <= 0) and (c_t != 0) is wrong. Those are non-swap
   eliminations with delta < 0, the "exception" steps the circuit records in its 7-qubit register;
   a shot with two of them fails.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import gf  # noqa: E402
from two_phase import N, phase1  # noqa: E402

BOUNDS = json.loads((Path(__file__).resolve().parents[1] / 'bounds_k20.json').read_text())
PACKED_LEN = 143


def check(seed):
    from dqi import verify_submission_dqi as V
    probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    rng = np.random.default_rng(seed)
    S, _, pos, _ = gf.sample(rng, probs=probs)
    _, _, stash_ref, swaps_ref, rec = phase1(S)
    A = [1] + [0] * (PACKED_LEN - 1 - N) + list(S)
    delta, packed_ok = 1, True
    for t in range(N):
        L = len(A)
        sb = delta >= 1 and A[L - 1] != 0
        if sb:
            for i in range(min(BOUNDS['W'][t], L // 2)):
                A[i], A[L - 1 - i] = A[L - 1 - i], A[i]
            delta = -delta
        delta += 1
        c = gf.div(A[L - 1], A[0])
        for i in range(min(BOUNDS['R'][t], L // 2)):
            A[L - 1 - i] ^= gf.mul(c, A[i])
        packed_ok &= A.pop() == 0 and c == stash_ref[t]
    exceptions = sum(
        (r['delta'] <= 0 and r['c'] != 0) != sw for r, sw in zip(rec, swaps_ref))
    return len(pos), packed_ok, exceptions


def main(n):
    with mp.Pool() as pool:
        res = pool.map(check, range(3_000_000, 3_000_000 + n), chunksize=50)
    ks = np.array([r[0] for r in res])
    ok = np.array([r[1] for r in res])
    ex = np.array([r[2] for r in res])
    print(f'{n} shots: packed layout wrong in {(~ok).sum()} ({(~ok & (ks >= 20)).sum()} with weight >= 20)')
    print(f'exception steps: >= 1 in {(ex >= 1).sum()} shots, >= 2 in {(ex >= 2).sum()} shots')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
