"""Classical model of the two-phase key-equation solver used by `decoder.py`.

Run it to check the algorithm on random shots before touching a quantum circuit:

    COMP_DIR=... python dqi/model/two_phase.py 400

For every shot it checks that sigma has its roots exactly at the error positions, that every quotient
popped in phase 2 is recoverable, and that the Forney formula used by the circuit gives every error value.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import gf  # noqa: E402

N = 70  # syndromes


def tlen(p):
    """Length without trailing zero coefficients."""
    n = len(p)
    while n and p[n - 1] == 0:
        n -= 1
    return n


def phase1(S, n=N):
    """Divsteps on the reversed remainders. Returns f, g, the stashed quotients and the swap bits."""
    f = [1] + [0] * n                              # reversed x^n
    g = [S[n - 1 - i] for i in range(n)] + [0]     # reversed S(x)
    delta = 1
    stash, swaps, rec = [], [], []
    for t in range(n):
        swap = delta > 0 and g[0] != 0
        if swap:
            f, g = g, f
            delta = -delta
        c = gf.div(g[0], f[0])
        L = max(len(f), len(g))
        f, g = f + [0] * (L - len(f)), g + [0] * (L - len(g))
        comb = [g[i] ^ gf.mul(c, f[i]) for i in range(L)]
        assert comb[0] == 0
        g = comb[1:]                               # exact division by x
        delta += 1
        stash.append(c)
        swaps.append(swap)
        rec.append(dict(t=t, lf=tlen(f), lg=tlen(g), delta=delta, swap=swap, c=c))
    return f, g, stash, swaps, rec


def phase2(stash, swaps):
    """Replays v <- v T_t P_t from t = n-1 down to 0, popping c_t from v_f's constant term."""
    vf, vg = [0], [1]
    rec = []
    for t in reversed(range(len(stash))):
        c = stash[t]
        L = max(len(vf) + 1, len(vg))
        new = [0] + vf
        new += [0] * (L - len(new))
        for i in range(len(vg)):
            new[i] ^= gf.mul(c, vg[i])
        vf = new
        recoverable = c == 0 or vg[0] != 0         # c = vf[0] / vg[0]
        rec.append(dict(t=t, c=c, recoverable=recoverable, lvf=tlen(vf), lvg=tlen(vg)))
        if swaps[t]:
            vf, vg = vg, vf
    return vf, vg, rec                             # (omega_rev, sigma_rev)


def forney_value(vf, vg, j):
    """Error value at position j (X = alpha^j) as the circuit computes it; 0 if sigma_rev(X) != 0."""
    s = odd = w = 0
    for i, cf in enumerate(vg):
        term = gf.mul(cf, gf.pw(i * j))
        s ^= term
        if i % 2:
            odd ^= term
    for i, cf in enumerate(vf):
        w ^= gf.mul(cf, gf.pw((i - 70) * j))
    return gf.mul(w, gf.inv(odd)) if s == 0 else 0


def main(num_shots):
    from dqi import verify_submission_dqi as V
    probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    rng = np.random.default_rng(0)
    bad_sigma = bad_pop = bad_value = checked = 0
    for _ in range(num_shots):
        S, sig, pos, vals = gf.sample(rng, probs=probs)
        _, _, stash, swaps, _ = phase1(S)
        vf, vg, rec = phase2(stash, swaps)
        bad_pop += sum(not r['recoverable'] for r in rec)
        sigma = list(reversed(vg[:tlen(vg)]))
        bad_sigma += not (len(sigma) == len(pos) + 1
                          and all(gf.poly_eval(sigma, gf.pw(-int(p))) == 0 for p in pos))
        errors = dict(zip(map(int, pos), map(int, vals)))
        for j in list(errors) + [int(x) for x in rng.integers(0, 4095, 20)]:
            checked += 1
            bad_value += forney_value(vf, vg, j) != errors.get(j, 0)
    print(f'{num_shots} shots: wrong sigma {bad_sigma}, unrecoverable pops {bad_pop}, '
          f'wrong error values {bad_value} of {checked} positions checked')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
