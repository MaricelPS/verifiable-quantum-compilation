"""Per-step register sizes for `decoder.py`, measured on simulated shots.

The circuit's layout is fixed classically: at each step it needs to know how many coefficient
registers each polynomial may occupy, which positions a swap or a multiply sweep must touch, and
which registers are guaranteed zero and can be reused. This script simulates the classical model on
random shots (plus extra shots at low error weights), takes the maxima over shots with weight
>= KMIN, and writes them to `bounds_k20.json`.

    COMP_DIR=... python dqi/model/bounds.py 100000 1000    # ~4 min on 2 cores
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401
import gf  # noqa: E402
from two_phase import N, tlen  # noqa: E402

KMIN = 20                     # shots with fewer errors (probability ~1e-5) are allowed to fail
OUT = Path(__file__).resolve().parents[1] / f'bounds_k{KMIN}.json'


def simulate(args):
    """One shot. Returns its weight, whether sigma was right, and per-step length records."""
    seed, weight = args
    from dqi import verify_submission_dqi as V
    probs = V.optimal_dqi_weights(4095, 35, 12, 2016) ** 2
    rng = np.random.default_rng(seed)
    S, _, pos, _ = gf.sample(rng, weight=weight, probs=probs)
    rec = np.zeros((N, 16), dtype=np.int64)
    f, g, delta = [1] + [0] * N, [S[N - 1 - i] for i in range(N)] + [0], 1
    stash, swaps = [], []
    for t in range(N):
        lf0, lg0 = tlen(f), tlen(g)
        swap = delta > 0 and g[0] != 0
        if swap:
            f, g, delta = g, f, -delta
        c = gf.div(g[0], f[0])
        L = max(len(f), len(g))
        f, g = f + [0] * (L - len(f)), g + [0] * (L - len(g))
        g = [g[i] ^ gf.mul(c, f[i]) for i in range(L)][1:]
        delta += 1
        stash.append(c)
        swaps.append(swap)
        rec[t, :7] = [lf0, lg0, swap, c != 0, tlen(f), tlen(g), delta]
    vf, vg = [], [1]
    for j, t in enumerate(reversed(range(N))):
        c = stash[t]
        lvf0, lvg0 = tlen(vf), tlen(vg)
        new = [0] + vf
        new += [0] * (max(len(new), len(vg)) - len(new))
        for i in range(len(vg)):
            new[i] ^= gf.mul(c, vg[i])
        vf = new
        lvf1 = tlen(vf)
        if swaps[t]:
            vf, vg = vg, vf
        rec[j, 8:15] = [lvf0, lvg0, c != 0, lvf1, swaps[t], tlen(vf), tlen(vg)]
    sigma = list(reversed(vg[:tlen(vg)]))
    ok = len(sigma) == len(pos) + 1 and all(gf.poly_eval(sigma, gf.pw(-int(p))) == 0 for p in pos)
    return len(pos), ok, rec


def bounds(ks, recs, kmin=KMIN):
    r = recs[ks >= kmin]
    lf0, lg0, sw, cnz, lf1, lg1, dl = (r[:, :, i] for i in range(7))
    fs, gs = np.where(sw, lg0, lf0), np.where(sw, lf0, lg0)          # after the conditional swap
    comb = np.where(cnz, np.maximum(fs, gs), gs)
    b = dict(
        AF=np.maximum.reduce([lf0, fs, lf1]).max(0),                  # registers holding f
        AG=np.maximum.reduce([lg0, gs, comb]).max(0),                 # registers holding g
        W=np.where(sw, np.maximum(lf0, lg0), 0).max(0),               # positions a swap touches
        R=np.where(cnz, fs, 0).max(0),                                # positions g += c f touches
    )
    vf0, vg0, c2, vf1, sw2, vfa, vga = (r[:, :, 8 + i] for i in range(7))
    b.update(
        VF=np.maximum.reduce([vf0 + 1, vf1, vfa]).max(0),
        VG=np.maximum.reduce([vg0, vga]).max(0),
        M2=np.where(c2, vg0, 0).max(0),                               # positions v_f += c v_g touches
        W2=np.where(sw2, np.maximum(vf1, vg0), 0).max(0),
    )
    out = {k: [int(x) for x in v] for k, v in b.items()}
    out['LVF'] = [max(a, c, d) for a, c, d in zip(out['VF'], out['W2'], out['M2'])]
    out['LVG'] = [max(a, c, d) for a, c, d in zip(out['VG'], out['W2'], out['M2'])]
    out['delta_range'] = [int(dl.min()), int(dl.max())]
    return out


def main(n_random, n_per_weight):
    tasks = [(1000 + i, None) for i in range(n_random)]
    tasks += [(500000 + 100 * k + i, k) for k in range(18, 36) for i in range(n_per_weight)]
    with mp.Pool() as pool:
        res = pool.map(simulate, tasks, chunksize=20)
    ks = np.array([r[0] for r in res])
    oks = np.array([r[1] for r in res])
    recs = np.stack([r[2] for r in res])
    print(f'{len(res)} shots, sigma correct in {oks.sum()}')
    out = bounds(ks, recs)
    OUT.write_text(json.dumps(out))
    print('wrote', OUT)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6000, int(sys.argv[2]) if len(sys.argv) > 2 else 150)
