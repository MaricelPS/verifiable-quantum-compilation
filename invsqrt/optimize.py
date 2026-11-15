"""Greedy coordinate descent over the polynomial design's widths, delta bits, bias and dropped phase bits."""
import dataclasses, json, sys
import _paths  # noqa: F401,E402  (competition kickmix + verifiers)
import invsqrt_poly as ip

LIMIT = 0.985


def evaluate(d):
    try:
        r = ip.worst_ratio(d)
    except (ValueError, OverflowError):
        return None
    if not r < LIMIT:
        return dict(ratio=r, score=float('inf'))
    out = ip.score(d, r)
    out.pop('circuit')
    return out


def moves(d):
    w, db = list(d.widths), list(d.delta_bits)
    for i in range(len(w)):
        for dv in (-1, 1):
            nw = w.copy(); nw[i] += dv
            yield dataclasses.replace(d, widths=tuple(nw))
    for i in range(len(db)):
        for dv in (-1, -2, 1):
            nd = db.copy(); nd[i] = min(d.frac, nd[i] + dv)
            yield dataclasses.replace(d, delta_bits=tuple(nd))
    for dv in (1, -1):
        yield dataclasses.replace(d, drop=max(0, d.drop + dv))
    for dv in (2, -2, 8, -8):
        yield dataclasses.replace(d, c0_bias=d.c0_bias + dv)


def descend(d):
    best = evaluate(d)
    print('start', d, {k: round(v, 3) for k, v in best.items()}, flush=True)
    improved = True
    while improved:
        improved = False
        for cand in moves(d):
            if cand == d:
                continue
            r = evaluate(cand)
            if r is not None and r['score'] < best['score']:
                d, best, improved = cand, r, True
                print('  ', d, {k: round(v, 3) for k, v in best.items()}, flush=True)
                break
    return d, best


if __name__ == '__main__':
    d0 = eval(sys.argv[1], {'Design': ip.Design})
    d, best = descend(d0)
    print('FINAL', json.dumps(dict(design=dataclasses.asdict(d), **best)), flush=True)
