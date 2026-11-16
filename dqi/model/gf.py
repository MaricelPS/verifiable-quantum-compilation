"""Classical GF(2^12) helpers matching kickmix's field (modulus 0x1053, primitive element 2)."""
import numpy as np
MOD, DEG, ORDER = 0x1053, 12, 4095
EXP = np.zeros(2 * ORDER, dtype=np.int64); LOG = np.zeros(4096, dtype=np.int64)
x = 1
for i in range(ORDER):
    EXP[i] = x; LOG[x] = i
    x <<= 1
    if x & 0x1000: x ^= MOD
EXP[ORDER:] = EXP[:ORDER]
def mul(a, b): return 0 if a == 0 or b == 0 else int(EXP[LOG[a] + LOG[b]])
def inv(a): return 0 if a == 0 else int(EXP[(ORDER - LOG[a]) % ORDER])
def div(a, b): return mul(a, inv(b))
def pw(k): return int(EXP[k % ORDER])
def poly_eval(p, xv):
    r = 0
    for c in reversed(p): r = mul(r, xv) ^ c
    return r

def sample(rng, m=4095, n=70, weight=None, probs=None):
    k = weight if weight is not None else int(rng.choice(len(probs), p=probs))
    pos = rng.choice(m, size=k, replace=False); vals = rng.integers(1, 4096, size=k)
    S = []
    for i in range(n):
        s = 0
        for p, v in zip(pos, vals): s ^= mul(int(v), pw((i + 1) * int(p)))
        S.append(s)
    # sigma(x) = prod (1 - alpha^p x)
    sig = [1]
    for p in pos:
        a = pw(int(p)); new = sig + [0]
        for i in range(len(sig)): new[i + 1] ^= mul(sig[i], a)
        sig = new
    return S, sig, pos, vals
