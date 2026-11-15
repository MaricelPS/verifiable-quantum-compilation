"""Local tools: cached reference, vector->angle encoder, fast evaluation."""
import sys, pathlib, numpy as np
import os
# Paths: set KICKMIX_DIR / VERIFIERS_DIR (on Kaggle, the competition data's kickmix/ and verifiers/),
# or QROOT with km/ and verifiers/ inside it.
ROOT = pathlib.Path(os.environ.get('QROOT', '/home/claude/kq'))
KICKMIX_DIR = pathlib.Path(os.environ.get('KICKMIX_DIR', ROOT/'km'))
VERIFIERS_DIR = pathlib.Path(os.environ.get('VERIFIERS_DIR', ROOT/'verifiers'))
sys.path[:0] = [str(KICKMIX_DIR), str(VERIFIERS_DIR)]
from dfthcsos import verify as V
from dfthcsos.scoreinfra import dfthc_math as M
from dfthcsos.scoreinfra.reference import load_reference, PACKAGED_FCIDUMP, load_fixed_shifts

CACHE = pathlib.Path(os.environ.get('QCACHE', '/tmp/femoco76_ref_cache.npz'))  # writable cache of the FCIDUMP
def ref():
    if CACHE.exists():
        z = np.load(CACHE)
        return {'orbitals':76,'electrons':113,'ecore':float(z['ecore']),'oei':z['oei'],
                'tei':z['tei'].astype(np.float64),'h1_thc':z['h1_thc']}
    r = load_reference(PACKAGED_FCIDUMP)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ecore=r['ecore'], oei=r['oei'], tei=r['tei'].astype(np.float32), h1_thc=r['h1_thc'])
    return r

def inputs(npz):
    p = {k: np.array(v) for k, v in np.load(npz).items() if k in ('U','W','H2_I','H1_I','Hsym_lower')}
    p.update(load_fixed_shifts())
    w, W, S, b1 = V.native_inputs(p)
    return p, w, W, S, b1

def target_vectors(U, w, W, S, b1, r):
    """Float vectors: N eigenvectors of h1_prime (with float X) + R*B vectors u."""
    R, B, N = U.shape
    u = U / np.linalg.norm(U, axis=2, keepdims=True)
    X = M.reconstruct_x(u, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1 = r['oei'] - 0.5*np.einsum('prrq->pq', r['tei'])
    h1p = M.build_h1_prime(h1, S, b1, 113, W, X)
    _, vecs = np.linalg.eigh(h1p)
    return np.vstack([vecs.T, u.reshape(R*B, N)])

def vectors_to_theta(V_, edges):
    """Exact inverse of the tree decoding: theta/pi in [0, 2) for every row."""
    v = np.array(V_, dtype=np.float64)
    th = np.zeros((v.shape[0], len(edges)))
    for t in range(len(edges)-1, -1, -1):
        p, c = edges[t]
        th[:, t] = np.arctan2(v[:, c], v[:, p])
        v[:, p] = np.hypot(v[:, p], v[:, c]); v[:, c] = 0
    return np.mod(th/np.pi, 2.0)

def quantize(theta_over_pi, brot):
    s = 1 << (brot-1)
    return np.mod(np.rint(theta_over_pi*s).astype(np.int64), 1 << brot)

def evaluate(words, edges, brot, w, W, S, b1, r):
    return M.evaluate_dfthc_angle_words(words, w, W, S, b1, r, brot=brot, tree_edges=edges, energy_upper=-22140.27335)

def h1_prime_from_words(words2, edges, brot, w, W, S, b1, r):
    """h1_prime with X decoded from the two-body words (as the verifier does)."""
    R, B, _ = w.shape
    u = M.reconstruct_u_vectors(words2 / (1 << (brot-1)), edges).reshape(R, B, -1)
    X = M.reconstruct_x(u, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    h1 = r['oei'] - 0.5*np.einsum('prrq->pq', r['tei'])
    return M.build_h1_prime(h1, S, b1, 113, W, X)

def encode(U, w, W, S, b1, r, edges, brot, two_body_words=None):
    """Full word table: rounded (or given) two-body words; one-body rows = eigenvectors of the quantized h1'."""
    R, B, N = U.shape
    if two_body_words is None:
        u = U / np.linalg.norm(U, axis=2, keepdims=True)
        two_body_words = quantize(vectors_to_theta(u.reshape(R*B, N), edges), brot)
    h1p = h1_prime_from_words(two_body_words, edges, brot, w, W, S, b1, r)
    _, vecs = np.linalg.eigh(h1p)
    one = quantize(vectors_to_theta(vecs.T, edges), brot)
    return np.vstack([one, two_body_words])

def refine_words(words, target, edges, brot, sweeps=4, steps=(1,)):
    """Coordinate descent: shift each word by +/-s if it increases |<u_hat, u>| (vectorized over rows)."""
    words = words.copy(); mod = 1 << brot; sc = 1 << (brot-1)
    tgt = target / np.linalg.norm(target, axis=1, keepdims=True)
    ov = lambda wd: np.abs(np.sum(M.reconstruct_u_vectors(wd/sc, edges)*tgt, axis=1))
    best = ov(words)
    for _ in range(sweeps):
        changed = 0
        for t in range(words.shape[1]):
            for s in steps:
                for d in (s, -s):
                    cand = words.copy(); cand[:, t] = (cand[:, t] + d) % mod
                    o = ov(cand); better = o > best + 1e-15
                    words[better, t] = cand[better, t]; best = np.where(better, o, best); changed += better.sum()
        if changed == 0: break
    return words, best

def refit_weights(u_float, u_q, w, ridge=0.0):
    """For each rho: w_new[:,c] = argmin ||sum_b w_b P^_b - X*_rc||_F, with P^_b = u^_b u^_b^T.
    Gram G_bb' = (u^_b.u^_b')^2 ; rhs_bc = u^_b^T X*_rc u^_b."""
    R, B, N = u_q.shape
    w_new = np.empty_like(w)
    for r_ in range(R):
        uq, uf = u_q[r_], u_float[r_]
        G = (uq @ uq.T)**2 + ridge*np.eye(B)
        proj = (uq @ uf.T)**2              # (B_q, B_f): (u^_b . u_b')^2
        rhs = proj @ w[r_]                 # sum_b' w_b'c (u^_b.u_b')^2
        w_new[r_] = np.linalg.solve(G, rhs)
    return w_new


def build_folder(factorization, out, brot=12, batch_size=None, refit=True, tree_edges=None, r=None, verbose=True):
    """Factorization .npz -> verifiable dfthcsos/ folder (circuits + submission.npz). Returns local metrics.

    batch_size defaults to the largest J that fits the 128-qubit cap with the reference QROM.
    """
    sys.path.insert(0, str(VERIFIERS_DIR / 'dfthcsos' / 'example_submission'))
    from angle_k_reference_circuit import build_angle_k_family, write_kmx_family
    r = r if r is not None else ref()
    p, w, W, S, b1 = inputs(factorization)
    if tree_edges is None:
        src = np.load(factorization)
        if 'tree_edges' in src.files:
            tree_edges = src['tree_edges']
        else:  # the baseline's log-depth tree, shipped next to lab.py
            tree_edges = np.load(pathlib.Path(__file__).resolve().parent / 'tree_edges.npy')
    edges = np.asarray(tree_edges, dtype=np.int64)
    U = p['U']; R, B, N = U.shape
    L = N + R * B; sel = (L - 1).bit_length()
    J = batch_size or min(75, (128 - sel - (sel - 1)) // brot)
    uf = U / np.linalg.norm(U, axis=2, keepdims=True)
    w2 = quantize(vectors_to_theta(uf.reshape(R * B, N), edges), brot)
    if refit:
        uq = M.reconstruct_u_vectors(w2 / (1 << (brot - 1)), edges).reshape(R, B, N)
        w = refit_weights(uf, uq, w)
    words = encode(U, w, W, S, b1, r, edges, brot, two_body_words=w2)
    m = evaluate(words, edges, brot, w, W, S, b1, r)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'submission.npz', U=U.astype(np.float32), W=w.astype(np.float32),
             H2_I=(W + w.sum(axis=1)).astype(np.float32), H1_I=np.float32(p['H1_I']),
             Hsym_lower=p['Hsym_lower'].astype(np.float32), tree_edges=edges, angle_words=words)
    write_kmx_family(build_angle_k_family(words, brot, J), out, format='kmb', overwrite=True)
    m = {k: float(v) for k, v in m.items()}; m.update(brot=brot, J=J, R=R, B=B, C=w.shape[2])
    if verbose:
        print(f"built {out}: R,B,C={R},{B},{w.shape[2]} brot={brot} J={J}  "
              f"e1={1e3*m['error_1']:.2f} e2={1e3*m['error_2']:.2f} mHa lam_eff={m['lambda_eff']:.4f}")
    return m
