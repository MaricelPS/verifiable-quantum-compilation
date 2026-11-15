"""Initialize a DF-THC factorization (R, B, C) from the eigendecomposition of the two-body tensor.

The target in pair space is G[pq, rs] = (pq|rs) + (S[pq] d_rs + d_pq S[rs]) / 2, so that sum_l x_l x_l^T ~ G gives
the verifier's error_2 residual. Its R*C leading eigencomponents become the X matrices, split into R groups of C;
each group gets B shared vectors (the leading eigenvectors of its matrices) and least-squares weights.
"""
import numpy as np


def pair_target(tei, S):
    n = tei.shape[0]
    G = tei.reshape(n * n, n * n).copy()
    eye = np.eye(n).reshape(-1)
    G += 0.5 * (np.outer(S.reshape(-1), eye) + np.outer(eye, S.reshape(-1)))
    return 0.5 * (G + G.T)


def packed_basis(n):
    """Orthonormal map between symmetric n x n matrices and packed vectors (off-diagonals scaled by sqrt 2)."""
    rows, cols = np.tril_indices(n)
    scale = np.where(rows == cols, 1.0, np.sqrt(2.0))
    return rows, cols, scale


def to_packed(M, rows, cols, scale):
    return M[..., rows, cols] * scale


def from_packed(v, n, rows, cols, scale):
    M = np.zeros(v.shape[:-1] + (n, n))
    M[..., rows, cols] = v / scale
    M[..., cols, rows] = v / scale
    return M


def initialize(tei, S, R, B, C, group='round_robin'):
    n = tei.shape[0]
    rows, cols, scale = packed_basis(n)
    G = pair_target(tei, S)
    # Packed (symmetric) target: G is symmetric in (p,q) and (r,s), so project onto symmetric pairs.
    Gm = G.reshape(n, n, n, n)
    Gp = Gm[rows, cols][:, rows, cols] * np.outer(scale, scale)
    evals, evecs = np.linalg.eigh(Gp)
    order = np.argsort(-evals)[: R * C]
    lam = np.clip(evals[order], 0, None)
    X = from_packed((evecs[:, order] * np.sqrt(lam)).T, n, rows, cols, scale)       # (R*C, n, n)
    groups = (np.arange(R * C) % R) if group == 'round_robin' else (np.arange(R * C) // C)
    U = np.zeros((R, B, n)); W = np.zeros((R, B, C))
    for r in range(R):
        Xg = X[groups == r]                                                          # (C, n, n)
        # Shared basis: leading eigenvectors of sum_c X_c^2 (captures the subspaces all copies use).
        ev, vec = np.linalg.eigh(np.einsum('cpq,cqs->ps', Xg, Xg))
        base = vec[:, np.argsort(-ev)]
        cand = [base[:, i] for i in range(n)]
        # Fill up to B vectors with the leading eigenvectors of each X_c not yet spanned well.
        for c in np.argsort(-np.linalg.norm(Xg, axis=(1, 2))):
            e, v = np.linalg.eigh(Xg[c])
            for i in np.argsort(-np.abs(e)):
                if len(cand) >= B:
                    break
                cand.append(v[:, i])
            if len(cand) >= B:
                break
        u = np.array(cand[:B])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        # Least squares: X_c ~ sum_b w_bc u_b u_b^T  ->  Gram (u_b.u_b')^2 w = (u_b^T X_c u_b)
        gram = (u @ u.T) ** 2 + 1e-9 * np.eye(B)
        rhs = np.einsum('bp,cpq,bq->bc', u, Xg, u)
        U[r] = u
        W[r] = np.linalg.solve(gram, rhs)
    return U, W
