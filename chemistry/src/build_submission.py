"""Build a ready-to-verify dfthcsos/ folder from a factorization.

Usage (set KICKMIX_DIR and VERIFIERS_DIR, or QROOT; lab.py alongside):
    python build_submission.py FACTORIZATION.npz OUT_DIR --brot 12 --batch-size 8 [--refit]

Steps: float vectors -> angles on the tree -> quantization to brot bits ->
(optional) refit of the weights w to the quantized vectors -> one-body eigenvectors
computed from h1' with the quantized X -> reference QROM circuits -> submission.npz.
"""
import argparse, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lab

ap = argparse.ArgumentParser()
ap.add_argument('factorization'); ap.add_argument('out')
ap.add_argument('--brot', type=int, default=12); ap.add_argument('--batch-size', type=int, default=None, help='J (default: largest that fits 128 qubits)')
ap.add_argument('--tree', default=None, help='npz/npy with tree_edges (default: the one in the factorization file)')
ap.add_argument('--refit', action='store_true')
a = ap.parse_args()

tree = None
if a.tree:
    t = np.load(a.tree); tree = t if isinstance(t, np.ndarray) else t['tree_edges']
lab.build_folder(a.factorization, a.out, brot=a.brot, batch_size=a.batch_size, refit=a.refit, tree_edges=tree)
