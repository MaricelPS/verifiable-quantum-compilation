"""Quantization-aware weight refit: freeze the two-body vectors at their brot-bit decoded values and retrain
only the weights (W, H2_I) with the starter kit's trainer, so the trainer's error equals the verifier's error_2."""
import functools
import json
import os
from pathlib import Path

import numpy as np


def qat(name, source_npz, *, RX, TQ, TS, REF, WORK, brot=12, epochs=20_000, lr=1e-6, lr_final=1e-7,
        rho=1e-4, epsilon_dfthc=0.015):
    """Refit the weights of `source_npz` for its brot-bit vectors, then build, verify and log it as `name`."""
    import jax.numpy as jnp
    from dfthc import df_thc_sga

    lab = RX.lab
    src = np.load(source_npz)
    U = np.asarray(src['U'], dtype=np.float64)
    R, B, N = U.shape
    edges = np.load(Path(lab.__file__).resolve().parent / 'tree_edges.npy')
    u = U / np.linalg.norm(U, axis=2, keepdims=True)
    words = lab.quantize(lab.vectors_to_theta(u.reshape(R * B, N), edges), brot)
    u_q = lab.M.reconstruct_u_vectors(words / (1 << (brot - 1)), edges).reshape(R, B, N)

    # Trainer reference data, resolved the same way as train_system does.
    cfg = TQ.SYSTEMS['FeMoCo76o113e']
    fcidump = os.path.join(TQ.INTEGRALS_DIRECTORY, cfg['fcidump'])
    if not os.path.isfile(fcidump):
        fcidump = TQ.PACKAGED_FCIDUMP
    ref_t = TS.Initialize(fcidump, cfg['upper_energy'])

    dtype = df_thc_sga.dtype
    frozen = {**TQ.frozen_symmetry_params(ref_t), 'U': jnp.asarray(u_q, dtype=dtype)}
    params = {key: jnp.asarray(src[key], dtype=dtype) for key in ('W', 'H2_I', 'H1_I', 'Hsym_lower')}
    params.update(frozen)
    solution = TS.Train(generating_set='Qaggle', method='DF-THC-SOS', epochs=epochs, learning_rate=lr,
                        learning_rate_final=lr_final, rho=rho, params=params, ref=ref_t,
                        print_interval=max(10, epochs // 20), epsilon_dfthc=epsilon_dfthc,
                        frozen_params=frozen, **TQ.FEMOCO_FLOORS)
    best = {key: np.asarray(value) for key, value in solution['params_best'].items()}

    out_npz = WORK / 'runs' / f'{name}.npz'
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, U=u_q, W=best['W'], H2_I=best['H2_I'], H1_I=best['H1_I'], Hsym_lower=best['Hsym_lower'])

    # The vectors are already on the brot-bit grid, so the build reproduces the same words; no extra refit.
    record = {'name': name, 'source': str(source_npz), 'qat_epochs': epochs, 'lr': lr, 'lr_final': lr_final}
    folder = WORK / 'builds' / f'{name}_b{brot}' / 'dfthcsos'
    m = lab.build_folder(out_npz, folder, brot=brot, refit=False, r=REF)
    record[f'b{brot}_local'] = {k: m[k] for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')}
    if m['error_1'] <= 0.0199 and m['error_2'] <= 0.0199 and m['Lambda'] > 49.0 and m['E_gap'] > 3.914898:
        summary = RX.verify(folder)
        record.update(brot=brot, folder=str(folder), status=summary.get('status'),
                      cost=summary.get('score'), error=summary.get('error'))
        if summary.get('status') == 'PASS':
            record['improvement'] = RX.BASELINE_COST / summary['score']
    else:
        record['status'] = 'NO_PRECISION_PASSES'
    print(json.dumps(record, indent=1, default=float))
    with open(WORK / 'results.jsonl', 'a') as f:
        f.write(json.dumps(record, default=float) + '\n')
    return record
