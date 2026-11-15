"""Train DF-THC factorizations on a Kaggle GPU, build each submission and score it with the official verifier.

Run on Kaggle with the competition data and the starter kit dataset attached (GPU accelerator on),
with lab.py and FeMoCo76o113e_65_60_60_sorted.npz uploaded next to this file:

    python run_experiments.py                 # all configs in CONFIGS
    python run_experiments.py lambda_only R60 # only those configs

For each config:
  1. train with the starter kit's trainer (train_system), seeded from the importance-sorted baseline,
     so that shrinking R, B or C drops the least important rho / bases / copies first;
  2. build the dfthcsos/ folder at the lowest brot whose quantized errors pass (with weight refit);
  3. run the official verifier (dfthcsos/verify.py) and report cost and improvement over the baseline.

Results are appended to /kaggle/working/results.jsonl. Nothing is submitted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Kaggle paths by default; override with environment variables to run elsewhere.
COMPETITION = Path(os.environ.get('COMPETITION_DIR', '/kaggle/input/competitions/google-verifiable-quantum-circuit-compilation'))
VERIFIERS = Path(os.environ.get('VERIFIERS_DIR', COMPETITION / 'verifiers'))
KICKMIX = Path(os.environ.get('KICKMIX_DIR', COMPETITION / 'kickmix'))
KIT = Path(os.environ['KIT_DIR']) if 'KIT_DIR' in os.environ else next(Path('/kaggle/input').glob('**/starter_kit/starter')).parent
WORK = Path(os.environ.get('WORK_DIR', '/kaggle/working'))
SEED = HERE / 'FeMoCo76o113e_65_60_60_sorted.npz'
BASELINE_COST = 3355599588.3155365  # official verifier, sample submission

os.environ['KICKMIX_DIR'], os.environ['VERIFIERS_DIR'] = str(KICKMIX), str(VERIFIERS)
WORK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('QCACHE', str(WORK / 'femoco76_ref_cache.npz'))
paths = [str(KICKMIX), str(VERIFIERS), str(KIT), str(HERE)]
sys.path[:0] = paths
os.environ['PYTHONPATH'] = os.pathsep.join([*paths, os.environ.get('PYTHONPATH', '')])

import lab  # noqa: E402
from starter.task1_factorization.train_df_thc_qaggle import train_system  # noqa: E402

# epsilon_dfthc is the trainer's target for the *continuous* two-body error. The verifier checks the
# error after 12-bit rounding, which adds roughly 9 mHa in quadrature, so the continuous error must
# stay below ~17 mHa: sqrt(17**2 + 9.3**2) = 19.4 mHa. rho weights lambda_eff against the error penalty
# (the kit uses 1e-4 for FeMoCo); a larger rho pushes Lambda down harder.
CONFIGS = {
    # Tiny run to check the pipeline end to end (a few minutes even on CPU); not a real experiment.
    'smoke':      dict(rank=65, bases=60, copies=60, epochs=20, lr=1e-5, rho=1e-4, epsilon_dfthc=0.015),
    # Same shape, push Lambda down: safest first experiment.
    'lambda_only': dict(rank=65, bases=60, copies=60, epochs=20_000, lr=1e-4, rho=3e-4, epsilon_dfthc=0.015),
    # Fewer addresses (L = 76 + R*B): every 5 rho removed is ~8% cheaper at brot 12.
    'R60':        dict(rank=60, bases=60, copies=60, epochs=50_000, lr=1e-4, rho=1e-4, epsilon_dfthc=0.015),
    'R60_C75':    dict(rank=60, bases=60, copies=75, epochs=50_000, lr=1e-4, rho=1e-4, epsilon_dfthc=0.015),
    'R55_C80':    dict(rank=55, bases=60, copies=80, epochs=80_000, lr=1e-4, rho=1e-4, epsilon_dfthc=0.015),
    'R60_B55_C70': dict(rank=60, bases=55, copies=70, epochs=80_000, lr=1e-4, rho=1e-4, epsilon_dfthc=0.015),
}


def verify(folder: Path) -> dict:
    """The official verifier's summary for a dfthcsos/ folder (no salt)."""
    result = subprocess.run([sys.executable, str(VERIFIERS / 'dfthcsos' / 'verify.py'), str(folder)],
                            capture_output=True, text=True, check=False)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {'status': 'FAIL', 'error': (result.stdout + result.stderr)[-2000:]}


def run(name: str, cfg: dict, reference: dict) -> dict:
    t0 = time.time()
    out_npz = WORK / 'runs' / f'{name}.npz'
    if out_npz.exists():
        print(f'[{name}] reusing {out_npz}')
    else:
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        train_system('FeMoCo76o113e', seed_solution_file=str(SEED), output_file=str(out_npz),
                     print_interval=1000, **cfg)
    train_seconds = time.time() - t0

    record = {'name': name, **cfg, 'train_seconds': round(train_seconds)}
    # Try the cheapest precision first; keep the first one whose quantized errors pass locally. The errors are
    # deterministic (they do not depend on the grader's salt), so a small margin below 20 mHa is enough.
    for brot in (10, 11, 12, 13):
        folder = WORK / 'builds' / f'{name}_b{brot}' / 'dfthcsos'
        m = lab.build_folder(out_npz, folder, brot=brot, refit=True, r=reference)
        record[f'b{brot}_local'] = {k: m[k] for k in ('error_1', 'error_2', 'Lambda', 'E_gap', 'lambda_eff')}
        if m['error_1'] <= 0.0199 and m['error_2'] <= 0.0199 and m['Lambda'] > 49.0 and m['E_gap'] > 3.914898:
            summary = verify(folder)
            record.update(brot=brot, folder=str(folder), status=summary.get('status'),
                          cost=summary.get('score'), error=summary.get('error'))
            if summary.get('status') == 'PASS':
                record['improvement'] = BASELINE_COST / summary['score']
                record['resources'] = {k: summary['resources'][k] for k in ('toffoli', 'qubits', 'angles_toffoli')}
            break
    else:
        record['status'] = 'NO_PRECISION_PASSES'
    print(json.dumps(record, indent=1, default=float))
    with open(WORK / 'results.jsonl', 'a') as f:
        f.write(json.dumps(record, default=float) + '\n')
    return record


if __name__ == '__main__':
    selected = sys.argv[1:] or [name for name in CONFIGS if name != 'smoke']
    reference = lab.ref()
    for name in selected:
        run(name, CONFIGS[name], reference)
