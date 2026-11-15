"""Toffolis per decoder stage, by ablation: rebuild with one stage replaced by a no-op and subtract.

    COMP_DIR=... python dqi/profile_stages.py results/stage_profile.csv
"""
import csv
import sys

import _paths  # noqa: F401
import kickmix as km
from dqi import dqi_reference_circuit as ref

import decoder

M, N, DEG = 4095, 70, 12
STAGES = {
    '_multiply_and_add_or_sub': 'EEA multiply sweep and quotients',
    '_chien_search_and_forney_mbuc': 'Chien search and Forney',
    '_shift_b_array_right': 'EEA shift of B',
    '_swap_a_and_b': 'EEA swap A <-> B',
    '_update_flag': 'EEA flag update',
    '_initial_shift_loop': 'initial shift',
    '_update_metadata': 'EEA degree bookkeeping',
    '_update_halting_counter': 'EEA halting counter',
}


def toffolis(base, skip=None):
    cls = type('Ablated', (base,), {skip: lambda self, *a, **k: None} if skip else {})
    c = decoder.build_circuit(M, N, DEG, cls)
    return c.operation_statistics().num_toffoli_like, c.num_qubits


def main(out):
    rows = []
    for label, base in (('reference', ref.ProosAndZalkaDecoder), ('tower', decoder.TowerDecoder)):
        total, qubits = toffolis(base)
        rows.append(dict(decoder=label, stage='total', toffolis=total, qubits=qubits))
        accounted = 0
        for name, title in STAGES.items():
            t, _ = toffolis(base, name)
            rows.append(dict(decoder=label, stage=title, toffolis=total - t, qubits=''))
            accounted += total - t
            print(label, title, total - t, flush=True)
        rows.append(dict(decoder=label, stage='other', toffolis=total - accounted, qubits=''))
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['decoder', 'stage', 'toffolis', 'qubits'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/stage_profile.csv')
