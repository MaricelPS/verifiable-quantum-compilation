"""Toffolis per decoder stage, by ablation: rebuild with one stage replaced by a no-op and subtract.

Covers the reference decoder, the tower-field EEA decoder and the two-phase decoder.

    COMP_DIR=... python dqi/profile_stages.py results/stage_profile.csv
"""
import csv
import sys

import _paths  # noqa: F401
import kickmix as km
from dqi import dqi_reference_circuit as ref

import decoder_tower_eea as decoder

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




def profile_two_phase(out):
    """Ablation for the two-phase decoder: drop Forney, then phase 2 as well."""
    import decoder as two_phase

    def total(skip=()):
        cls = type('Ablated', (two_phase.TwoPhaseDecoder,), {name: (lambda self, *a, **k: None) for name in skip})
        b = km.CircuitBuilder()
        syn = [b.create_quantum_register(DEG, name=f'S{i}') for i in range(N)]
        err = [b.create_classical_register(DEG, name=f'E{j}') for j in range(M)]
        cls(b, syn, err).build()
        c = b.finish_circuit()
        return c.operation_statistics().num_toffoli_like, c.num_qubits

    full, qubits = total()
    no_forney, _ = total(['_forney'])
    no_p2, _ = total(['_forney', '_p2', '_p2_rev'])
    rows = [
        dict(decoder='two-phase', stage='total', toffolis=full, qubits=qubits),
        dict(decoder='two-phase', stage='Chien search and Forney', toffolis=full - no_forney, qubits=''),
        dict(decoder='two-phase', stage='key equation, phase 2 (replay)', toffolis=no_forney - no_p2, qubits=''),
        dict(decoder='two-phase', stage='key equation, phase 1 (divsteps)', toffolis=no_p2, qubits=''),
    ]
    with open(out, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=['decoder', 'stage', 'toffolis', 'qubits']).writerows(rows)
    for r in rows:
        print(r)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'results/stage_profile.csv'
    main(path)
    profile_two_phase(path)
