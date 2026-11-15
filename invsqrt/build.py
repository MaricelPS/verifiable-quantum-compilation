"""Builds the submitted invsqrt circuit (x4.81 over the reference) and writes it as .kmb.

    COMP_DIR=/path/to/competition python invsqrt/build.py invsqrt.kmb
"""
import sys

import _paths  # noqa: F401
import invsqrt_poly as ip

BEST = ip.Design(k=8, degree=2, widths=(28, 21, 12), delta_bits=(17, 21), drop=2, bitnorm=True)

if __name__ == '__main__':
    result = ip.score(BEST)
    circuit = result.pop('circuit')
    print({k: v for k, v in result.items()})
    circuit.to_file(sys.argv[1] if len(sys.argv) > 1 else 'invsqrt.kmb', format='kmb')
