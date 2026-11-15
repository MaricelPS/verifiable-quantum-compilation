"""Puts the competition's kickmix module and verifiers on sys.path.

Set COMP_DIR to the unpacked competition data (the folder holding `kickmix/` and `verifiers/`).
"""
import os
import sys
from pathlib import Path

COMP_DIR = Path(os.environ.get('COMP_DIR', Path(__file__).resolve().parents[1] / 'competition'))
for sub in ('kickmix', 'verifiers'):
    path = str(COMP_DIR / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
