"""Parse Kaggle commit logs of the dfthcsos experiments into results/experiments.csv.

Each run contributes: its shape (R, B, C), the continuous two-body error and lambda_eff at the end of training,
the quantized errors at every precision that was built, and the official verifier's status and improvement.

    python scripts/parse_logs.py LOG [LOG ...] --out results/experiments.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

EPOCH = re.compile(r'epoch (\d+): ε_fro=([\d.]+)mHa, λ_eff=([\d.]+)Ha, Λ=([\d.]+)Ha')
BUILT = re.compile(r'builds/(.+?)_b(\d+)/dfthcsos: R,B,C=(\d+),(\d+),(\d+) brot=\d+ J=(\d+)\s+'
                   r'e1=([\d.]+) e2=([\d.]+) mHa lam_eff=([\d.]+)')
HEADER = re.compile(r'=== (\S+): seed (\S+), (\{.*\})')
TABLE = re.compile(r'^(\S+)\s+status=(\S+)\s+brot=(\S+)\s+improvement=(\S+)')
JNAME, JSTATUS = re.compile(r'"name": "(.+?)"'), re.compile(r'"status": "(.+?)"')
JIMPROV, JBROT = re.compile(r'"improvement": ([\d.]+)'), re.compile(r'"brot": (\d+)')
PREFIX = re.compile(r'^\s*[\d.]+s\s+\d+\s')     # "123.4s 56 " added by the Kaggle log viewer


def parse(path: Path, version: str, runs: dict) -> None:
    last_epoch, last_best, json_name = None, None, None
    for raw in path.read_text(errors='replace').splitlines():
        line = PREFIX.sub('', raw).strip()
        if m := HEADER.search(line):
            run = runs.setdefault(m[1], {'name': m[1]})
            run.update(version=version, seed=m[2], config=m[3])
            last_epoch = last_best = None
        elif m := EPOCH.search(line):
            last_epoch = m
            if 'B@epoch' in line:   # the saved factorization is the best one seen
                last_best = m
        elif m := BUILT.search(line):
            if 'eps_cont' in runs.get(m[1], {}):
                last_epoch = last_best = None
            run = runs.setdefault(m[1], {'name': m[1]})
            run.setdefault('version', version)
            run.update(R=int(m[3]), B=int(m[4]), C=int(m[5]))
            best = last_best or last_epoch
            if best is not None and 'eps_cont' not in run:
                run.update(epochs=int(last_epoch[1]) + 10, eps_cont=float(best[2]),
                           lam_eff_cont=float(best[3]), Lambda=float(best[4]))
            run[f'e2_b{m[2]}'] = float(m[8])
            run[f'e1_b{m[2]}'] = float(m[7])
            run[f'lam_b{m[2]}'] = float(m[9])
        elif m := JNAME.search(line):
            json_name = m[1]
        elif json_name and (m := JSTATUS.search(line)):
            runs.setdefault(json_name, {'name': json_name})['status'] = m[1]
        elif json_name and (m := JBROT.search(line)):
            runs.setdefault(json_name, {'name': json_name})['brot'] = int(m[1])
        elif json_name and (m := JIMPROV.search(line)):
            runs.setdefault(json_name, {'name': json_name})['improvement'] = float(m[1])
        elif m := TABLE.match(line):
            run = runs.setdefault(m[1], {'name': m[1]})
            run.setdefault('status', m[2])
            if m[2] == 'PASS':
                run.update(brot=int(m[3]), improvement=float(m[4]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+', help='VERSION=path pairs, in run order')
    ap.add_argument('--out', default='results/experiments.csv')
    args = ap.parse_args()
    runs: dict = {}
    for item in args.logs:
        version, path = item.split('=', 1)
        parse(Path(path), version, runs)
    cols = ['version', 'name', 'R', 'B', 'C', 'seed', 'config', 'epochs', 'eps_cont', 'lam_eff_cont', 'Lambda',
            *[f'{k}_b{b}' for b in (10, 11, 12, 13, 14) for k in ('e1', 'e2', 'lam')],
            'status', 'brot', 'improvement']
    rows = [r for r in runs.values() if 'R' in r]
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} runs -> {args.out}')


if __name__ == '__main__':
    main()
