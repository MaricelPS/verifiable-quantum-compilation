"""Draw the README figures from results/experiments.csv.

    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
S1, S2, S3 = '#2a78d6', '#eb6834', '#1baf7a'              # categorical slots 1-3
INK, INK2, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
GATE = 20.0                                               # verifier gate on error_1 and error_2 (mHa)

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'axes.edgecolor': GRID,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'axes.axisbelow': True, 'font.size': 11, 'legend.frameon': False,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})


def gate(ax, label='verifier gate: 20 mHa', side='right'):
    ax.axhline(GATE, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    x, dx = (1, -4) if side == 'right' else (0, 4)
    ax.annotate(label, (x, GATE), xycoords=('axes fraction', 'data'), xytext=(dx, 4),
                textcoords='offset points', ha=side, color=INK2, fontsize=9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print('wrote', FIG / name)


def copies(d):
    """Error against the number of copies C for the (8, 143, C) shape (final training run of each C)."""
    rows = d[d.name.isin(['S8', 'S80', 'S100', 'S143', 'S200'])].sort_values('C')
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for col, color, label in (('eps_cont', S1, 'continuous (before rounding)'),
                              ('e2_b13', S2, 'after rounding to 13 bits')):
        ax.plot(rows.C, rows[col], color=color, lw=2, marker='o', ms=8, mec=BG, mew=2, label=label)
    for c, y in zip(rows.C, rows.e2_b13):
        ax.annotate(f'{y:.1f}', (c, y), xytext=(0, 9), textcoords='offset points', ha='center',
                    color=INK2, fontsize=9)
    best = d[d.name == 'S200_rho3'].iloc[0]
    ax.plot([best.C], [best.e2_b13], ls='none', marker='D', ms=9, color=S3, mec=BG, mew=2,
            label='C = 200 with rho 3e-5, 13 bits: passes')
    ax.annotate(f'{best.e2_b13:.1f}', (best.C, best.e2_b13), xytext=(-12, -4), textcoords='offset points',
                ha='right', color=INK2, fontsize=9)
    gate(ax)
    ax.set_xticks(rows.C)
    ax.set_xlabel('copies C   (shape R = 8, B = 143)')
    ax.set_ylabel('two-body error (mHa)')
    ax.set_ylim(15, 40)
    ax.set_title('More copies help; a smaller rho gets under the gate')
    ax.legend(loc='upper right')
    save(fig, 'error_vs_copies.png')


def tradeoff(d):
    """Two-body error at 12 bits against lambda_eff for every trained factorization (no weight-only refits)."""
    t = d[~d.name.str.contains(r'_q1\d|stage1|C70_b12')].dropna(subset=['e2_b12'])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    fam = t.R >= 50
    ax.scatter(t[fam].lam_b12, t[fam].e2_b12, s=70, color=S1, edgecolor=BG, lw=2, zorder=3,
               label='R = 50-65, from the baseline')
    ax.scatter(t[~fam].lam_b12, t[~fam].e2_b12, s=70, color=S2, edgecolor=BG, lw=2, zorder=3,
               label='R = 8, B = 143, own initialization')
    for _, r in t.iterrows():
        if r['name'] in ('R55_B55_C70', 'R55_B50', 'R55_B52_C75', 'S8', 'S100', 'S200', 'R55_decay'):
            ax.annotate(f"({r.R},{r.B},{r.C})", (r.lam_b12, r.e2_b12), xytext=(7, -3),
                        textcoords='offset points', fontsize=8.5, color=INK2)
    gate(ax)
    ax.set_xlabel('λ_eff (Ha), lower is cheaper')
    ax.set_ylabel('two-body error at 12 bits (mHa)')
    ax.set_title('Cheaper shapes trade normalization against error')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2)
    save(fig, 'error_vs_lambda.png')


def bits(d):
    """Two-body error against the rotation precision for three factorizations."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, color, label in (('R55_B55_C70', S1, '(55, 55, 70): passes at 12 bits'),
                               ('S200', S2, '(8, 143, 200)'),
                               ('S8', S3, '(8, 143, 44)')):
        r = d[d.name == name].iloc[0]
        xs = [b for b in (10, 11, 12, 13) if pd.notna(r.get(f'e2_b{b}'))]
        ys = [r[f'e2_b{b}'] for b in xs]
        ax.plot(xs, ys, color=color, lw=2, marker='o', ms=8, mec=BG, mew=2, label=label)
        ax.axhline(r.eps_cont, color=color, lw=1, ls=':', alpha=0.9)
    gate(ax, side='left')
    ax.plot([], [], color=MUTED, lw=1, ls=':', label='error before rounding (same color)')
    ax.set_xticks([10, 11, 12, 13])
    ax.set_xlabel('bits per rotation angle (brot)')
    ax.set_ylabel('two-body error (mHa)')
    ax.set_title('Each bit removed roughly doubles the rounding error')
    ax.legend(loc='upper right')
    save(fig, 'error_vs_bits.png')


if __name__ == '__main__':
    FIG.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / 'results' / 'experiments.csv')
    copies(data)
    tradeoff(data)
    bits(data)
