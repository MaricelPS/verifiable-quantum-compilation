"""Draws the overview figures: leaderboard history and the dqi Toffoli breakdown.

    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
S1, S2 = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID, BG = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG, 'axes.edgecolor': GRID,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.8, 'axes.axisbelow': True, 'font.size': 11, 'legend.frameon': False,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
})


def save(fig, name):
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)
    print('wrote', FIG / name)


def leaderboard():
    d = pd.read_csv(ROOT / 'results' / 'leaderboard.csv')
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.grid(axis='x', visible=False)
    bars = ax.bar(range(len(d)), d.public_score, width=0.6, color=S1, edgecolor=BG, linewidth=2)
    for rect, v in zip(bars, d.public_score):
        ax.annotate(f'{v:.3f}', (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 4),
                    textcoords='offset points', ha='center', color=INK2, fontsize=9)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([s.replace(' + ', '\n+ ').replace('chemistry ', 'chem ') for s in d.submission],
                       fontsize=8.5)
    ax.set_ylabel('public score (geometric mean of\nimprovements over the 3 tasks)')
    ax.set_ylim(0, 2.6)
    ax.set_title('Public leaderboard score, submission by submission')
    save(fig, 'leaderboard.png')


def dqi_breakdown():
    d = pd.read_csv(ROOT / 'dqi' / 'results' / 'stage_profile.csv')
    d = d[~d.stage.isin(['total'])]
    piv = d.pivot(index='stage', columns='decoder', values='toffolis')
    piv = piv.sort_values('reference')
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.grid(axis='y', visible=False)
    y = range(len(piv))
    h = 0.38
    ax.barh([i + h / 2 for i in y], piv.reference / 1e6, height=h, color=S1, edgecolor=BG, linewidth=1.5,
            label='reference decoder')
    ax.barh([i - h / 2 for i in y], piv.tower / 1e6, height=h, color=S2, edgecolor=BG, linewidth=1.5,
            label='tower-field decoder (this work)')
    for i, (r, t) in enumerate(zip(piv.reference, piv.tower)):
        if r > 2e5:
            ax.annotate(f'{r / 1e6:.2f} M → {t / 1e6:.2f} M' if r != t else f'{r / 1e6:.2f} M (unchanged)',
                        (max(r, t) / 1e6, i), xytext=(6, 0), textcoords='offset points', va='center',
                        color=INK2, fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(piv.index, fontsize=9.5)
    ax.set_xlabel('Toffoli gates (millions)')
    ax.set_xlim(0, 4.3)
    ax.set_title('dqi: where the Toffolis go (5.73 M → 4.15 M)')
    ax.legend(loc='lower right')
    save(fig, 'dqi_toffoli_breakdown.png')


if __name__ == '__main__':
    leaderboard()
    dqi_breakdown()
