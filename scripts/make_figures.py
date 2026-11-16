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
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.grid(axis='x', visible=False)
    bars = ax.bar(range(len(d)), d.public_score, width=0.6, color=S1, edgecolor=BG, linewidth=2)
    for rect, v in zip(bars, d.public_score):
        ax.annotate(f'{v:.3f}', (rect.get_x() + rect.get_width() / 2, v), xytext=(0, 4),
                    textcoords='offset points', ha='center', color=INK2, fontsize=9)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([s.replace('|', '\n') for s in d.label], fontsize=8.5)
    ax.set_ylabel('public score (geometric mean of\nimprovements over the 3 tasks)')
    ax.set_ylim(0, 3.8)
    ax.set_title('Public leaderboard score, submission by submission')
    save(fig, 'leaderboard.png')


def dqi_breakdown():
    d = pd.read_csv(ROOT / 'dqi' / 'results' / 'stage_profile.csv')
    d = d[d.stage != 'total']

    def group(stage):
        if stage.startswith('EEA') or stage.startswith('key equation'):
            return 'key-equation solver'
        if stage.startswith('Chien'):
            return 'Chien search and Forney'
        return 'other'

    d = d.assign(group=d.stage.map(group))
    piv = d.pivot_table(index='decoder', columns='group', values='toffolis', aggfunc='sum').fillna(0) / 1e6
    order = ['two-phase', 'tower', 'reference']
    labels = {'reference': 'reference decoder\n(synchronized EEA)', 'tower': 'tower-field arithmetic\n(same EEA)',
              'two-phase': 'two-phase solver\n+ tower field'}
    groups = [('key-equation solver', S1), ('Chien search and Forney', S2), ('other', '#b4b2a9')]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.grid(axis='y', visible=False)
    left = [0.0] * len(order)
    for name, color in groups:
        vals = [piv.loc[o, name] for o in order]
        ax.barh(range(len(order)), vals, left=left, height=0.55, color=color, edgecolor=BG, linewidth=2,
                label=name)
        left = [a + b for a, b in zip(left, vals)]
    for i, total in enumerate(left):
        ax.annotate(f'{total:.2f} M', (total, i), xytext=(6, 0), textcoords='offset points', va='center',
                    color=INK2, fontsize=9.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[o] for o in order], fontsize=9.5)
    ax.set_xlabel('Toffoli gates (millions)')
    ax.set_xlim(0, 6.6)
    ax.set_title('dqi: Toffolis by decoder stage')
    ax.legend(loc='lower right', fontsize=9)
    save(fig, 'dqi_toffoli_breakdown.png')


if __name__ == '__main__':
    leaderboard()
    dqi_breakdown()
