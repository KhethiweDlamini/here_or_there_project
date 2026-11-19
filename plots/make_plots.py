"""Five slide-ready (16:9) charts of the Pause & Find round 3 results.

The numbers are copied from the executed notebook outputs of Kaggle run "Version 5" (git commit 8bdb580,
round 3: OWLv2 + contrast boost + curated close-ups + best single example + usual-zone prior, tau 5.284).
Cells: C3/C4 (Session 1), D4 (Session 2 answers), D5 (frame-level), D7 (figures, hypotheses).
If you re-run the notebook, update DATA below and run:  python plots/make_plots.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- data (round 3, Kaggle Version 5)
DATA = {
    # C4 (S1 at frozen tau) and D5 / D7 zone table (S2: 24 of 238 present pairs ranked first, 9 named at tau)
    'gap': [  # metric, Session 1 %, Session 2 %
        ('Right zone ranked first\n(ignoring the threshold)', 89, 100 * 24 / 238),
        ('Sighted in the right zone\n(at the frozen threshold)', 36, 100 * 9 / 238),
        ('False sightings\n(object absent, system said "here")', 0, 0),
    ],
    # D4: right zone named, questions where the object had been seen (n = 48)
    'methods': [('Random zone', 7), ('Ours (frozen method)', 5), ('Prior only (usual S1 zone)', 3),
                ('Whole photo, no zones', 0)],
    'n_seen': 48,
    # D7 figure 1: right / wrong zone named / said "not seen" (it was there)
    'breakdown': [('Random zone', 7, 41, 0), ('Prior only (usual S1 zone)', 3, 45, 0),
                  ('Whole photo, no zones', 0, 6, 42), ('Ours (frozen method)', 5, 24, 19)],
    'false_sightings': (0, 8), 'never_in_view': (2, 2),
    # D7 zone table: present frame pairs, ranked first ignoring tau, named at tau (zones 5, 6, 8 had none)
    'zones': [('Counter', 127, 15, 0), ('Armchair', 41, 0, 0), ('Floor by armchair', 32, 0, 0),
              ('Corner shelf', 19, 9, 9), ('Centre floor', 19, 0, 0)],
    # D7 hypotheses: (method, right, questions)
    'H1': [('Random zone', 0, 3), ('Prior only', 3, 3), ('Whole photo', 0, 3), ('Ours', 3, 3)],
    'H2': [('Random zone', 7, 45), ('Prior only', 0, 45), ('Whole photo', 0, 45), ('Ours', 2, 45)],
}
SOURCE = 'Source: pause_and_find.ipynb, round 3 (Kaggle run v5). Session 2 is held out: 60 frames, 50 questions.'

# ---------------------------------------------------------------- design tokens (dataviz reference palette, light)
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS, TRACK = '#e1e0d9', '#c3c2b7', '#ecebe6'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'       # categorical slots 1-3 (validated all-pairs)
BLUE_LIGHT = '#86b6ef'                                     # blue ramp step 250
GREY_BAR, NOT_SEEN = '#b5b4ac', '#d9d8d2'
CHANCE = 12.5

plt.rcParams.update({
    'font.family': ['Segoe UI', 'Arial', 'DejaVu Sans'], 'font.size': 13,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': INK,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
})
SEMI = {'family': ['Segoe UI Semibold', 'Segoe UI', 'Arial', 'DejaVu Sans']}


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def slide(title, subtitle, rect=(0.25, 0.13, 0.69, 0.55)):
    fig = plt.figure(figsize=(13.333, 7.5))
    fig.text(0.05, 0.925, title, fontsize=27, color=INK, va='top', fontdict=SEMI)
    fig.text(0.05, 0.845, subtitle, fontsize=14, color=INK2, va='top', linespacing=1.4)
    fig.text(0.05, 0.035, SOURCE, fontsize=10, color=MUTED)
    ax = fig.add_axes(rect) if rect else None
    return fig, ax


def clean(ax, grid='x'):
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(AXIS)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=12)


def pct_axis(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))


def chance_line(ax, top, label='chance 12.5%'):
    ax.axvline(CHANCE, color=MUTED, lw=1.2, zorder=1)
    ax.text(CHANCE + 0.8, top, label, fontsize=11, color=INK2, va='bottom')


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig); print('wrote', OUT / name)


# ---------------------------------------------------------------- 1. the calibration-to-held-out gap
def plot_gap():
    fig, ax = slide('89% on calibration, 10% on new data',
                    'The round 3 detector put the right zone first on Session 1, which also chose its settings.\n'
                    'On the held-out Session 2 it dropped to chance level. It never reported an object that was absent.')
    clean(ax)
    rows = DATA['gap']; y = np.arange(len(rows))[::-1] * 1.0
    for yi, (name, s1, s2) in zip(y, rows):
        if s1 != s2:
            ax.annotate('', xy=(s2 + 1.6, yi), xytext=(s1 - 1.6, yi),
                        arrowprops=dict(arrowstyle='-|>', color=AXIS, lw=2.5, mutation_scale=22), zorder=2)
            ax.text((s1 + s2) / 2, yi + 0.2, f'−{s1 - s2:.0f} pts', ha='center', fontsize=15, color=INK, fontdict=SEMI)
            ax.scatter([s1], [yi], s=360, color=BLUE, edgecolor=SURFACE, lw=2.5, zorder=4)
            ax.scatter([s2], [yi], s=360, color=ORANGE, edgecolor=SURFACE, lw=2.5, zorder=4)
            ax.text(s1 + 2.6, yi - 0.03, f'{s1:.0f}%', va='center', fontsize=17, color=INK, fontdict=SEMI)
            ax.text(s2 - 2.6, yi - 0.03, f'{s2:.0f}%', va='center', ha='right', fontsize=17, color=INK, fontdict=SEMI)
        else:
            ax.scatter([s1], [yi], s=360, color=BLUE, edgecolor=SURFACE, lw=2.5, zorder=4)
            ax.scatter([s2], [yi], s=150, color=ORANGE, edgecolor=SURFACE, lw=2, zorder=5)
            ax.text(s1 + 2.6, yi - 0.03, '0% in both sessions', va='center', fontsize=17, color=INK, fontdict=SEMI)
    chance_line(ax, y.max() + 0.45)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=13, color=INK)
    ax.set_xlim(-3, 100); ax.set_ylim(y.min() - 0.5, y.max() + 0.75); pct_axis(ax)
    ax.scatter([], [], s=160, color=BLUE, label='Session 1 (calibration, 20 frames)')
    ax.scatter([], [], s=160, color=ORANGE, label='Session 2 (held out, 60 frames)')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=12, handletextpad=0.3)
    save(fig, '01_calibration_vs_heldout.png')


# ---------------------------------------------------------------- 2. methods vs chance
def plot_methods():
    fig, ax = slide('On held-out questions, nothing beat a random guess',
                    'Right zone named for the 48 Session 2 questions where the object had already been in view.\n'
                    'Whiskers show the 95% range. Every range overlaps chance.')
    clean(ax)
    rows = DATA['methods']; n = DATA['n_seen']; y = np.arange(len(rows))[::-1]
    for yi, (name, k) in zip(y, rows):
        ours = name.startswith('Ours'); lo, hi = wilson(k, n); v = 100 * k / n
        ax.barh(yi, max(v, 0.4), height=0.56, color=BLUE if ours else GREY_BAR, zorder=2)
        ax.plot([100 * lo, 100 * hi], [yi, yi], color=INK2, lw=1.6, zorder=3)
        for e in (lo, hi): ax.plot([100 * e] * 2, [yi - 0.13, yi + 0.13], color=INK2, lw=1.6, zorder=3)
        ax.text(100 * hi + 1.2, yi, f'{v:.0f}%', va='center', fontsize=20, color=INK, fontdict=SEMI)
        ax.text(100 * hi + 1.2 + 5.2, yi, f'{k} of {n}', va='center', fontsize=13, color=INK2)
    chance_line(ax, y.max() + 0.42)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=14)
    for t, (name, _) in zip(ax.get_yticklabels(), rows):
        if name.startswith('Ours'): t.set_fontproperties(SEMI['family'][0]); t.set_fontsize(14)
    ax.set_xlim(0, 50); ax.set_ylim(-0.6, y.max() + 0.75); pct_axis(ax)
    save(fig, '02_methods_vs_chance.png')


# ---------------------------------------------------------------- 3. what each method answered
def plot_breakdown():
    fig, ax = slide('Guessing sends you to the wrong spot 85–94% of the time. Ours: 50%.',
                    'Answers to the 48 questions where the object had been in view. When unsure, ours\n'
                    'often says "not seen" instead of sending you to look in the wrong place.',
                    rect=(0.22, 0.13, 0.49, 0.55))
    clean(ax, grid=None); ax.spines['bottom'].set_visible(False); ax.set_xticks([])
    parts = [('Right zone', BLUE, 'white'), ('Wrong zone named', ORANGE, 'white'),
             ('Said "not seen" (it was there)', NOT_SEEN, INK)]
    rows = DATA['breakdown']; y = np.arange(len(rows))[::-1]
    for yi, (name, *vals) in zip(y, rows):
        tot, left = sum(vals), 0.0
        for (lab, col, tc), v in zip(parts, vals):
            w = 100 * v / tot
            if w:
                ax.barh(yi, w, left=left, height=0.62, color=col, edgecolor=SURFACE, lw=2.5, zorder=2)
                if w >= 8: ax.text(left + w / 2, yi, f'{w:.0f}%', ha='center', va='center', fontsize=15, color=tc, fontdict=SEMI)
            left += w
    ax.set_yticks(y, [r[0] for r in rows], fontsize=14)
    for t in ax.get_yticklabels():
        if t.get_text().startswith('Ours'): t.set_fontproperties(SEMI['family'][0]); t.set_fontsize(14)
    ax.set_xlim(0, 100); ax.set_ylim(-0.55, y.max() + 0.55)
    ax.legend([Patch(color=c) for _, c, _ in parts], [p[0] for p in parts], loc='lower left',
              bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=12,
              handlelength=1.1, handletextpad=0.4, columnspacing=1.2)
    # stat tiles: the safety side of ours
    tiles = [(f'{DATA["false_sightings"][0]} of {DATA["false_sightings"][1]}', 'false sightings', 'questions where the object was\nnot in view at that moment'),
             (f'{DATA["never_in_view"][0]} of {DATA["never_in_view"][1]}', 'never-in-view questions', 'answered "not seen", where random\nand prior-only named a zone')]
    for j, (big, head, sub) in enumerate(tiles):
        y0 = 0.47 - j * 0.30
        fig.patches.append(FancyBboxPatch((0.745, y0), 0.205, 0.255, boxstyle='round,pad=0,rounding_size=0.012',
                                          transform=fig.transFigure, facecolor='#f3f2ee', edgecolor='none'))
        fig.text(0.765, y0 + 0.215, 'OURS', fontsize=10.5, color=MUTED, va='top', fontdict=SEMI)
        fig.text(0.765, y0 + 0.17, big, fontsize=32, color=INK, va='top', fontdict=SEMI)
        fig.text(0.765, y0 + 0.085, head, fontsize=13, color=INK, va='top', fontdict=SEMI)
        fig.text(0.765, y0 + 0.052, sub, fontsize=10.5, color=INK2, va='top', linespacing=1.3)
    save(fig, '03_answer_breakdown.png')


# ---------------------------------------------------------------- 4. where it found anything
def plot_zones():
    fig, ax = slide('All 9 correct sightings in Session 2 came from one zone',
                    'Session 2 frames × objects, by the zone the object was in. The counter held over half the\n'
                    'objects; the right zone often scored highest there, but never cleared the threshold.')
    clean(ax)
    rows = DATA['zones']; y = np.arange(len(rows))[::-1]
    for yi, (name, present, ranked, named) in zip(y, rows):
        ax.barh(yi, present, height=0.6, color=TRACK, zorder=2)
        if ranked: ax.barh(yi, ranked, height=0.6, color=BLUE_LIGHT, edgecolor=SURFACE, lw=2, zorder=3)
        if named: ax.barh(yi, named, height=0.6, color=BLUE, edgecolor=SURFACE, lw=2, zorder=4)
        note = (f'{named} of {present} sighted in the right zone' if named else
                f'{ranked} of {present} ranked first, 0 sighted' if ranked else f'0 of {present} found')
        ax.text(present + 1.5, yi, note, va='center', fontsize=13, color=INK if (named or ranked) else INK2,
                fontdict=SEMI if named else None)
    ax.legend([Patch(color=TRACK), Patch(color=BLUE_LIGHT), Patch(color=BLUE)],
              ['Object present', 'Right zone ranked first, below the threshold', 'Sighted in the right zone'],
              loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=3,
              frameon=False, fontsize=12, handlelength=1.1, handletextpad=0.4)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=14)
    for t in ax.get_yticklabels():
        if t.get_text() == 'Corner shelf': t.set_fontproperties(SEMI['family'][0]); t.set_fontsize(14)
    ax.set_xlim(0, 185); ax.set_ylim(-0.55, y.max() + 0.55)
    ax.set_xlabel('frame × object pairs (carried and person-blocked left out). Under-counter cupboard, glass shelves '
                  'and floor by cupboards held no objects.', fontsize=11, labelpad=8, color=MUTED)
    save(fig, '04_zones_where_found.png')


# ---------------------------------------------------------------- 5. hypotheses H1 vs H2
def plot_hypotheses():
    fig, _ = slide('Clear view, usual place: 3 of 3 right. Everything else: 2 of 45.',
                   'Right zone named, by question group (proposal section 3). Whiskers show the 95% range.',
                   rect=None)
    panels = [('H1', 'Clear view and usual place', 'visible, usual zone, not keys/phone', 'too few questions to tell'),
              ('H2', 'Everything else', 'keys/phone, hidden, far shelf, unusual zone', 'supported (guessing matches or beats ours)')]
    for j, (key, head, sub, verdict) in enumerate(panels):
        x0 = 0.15 + j * 0.44
        ax = fig.add_axes((x0, 0.13, 0.33, 0.5)); clean(ax)
        rows = DATA[key]; y = np.arange(len(rows))[::-1]
        for yi, (name, k, n) in zip(y, rows):
            lo, hi = wilson(k, n); v = 100 * k / n; ours = name == 'Ours'
            ax.barh(yi, max(v, 0.6), height=0.56, color=BLUE if ours else GREY_BAR, zorder=2)
            ax.plot([100 * lo, 100 * hi], [yi, yi], color=INK2, lw=1.5, zorder=3)
            for e in (lo, hi): ax.plot([100 * e] * 2, [yi - 0.12, yi + 0.12], color=INK2, lw=1.5, zorder=3)
            tx = min(100 * hi + 2, 103)
            ax.text(tx, yi, f'{k}/{n}', va='center', fontsize=15, color=INK, fontdict=SEMI if ours else None)
        chance_line(ax, y.max() + 0.42, 'chance')
        ax.set_yticks(y, [r[0] for r in rows], fontsize=13)
        for t in ax.get_yticklabels():
            if t.get_text() == 'Ours': t.set_fontproperties(SEMI['family'][0]); t.set_fontsize(13)
        ax.set_xlim(0, 115); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_ylim(-0.6, y.max() + 0.75); pct_axis(ax)
        n_q = rows[0][2]
        fig.text(x0 - 0.1, 0.735, f'{key} · {head}', fontsize=17, color=INK, fontdict=SEMI)
        fig.text(x0 - 0.1, 0.70, f'{n_q} questions: {sub}', fontsize=12, color=INK2)
        fig.text(x0 - 0.1, 0.665, f'Verdict: {verdict}', fontsize=12, color=INK, fontdict=SEMI)
    save(fig, '05_hypotheses_h1_h2.png')


if __name__ == '__main__':
    plot_gap(); plot_methods(); plot_breakdown(); plot_zones(); plot_hypotheses()
