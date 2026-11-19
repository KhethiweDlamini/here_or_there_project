"""Five slide-ready (16:9) charts of the Pause & Find round 3 results.

The numbers are copied from the executed notebook outputs of Kaggle run "Version 5" (git commit 8bdb580,
round 3: OWLv2 + contrast boost + curated close-ups + best single example + usual-zone prior, tau 5.284).
Cells: C3/C4 (Session 1), D4 (Session 2 answers), D5 (frame-level), D7 (figures, hypotheses).
If you re-run the notebook, update DATA below and run:  python plots/make_plots.py

Theme: matches the deck's "Session 2 results" slide (white surface, Noto Serif titles, Lato text, navy / ochre on
cool grey, hatched ochre for the OWLv2 method). Fonts are in plots/fonts (SIL Open Font License).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

OUT = Path(__file__).resolve().parent
for f in (OUT / 'fonts').glob('*.ttf'):
    font_manager.fontManager.addfont(str(f))

# ---------------------------------------------------------------- data (round 3, Kaggle Version 5)
OURS = 'OWLv2 + usual-place prior'
PRIOR = 'Prior only (usual place)'
DATA = {
    # C4 (S1 at frozen tau) and D5 / D7 zone table (S2: 24 of 238 present pairs ranked first, 9 named at tau)
    'gap': [  # metric, Session 1 %, Session 2 %
        ('Right zone ranked first\n(ignoring the threshold)', 89, 100 * 24 / 238),
        ('Sighted in the right zone\n(at the frozen threshold)', 36, 100 * 9 / 238),
        ('False sightings\n(object absent, system said "here")', 0, 0),
    ],
    # D4: right zone named, questions where the object had been seen (n = 48)
    'methods': [(OURS, 5), ('Whole photo, no zones', 0), (PRIOR, 3), ('Random zone', 7)],
    'n_seen': 48,
    # D7 figure 1: right / wrong zone named / said "not seen" (it was there)
    'breakdown': [(OURS, 5, 24, 19), ('Whole photo, no zones', 0, 6, 42), (PRIOR, 3, 45, 0),
                  ('Random zone', 7, 41, 0)],
    'false_sightings': (0, 8), 'never_in_view': (2, 2),
    # D7 zone table: present frame pairs, ranked first ignoring tau, named at tau (zones 5, 6, 8 had none)
    'zones': [('Counter', 127, 15, 0), ('Armchair', 41, 0, 0), ('Floor by armchair', 32, 0, 0),
              ('Corner shelf', 19, 9, 9), ('Centre floor', 19, 0, 0)],
    # D7 hypotheses: (method, right, questions)
    'H1': [(OURS, 3, 3), ('Whole photo, no zones', 0, 3), (PRIOR, 3, 3), ('Random zone', 0, 3)],
    'H2': [(OURS, 2, 45), ('Whole photo, no zones', 0, 45), (PRIOR, 0, 45), ('Random zone', 7, 45)],
}
SOURCE = 'Source: pause_and_find.ipynb, round 3 (Kaggle run v5). Session 2 is held out: 60 frames, 50 questions.'

# ---------------------------------------------------------------- theme tokens (sampled from the deck's slide)
SURFACE, INK, INK2, MUTED = '#ffffff', '#2b2f36', '#4b5563', '#6b7280'
GRID, AXIS = '#e5e7eb', '#d8dce1'
NAVY, OCHRE = '#2b5f94', '#b5762e'            # validated pair (CVD dE 21, normal dE 26, both >= 3:1 on white)
OCHRE_LIGHT = '#e3c7a2'                       # lighter step of the ochre, for "ranked first, below threshold"
GREY_BAR, TRACK, TILE = '#c4c9d1', '#e1e4e9', '#f3f4f6'
CHANCE = 12.5
SERIF, SANS = 'Noto Serif', 'Lato'

plt.rcParams.update({
    'font.family': [SANS, 'Calibri', 'DejaVu Sans'], 'font.size': 13,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': INK,
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'hatch.color': 'white', 'hatch.linewidth': 2.6,
})


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def slide(title, subtitle, rect=(0.25, 0.13, 0.69, 0.55)):
    fig = plt.figure(figsize=(13.333, 7.5))
    fig.text(0.045, 0.925, title, fontsize=31, color=INK, va='top', family=[SERIF, 'Georgia'])
    fig.text(0.045, 0.835, subtitle, fontsize=15, color=MUTED, va='top', linespacing=1.25)
    fig.text(0.045, 0.035, SOURCE, fontsize=11, color=MUTED)
    ax = fig.add_axes(rect) if rect else None
    return fig, ax


def clean(ax, grid='x'):
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(AXIS); ax.spines['bottom'].set_linewidth(1.2)
    if grid: ax.grid(axis=grid, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=13)


def pct_axis(ax):
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))


def bold_tick(ax, text):
    for t in ax.get_yticklabels():
        if t.get_text() == text: t.set_fontweight('bold')


def chance_line(ax, label='chance (1 in 8)'):
    ax.axvline(CHANCE, color=INK2, lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.annotate(label, (CHANCE, 1), xycoords=('data', 'axes fraction'), xytext=(7, 4), textcoords='offset points',
                fontsize=13, color=INK2, va='bottom', annotation_clip=False)


def value_label(ax, x, y, main, sub, bold=False, size=20):
    """'48%' then a muted '(24/50)', as on the deck's slide."""
    mask = dict(facecolor=SURFACE, edgecolor='none', pad=1.5)     # the dashed chance line passes behind labels
    t = ax.text(x, y, main, va='center', fontsize=size, color=INK, fontweight='bold' if bold else 'normal', zorder=6,
                bbox=mask)
    if sub:
        w = t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width * 72 / ax.figure.dpi
        ax.annotate(sub, (x, y), xytext=(w + 7, 0), textcoords='offset points', va='center', fontsize=size * 0.72,
                    color=MUTED, zorder=6, bbox=mask)


def whisker(ax, lo, hi, y, cap=0.14):
    ax.plot([100 * lo, 100 * hi], [y, y], color=INK2, lw=1.8, zorder=4, solid_capstyle='butt')
    for e in (lo, hi): ax.plot([100 * e] * 2, [y - cap, y + cap], color=INK2, lw=1.8, zorder=4)


def method_bar(ax, y, v, name, height=0.6):
    ours = name == OURS
    ax.barh(y, max(v, 0.4), height=height, color=OCHRE if ours else GREY_BAR, hatch='//' if ours else None,
            lw=0, zorder=2)


def round_bars(fig, px=5):
    """Replace every bar rectangle with the same box, corners rounded by px screen pixels (as on the deck)."""
    for ax in fig.axes:
        (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
        sx, sy = abs(x1 - x0), abs(y1 - y0)                      # pixels per data unit
        for p in [p for p in ax.patches if type(p) is Rectangle and p.get_width() > 0]:
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            r = min(px / sx, w / 2, abs(h) / 2 * sy / sx)
            box = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', mutation_aspect=sx / sy,
                                 facecolor=p.get_facecolor(), edgecolor=p.get_edgecolor(), lw=p.get_linewidth(),
                                 hatch=p.get_hatch(), zorder=p.get_zorder())
            p.remove(); ax.add_patch(box)


def save(fig, name):
    round_bars(fig)
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
                        arrowprops=dict(arrowstyle='-|>', color=GREY_BAR, lw=2.5, mutation_scale=22), zorder=2)
            ax.text((s1 + s2) / 2, yi + 0.2, f'−{s1 - s2:.0f} pts', ha='center', fontsize=16, color=INK, fontweight='bold')
            ax.scatter([s1], [yi], s=360, color=NAVY, edgecolor=SURFACE, lw=2.5, zorder=6)
            ax.scatter([s2], [yi], s=360, color=OCHRE, edgecolor=SURFACE, lw=2.5, zorder=6)
            ax.text(s1 + 2.6, yi - 0.03, f'{s1:.0f}%', va='center', fontsize=19, color=INK, fontweight='bold')
            ax.text(s2 - 2.6, yi - 0.03, f'{s2:.0f}%', va='center', ha='right', fontsize=19, color=INK, fontweight='bold')
        else:
            ax.scatter([s1], [yi], s=360, color=NAVY, edgecolor=SURFACE, lw=2.5, zorder=6)
            ax.scatter([s2], [yi], s=150, color=OCHRE, edgecolor=SURFACE, lw=2, zorder=7)
            ax.text(s1 + 2.6, yi - 0.03, '0% in both sessions', va='center', fontsize=19, color=INK, fontweight='bold',
                    zorder=6, bbox=dict(facecolor=SURFACE, edgecolor='none', pad=2))
    ax.set_yticks(y, [r[0] for r in rows], fontsize=14, color=INK)
    ax.set_xlim(-3, 100); ax.set_ylim(y.min() - 0.5, y.max() + 0.6); pct_axis(ax)
    chance_line(ax)
    ax.scatter([], [], s=160, color=NAVY, label='Session 1 (calibration, 20 frames)')
    ax.scatter([], [], s=160, color=OCHRE, label='Session 2 (held out, 60 frames)')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.04), ncol=2, frameon=False, fontsize=13, handletextpad=0.3)
    save(fig, '01_calibration_vs_heldout.png')


# ---------------------------------------------------------------- 2. methods vs chance
def plot_methods():
    fig, ax = slide('On held-out questions, nothing beat a random guess',
                    'Right zone named for the 48 Session 2 questions where the object had already been in view.\n'
                    'Whiskers show the 95% Wilson range. Every range overlaps chance.')
    clean(ax)
    rows = DATA['methods']; n = DATA['n_seen']; y = np.arange(len(rows))[::-1]
    for yi, (name, k) in zip(y, rows):
        lo, hi = wilson(k, n); v = 100 * k / n
        method_bar(ax, yi, v, name); whisker(ax, lo, hi, yi)
        value_label(ax, 100 * hi + 1.2, yi, f'{v:.0f}%', f'({k}/{n})', bold=name == OURS)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16); bold_tick(ax, OURS)
    ax.set_xlim(0, 50); ax.set_ylim(-0.6, y.max() + 0.6); pct_axis(ax)
    chance_line(ax)
    save(fig, '02_methods_vs_chance.png')


# ---------------------------------------------------------------- 3. what each method answered
def plot_breakdown():
    fig, ax = slide('Wrong-zone answers: 50% for ours, 85–94% for guessing',
                    'Answers to the 48 questions where the object had been in view. When unsure, ours\n'
                    'often says "not seen" instead of sending you to look in the wrong place.',
                    rect=(0.25, 0.13, 0.46, 0.55))
    clean(ax, grid=None); ax.spines['bottom'].set_visible(False); ax.set_xticks([])
    parts = [('Right zone', NAVY, 'white'), ('Wrong zone named', OCHRE, 'white'),
             ('Said "not seen" (it was there)', GREY_BAR, INK)]
    rows = DATA['breakdown']; y = np.arange(len(rows))[::-1]
    for yi, (name, *vals) in zip(y, rows):
        tot, left = sum(vals), 0.0
        for (lab, col, tc), v in zip(parts, vals):
            w = 100 * v / tot
            if w:
                ax.barh(yi, w, left=left, height=0.62, color=col, edgecolor=SURFACE, lw=2.5, zorder=2)
                if w >= 8: ax.text(left + w / 2, yi, f'{w:.0f}%', ha='center', va='center', fontsize=16, color=tc,
                                   fontweight='bold', zorder=6)
            left += w
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16); bold_tick(ax, OURS)
    ax.set_xlim(0, 100); ax.set_ylim(-0.55, y.max() + 0.55)
    ax.legend([Patch(color=c) for _, c, _ in parts], [p[0] for p in parts], loc='lower left',
              bbox_to_anchor=(-0.02, 1.0), ncol=3, frameon=False, fontsize=13,
              handlelength=1.1, handletextpad=0.4, columnspacing=1.2)
    # stat tiles: the safety side of ours
    tiles = [(f'{DATA["false_sightings"][0]} of {DATA["false_sightings"][1]}', 'false sightings', 'questions where the object was\nnot in view at that moment'),
             (f'{DATA["never_in_view"][0]} of {DATA["never_in_view"][1]}', 'never-in-view questions', 'answered "not seen"; random and\nprior only both named a zone')]
    for j, (big, head, sub) in enumerate(tiles):
        y0 = 0.40 - j * 0.32
        fig.patches.append(FancyBboxPatch((0.745, y0), 0.215, 0.30, boxstyle='round,pad=0,rounding_size=0.008',
                                          transform=fig.transFigure, facecolor=TILE, edgecolor='none'))
        fig.text(0.765, y0 + 0.268, OURS.upper(), fontsize=10.5, color=MUTED, va='top', fontweight='bold')
        fig.text(0.765, y0 + 0.228, big, fontsize=34, color=INK, va='top', family=[SERIF, 'Georgia'])
        fig.text(0.765, y0 + 0.135, head, fontsize=14, color=INK, va='top', fontweight='bold')
        fig.text(0.765, y0 + 0.095, sub, fontsize=11.5, color=INK2, va='top', linespacing=1.3)
    save(fig, '03_answer_breakdown.png')


# ---------------------------------------------------------------- 4. where it found anything
def plot_zones():
    fig, ax = slide('All 9 correct sightings in Session 2 came from one zone',
                    'Session 2 frames × objects, by the zone the object was in. The counter held over half the\n'
                    'objects; the right zone often scored highest there, but never cleared the threshold.',
                    rect=(0.25, 0.18, 0.69, 0.50))
    clean(ax)
    rows = DATA['zones']; y = np.arange(len(rows))[::-1]
    for yi, (name, present, ranked, named) in zip(y, rows):
        ax.barh(yi, present, height=0.6, color=TRACK, lw=0, zorder=2)
        if ranked: ax.barh(yi, ranked, height=0.6, color=OCHRE_LIGHT, edgecolor=SURFACE, lw=2, zorder=3)
        if named: ax.barh(yi, named, height=0.6, color=OCHRE, hatch='//', edgecolor=SURFACE, lw=2, zorder=4)
        note = (f'{named} of {present} sighted in the right zone' if named else
                f'{ranked} of {present} ranked first, 0 sighted' if ranked else f'0 of {present} found')
        ax.text(present + 1.5, yi, note, va='center', fontsize=14, color=INK if (named or ranked) else MUTED,
                fontweight='bold' if named else 'normal')
    ax.legend([Patch(color=TRACK), Patch(color=OCHRE_LIGHT), Patch(facecolor=OCHRE, hatch='//', lw=0)],
              ['Object present', 'Right zone ranked first, below the threshold', 'Sighted in the right zone'],
              loc='lower left', bbox_to_anchor=(-0.01, 1.0), ncol=3,
              frameon=False, fontsize=13, handlelength=1.4, handleheight=1.0, handletextpad=0.4)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16); bold_tick(ax, 'Corner shelf')
    ax.set_xlim(0, 185); ax.set_ylim(-0.55, y.max() + 0.55)
    ax.set_xlabel('frame × object pairs (carried and person-blocked left out). Under-counter cupboard, glass shelves '
                  'and floor by cupboards held no objects.', fontsize=11.5, labelpad=8, color=MUTED)
    save(fig, '04_zones_where_found.png')


# ---------------------------------------------------------------- 5. hypotheses H1 vs H2
def plot_hypotheses():
    fig, _ = slide('Clear view, usual place: 3 of 3 right. Everything else: 2 of 45.',
                   'Right zone named, by question group (proposal section 3). Whiskers show the 95% Wilson range.',
                   rect=None)
    panels = [('H1', 'Clear view and usual place', 'visible, usual zone, not keys/phone', 'too few questions to tell'),
              ('H2', 'Everything else', 'keys/phone, hidden, far shelf, unusual zone', 'supported (guessing matches or beats ours)')]
    for j, (key, head, sub, verdict) in enumerate(panels):
        x0 = 0.215 + j * 0.46
        ax = fig.add_axes((x0, 0.13, 0.235, 0.47)); clean(ax)
        rows = DATA[key]; y = np.arange(len(rows))[::-1]
        for yi, (name, k, n) in zip(y, rows):
            lo, hi = wilson(k, n); v = 100 * k / n
            method_bar(ax, yi, v, name, height=0.58); whisker(ax, lo, hi, yi, cap=0.13)
            value_label(ax, min(100 * hi + 2.5, 104), yi, f'{k}/{n}', '', bold=name == OURS, size=16)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=13.5); bold_tick(ax, OURS)
        ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_ylim(-0.6, y.max() + 0.6); pct_axis(ax)
        chance_line(ax, 'chance')
        n_q = rows[0][2]
        fig.text(x0 - 0.17, 0.745, f'{key} · {head}', fontsize=17, color=INK, fontweight='bold')
        fig.text(x0 - 0.17, 0.705, f'{n_q} questions: {sub}', fontsize=13, color=MUTED)
        fig.text(x0 - 0.17, 0.668, f'Verdict: {verdict}', fontsize=13, color=INK2, fontweight='bold')
    save(fig, '05_hypotheses_h1_h2.png')


if __name__ == '__main__':
    plot_gap(); plot_methods(); plot_breakdown(); plot_zones(); plot_hypotheses()
