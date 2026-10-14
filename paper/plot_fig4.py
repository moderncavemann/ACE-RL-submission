"""Figure 4 of the paper: controlled experiments.
Lines are means and bands are +/- 1 SD across seeds (in panel b only for Full ACE-RL and the shuffled mask), computed by fig4_data_sd.py from the per-seed
outputs of the released runners (50, 100 and 50 seeds).
Text is set by LaTeX with the paper's fonts (Linux Libertine + newtxmath).
Usage (from the repository root): python paper/plot_fig4.py paper/data/fig4 outputs/fig_controlled.pdf"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

data_dir, out_pdf = sys.argv[1], sys.argv[2]
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage[tt=false,type1=true]{libertine}\usepackage[libertine]{newtxmath}',
    'font.family': 'serif', 'font.size': 8.5,
    'axes.labelsize': 8.5, 'axes.titlesize': 8.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42,
})


def load(path, key):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r[key], []).append(r)
    return out


A = load(f'{data_dir}/panel_a.csv', 'series')
B = load(f'{data_dir}/panel_b.csv', 'series')
C = list(csv.DictReader(open(f'{data_dir}/panel_c.csv')))
arr = lambda rows, k: np.array([float(r[k]) for r in rows])

# one style per method, shared by all panels
BLUE, GREEN, ORANGE, VERM, GRAY, INK = '#0072B2', '#009E73', '#E69F00', '#D55E00', '#9A9A9A', '#1A1A1A'
STY = {
    'full':    dict(color=BLUE,   lw=1.7, ls='-',            zorder=6),
    'vanilla': dict(color=GRAY,   lw=2.6, ls='-',            zorder=3, solid_capstyle='butt'),
    'scalar':  dict(color=INK,    lw=0.9, ls=(0, (3.2, 2.0)), zorder=4),
    'cost':    dict(color=GREEN,  lw=1.4, ls=(0, (6, 1.6, 1.4, 1.6)), zorder=5),
    'shuffle': dict(color=ORANGE, lw=1.5, ls=(0, (4.5, 1.8)), zorder=5),
    'always':  dict(color=VERM,   lw=1.6, ls=(0, (1.0, 1.4)), zorder=5),
}
BAND = dict(alpha=0.13, lw=0)


def series(ax, rows, sty, band=True):
    x, m, s = arr(rows, 'policy_update'), arr(rows, 'mean'), arr(rows, 'sd')
    if band:
        ax.fill_between(x, m - s, m + s, color=sty['color'], zorder=sty['zorder'] - 2, **BAND)
    ax.plot(x, m, **sty)


def tidy(ax):
    ax.grid(axis='y', color='#E4E4E4', lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


fig = plt.figure(figsize=(7.03, 1.94))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.92], wspace=0.36,
              left=0.062, right=0.992, bottom=0.1995, top=0.7897)

# (a) action direction
ax = fig.add_subplot(gs[0])
series(ax, A['Vanilla PG'], STY['vanilla'])
series(ax, A['Scalar shift'], STY['scalar'], band=False)       # identical to Vanilla PG
series(ax, A['Action-cost shaping'], STY['cost'])
series(ax, A['Full ACE'], STY['full'])
ax.set_xlim(0, 400); ax.set_ylim(0.5, 0.95)
ax.set_xticks([0, 100, 200, 300, 400]); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_yticklabels(['.5', '.6', '.7', '.8', '.9'])
ax.set_xlabel('Policy update'); ax.set_ylabel(r'Valid response $q_V$', labelpad=2)
ax.set_title(r'\textbf{(a)} Response set (CH1)', loc='left', pad=4)
ax.text(398, 0.522, 'Vanilla PG and scalar shift coincide', color='#555555', ha='right', va='center', fontsize=7.8)
tidy(ax)

# (b) validity routing under a shared MLP
ax = fig.add_subplot(gs[1])
series(ax, B['Vanilla shared MLP'], STY['vanilla'], band=False)   # bands only for the S3 contrast,
series(ax, B['Always-on'], STY['always'], band=False)             # to keep the panel legible
series(ax, B['Shuffled mask'], STY['shuffle'])
series(ax, B['Full ACE'], STY['full'])
ax.set_xlim(0, 400); ax.set_ylim(-0.03, 0.34)
ax.set_xticks([0, 100, 200, 300, 400]); ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_yticklabels(['0', '.1', '.2', '.3'])
ax.set_xlabel('Policy update'); ax.set_ylabel(r'Selectivity $S=q_V-q_X$', labelpad=2)
ax.set_title(r'\textbf{(b)} Validity gating, shared MLP (CH2)', loc='left', pad=4)
tidy(ax)

# (c) audit coefficient: probabilities on top, financial utility below (no twin axis)
sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2], height_ratios=[2.05, 1], hspace=0.16)
axp = fig.add_subplot(sub[0]); axu = fig.add_subplot(sub[1], sharex=axp)
al = arr(C, 'alpha')
LB = '#56B4E9'
for k, col in (('q_V', BLUE), ('q_X', LB)):
    m, s = arr(C, f'{k}_mean'), arr(C, f'{k}_sd')
    axp.fill_between(al, m - s, m + s, color=col, zorder=2, **BAND)
axp.plot(al, arr(C, 'q_V_mean'), color=BLUE, lw=1.6, marker='o', ms=3.2, mec='white', mew=0.5, zorder=6)
axp.plot(al, arr(C, 'q_X_mean'), color=LB, lw=1.3, marker='o', ms=3.0, mfc='white', mec=LB, mew=0.9, zorder=5)
axp.set_ylim(0, 1.0); axp.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); axp.set_yticklabels(['0', '.25', '.50', '.75', '1'])
axp.set_ylabel('Probability', labelpad=2)
axp.text(0.49, 0.80, r'$q_V$ (valid)', color=BLUE, ha='right', va='top', fontsize=7.8)
axp.text(0.012, 0.965, 'Full ACE-RL', color='#555555', ha='left', va='top', fontsize=7.8)
axp.text(0.49, 0.31, r'$q_X=0.244$ (invalid)', color='#2B8CC4', ha='right', va='bottom', fontsize=7.8)
axp.set_title(r'\textbf{(c)} Audit coefficient $\alpha$', loc='left', pad=4)
plt.setp(axp.get_xticklabels(), visible=False)
tidy(axp)
m, s = arr(C, 'U_mean'), arr(C, 'U_sd')
axu.fill_between(al, m - s, m + s, color='#555555', zorder=2, **BAND)
axu.plot(al, m, color='#444444', lw=1.2, marker='s', ms=2.6, mec='white', mew=0.4, zorder=5)
axu.set_ylim(0.886, 0.918); axu.set_yticks([0.89, 0.90, 0.91]); axu.set_yticklabels(['.89', '.90', '.91'])
axu.set_ylabel(r'Utility $U$', labelpad=2)
axu.set_xlim(-0.02, 0.52); axu.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); axu.set_xticklabels(['0', '.1', '.2', '.3', '.4', '.5'])
axu.set_xlabel(r'Audit coefficient $\alpha$')
tidy(axu)
for a_ in (axp, axu):
    a_.yaxis.set_label_coords(-0.205, 0.5)


def h(k, label):
    st = {kk: vv for kk, vv in STY[k].items() if kk not in ('zorder',)}
    return Line2D([], [], label=label, **st)


handles = [h('full', 'Full ACE-RL'), h('vanilla', 'Vanilla PG'), h('scalar', 'Scalar shift'),
           h('cost', 'Action-cost shaping'), h('shuffle', 'Shuffled mask'), h('always', 'Always-on')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.055, 1.02), ncol=6, frameon=False,
           fontsize=7.9, handlelength=2.6, handletextpad=0.5, columnspacing=1.5, borderaxespad=0.2)
os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
fig.savefig(out_pdf)
print('saved', out_pdf)
