"""Figure 5 of the paper: robustness of Full ACE-RL to sparse and noisy audits.
Independent-logit assay, seeds 1000-1049 (50 runs per setting); lines are means and bands +/- 1 SD over seeds.
(a) Learning curves when a uniform fraction rho of each batch is audited (100%, 10%, 2%; drawn with decreasing
    width because they coincide; band only for rho = 100%).
(b) Learning curves when each validity flag is flipped with probability eps; the shuffled mask and
    Vanilla PG (released runners, same seeds) are shown for reference.
(c) Selectivity at the end of training against eps, one dot per seed.
Data: paper/data/robustness.json, written by robustness.py.
Usage (from the repository root): python paper/plot_robustness.py paper/data/robustness.json outputs/fig_robustness.pdf"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage[tt=false,type1=true]{libertine}\usepackage[libertine]{newtxmath}',
    'font.family': 'serif', 'font.size': 8.5,
    'axes.labelsize': 8.5, 'axes.titlesize': 8.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})
res = json.load(open(sys.argv[1]))
BLUE, GRAY, ORANGE = '#0072B2', '#8C8C8C', '#E69F00'
SHADES = ['#08306B', '#2F7FC1', '#8CC2E8']          # dark to light: clean -> perturbed
UPD = np.arange(0, 401, 10)


def curve(rows, key):
    C = np.array([r['curve'][key] for r in rows])
    return C.mean(axis=0), C.std(axis=0, ddof=1)


def tidy(ax):
    ax.grid(axis='y', color='#E4E4E4', lw=0.45, zorder=0)
    ax.set_axisbelow(True)


def curves_axes(ax, title):
    ax.set_xlim(0, 400); ax.set_ylim(0.15, 0.95)
    ax.set_xticks([0, 100, 200, 300, 400]); ax.set_yticks([.2, .4, .6, .8])
    ax.set_yticklabels(['.2', '.4', '.6', '.8'])
    ax.set_xlabel('Policy update'); ax.set_ylabel('Response probability', labelpad=2)
    ax.set_title(title, loc='left', pad=4)
    tidy(ax)


fig = plt.figure(figsize=(7.03, 1.98))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.84], wspace=0.30,
              left=0.058, right=0.992, bottom=0.2015, top=0.7614)

# ---------------- (a) sparse audits
ax = fig.add_subplot(gs[0])
m, s = curve(res['vanilla'], 'q_V')
ax.plot(UPD, m, color=GRAY, lw=1.2, zorder=3)
for (lev, lab), col, lw in zip((('1.0', r'100\%'), ('0.1', r'10\%'), ('0.02', r'2\%')), SHADES, (2.6, 1.6, 0.8)):
    for key, ls in (('q_V', '-'), ('q_X', (0, (4, 1.6)))):
        m, s = curve(res['rho'][lev], key)
        if lev == '1.0':
            ax.fill_between(UPD, m - s, m + s, color=BLUE, alpha=0.12, lw=0, zorder=2)
        ax.plot(UPD, m, color=col, ls=ls, lw=lw, zorder=10 - lw)   # thick first, thin on top: all three stay visible
ax.text(250, 0.905, r'valid $q_V$, every $\rho$', color=SHADES[0], ha='right', va='bottom', fontsize=7.7)
ax.text(396, 0.64, 'Vanilla PG', color='#666666', ha='right', va='bottom', fontsize=7.7)
ax.text(300, 0.172, r'invalid $q_X$, every $\rho$', color=SHADES[0], ha='right', va='bottom', fontsize=7.7)
curves_axes(ax, r'\textbf{(a)} Sparse audits: audit fraction $\rho$')

# ---------------- (b) noisy validity flags
ax = fig.add_subplot(gs[1])
m, s = curve(res['vanilla'], 'q_V')
ax.plot(UPD, m, color=GRAY, lw=1.2, zorder=3)
m, s = curve(res['shuffled'], 'q_X')
ax.plot(UPD, m, color=ORANGE, lw=1.2, ls=(0, (1.2, 1.2)), zorder=3)
for (lev, lab), col in zip((('0.0', '0'), ('0.1', r'10\%'), ('0.3', r'30\%')), SHADES):
    for key, ls in (('q_V', '-'), ('q_X', (0, (4, 1.6)))):
        m, s = curve(res['eps'][lev], key)
        ax.fill_between(UPD, m - s, m + s, color=col, alpha=0.10, lw=0, zorder=2)
        ax.plot(UPD, m, color=col, ls=ls, lw=1.35, zorder=4)
ax.text(250, 0.905, r'valid $q_V$', color=SHADES[0], ha='right', va='bottom', fontsize=7.7)
ax.text(396, 0.64, 'Vanilla PG', color='#666666', ha='right', va='bottom', fontsize=7.7)
ax.text(396, 0.468, 'shuffled mask', color='#B07A00', ha='right', va='bottom', fontsize=7.7)
ax.text(300, 0.172, r'invalid $q_X$', color=SHADES[0], ha='right', va='bottom', fontsize=7.7)
curves_axes(ax, r'\textbf{(b)} Noisy validity: flip rate $\varepsilon$')

# ---------------- (c) selectivity against the error rate
ax = fig.add_subplot(gs[2])
EPSS = ['0.0', '0.05', '0.1', '0.2', '0.3']
xs = np.array([float(e) for e in EPSS])
rng = np.random.default_rng(3)            # horizontal jitter only
for x, lev in zip(xs, EPSS):
    S = np.array([r['S'] for r in res['eps'][lev]])
    ax.scatter(x + rng.uniform(-0.006, 0.006, S.size), S, s=3.2, color=BLUE, alpha=0.35, lw=0, zorder=3)
mS = np.array([np.mean([r['S'] for r in res['eps'][l]]) for l in EPSS])
ax.plot(xs, mS, color=SHADES[0], lw=1.4, marker='o', ms=3.2, zorder=5)
van_S = np.mean([r['S'] for r in res['vanilla']]); shu_S = np.mean([r['S'] for r in res['shuffled']])
ax.axhline(van_S, color=GRAY, lw=1.1, zorder=2)
ax.axhline(shu_S, color=ORANGE, lw=1.1, ls=(0, (1.2, 1.2)), zorder=2)
ax.text(0.15, van_S - 0.012, 'Vanilla PG', color='#666666', ha='center', va='top', fontsize=7.7)
ax.text(0.15, shu_S + 0.012, 'shuffled mask', color='#B07A00', ha='center', va='bottom', fontsize=7.7)
ax.text(0.075, 0.675, 'Full ACE-RL', color=SHADES[0], ha='left', va='bottom', fontsize=7.7)
ax.set_xlim(-0.02, 0.32); ax.set_ylim(0.3, 0.72)
ax.set_xticks(xs); ax.set_xticklabels(['0', '5', '10', '20', '30'])
ax.set_yticks([.3, .4, .5, .6, .7]); ax.set_yticklabels(['.3', '.4', '.5', '.6', '.7'])
ax.set_xlabel(r'Flip rate $\varepsilon$ (\%)'); ax.set_ylabel(r'Selectivity $S$', labelpad=2)
ax.set_title(r'\textbf{(c)} Selectivity after training', loc='left', pad=4)
tidy(ax)

# ---------------- legend for the level shading
h = [Line2D([], [], color=SHADES[0], lw=1.4, label=r'$\rho=100\%$ / $\varepsilon=0$'),
     Line2D([], [], color=SHADES[1], lw=1.4, label=r'$\rho=10\%$ / $\varepsilon=10\%$'),
     Line2D([], [], color=SHADES[2], lw=1.4, label=r'$\rho=2\%$ / $\varepsilon=30\%$'),
     Line2D([], [], color='#444444', lw=1.2, ls='-', label=r'valid $q_V$'),
     Line2D([], [], color='#444444', lw=1.2, ls=(0, (4, 1.6)), label=r'invalid $q_X$')]
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=7.9,
           handlelength=2.2, columnspacing=1.5, handletextpad=0.5, borderaxespad=0.2)
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
fig.savefig(sys.argv[2])
print('saved', sys.argv[2])
for l in EPSS:
    S = np.array([r['S'] for r in res['eps'][l]]); print('eps', l, 'S %.3f' % S.mean())
print('vanilla S %.3f shuffled S %.3f' % (van_S, shu_S))
