"""Figure 3 of the paper: the ten-dealer market.
q_V and q_X are shadow-audit response probabilities: the probability that the policy, evaluated on the shadow
observation with the live recurrent state, assigns to the one- to three-tick quotes, averaged over valid (q_V) and
invalid (q_X) audited states.  Gross PnL is measured in the live market.
(a, b) Shadow-audit response in valid and invalid states on the recorded training rollouts of the formal run
    (10 policy seeds x 96 episodes; each history row is the rollout consumed by that episode's update),
    mean +/- 1 SD with the band truncated to [0, 1].
(c) Held-out evaluation, one point per policy seed: response in invalid states (x) and valid states (y).
    Grey segments link the Always-on and Full ACE-RL policies of the same seed.
(d) Held-out evaluation, one point per policy seed: selectivity and gross PnL per dealer; large markers and bars
    are means +/- 1 SD.
Data: paper/data/training_history.csv (written by extract_training_history.py from the per-seed logs of the
      formal run) and revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/rows.json.
Usage (from the repository root):
  python paper/plot_tendealer.py paper/data/training_history.csv revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/rows.json outputs/fig_tendealer.pdf"""
import csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

hist_csv, rows_json, out_pdf = sys.argv[1], sys.argv[2], sys.argv[3]
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[T1]{fontenc}\usepackage[tt=false,type1=true]{libertine}\usepackage[libertine]{newtxmath}',
    'font.family': 'serif', 'font.size': 8.5,
    'axes.labelsize': 8.5, 'axes.titlesize': 8.8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.major.pad': 2, 'ytick.major.pad': 2,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})
BLUE, GREEN, VERM, GRAY = '#0072B2', '#009E73', '#D55E00', '#8C8C8C'
METHODS = [  # key, label, colour, marker
    ('vanilla_ippo', 'Vanilla IPPO', GRAY, 'o'),
    ('action_cost', 'Action-cost shaping', GREEN, 's'),
    ('always_on_actor', 'Always-on', VERM, '^'),
    ('full_ace', 'Full ACE-RL', BLUE, 'D'),
]

# ---------------- data
H = {}
for r in csv.DictReader(open(hist_csv)):
    H.setdefault(r['method'], []).append(r)


def curves(method, key):
    rows = H[method]
    seeds = sorted({int(r['policy_seed']) for r in rows})
    X = np.full((len(seeds), 96), np.nan)
    for r in rows:
        X[seeds.index(int(r['policy_seed'])), int(r['episode']) - 1] = float(r[key])
    assert not np.isnan(X).any()
    return np.arange(1, 97), X.mean(axis=0), X.std(axis=0, ddof=1)


R = json.load(open(rows_json))
ev = lambda m, k: np.array([r[f'{m}__{k}'] for r in R])

fig = plt.figure(figsize=(7.03, 1.98))
gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 0.86, 1.02], wspace=0.42,
              left=0.056, right=0.993, bottom=0.2015, top=0.7667)


def tidy(ax, grid='y'):
    ax.grid(axis=grid, color='#E4E4E4', lw=0.45, zorder=0)
    ax.set_axisbelow(True)


# ---------------- (a), (b) training dynamics
for j, (q, title) in enumerate((('q_v', r'\textbf{(a)} Training: valid states'),
                                ('q_x', r'\textbf{(b)} Training: invalid states'))):
    ax = fig.add_subplot(gs[j])
    for key, lab, col, _ in METHODS:
        x, m, s_ = curves(key, q)
        ax.fill_between(x, np.clip(m - s_, 0, 1), np.clip(m + s_, 0, 1), color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot(x, m, color=col, lw=1.35, zorder=4 if key in ('always_on_actor', 'full_ace') else 3)
    ax.set_xlim(1, 96); ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([1, 24, 48, 72, 96]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(['0', '.25', '.5', '.75', '1'])
    ax.set_xlabel('Training episode')
    ax.set_ylabel(r'Shadow-audit $q_V$' if q == 'q_v' else r'Shadow-audit $q_X$', labelpad=2)
    ax.set_title(title, loc='left', pad=4)
    tidy(ax)

# ---------------- (b) held-out operating points
ax = fig.add_subplot(gs[2])
ax.plot([0, 1], [0, 1], color='#9A9A9A', lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(0.60, 0.53, r'$S=0$', color='#555555', fontsize=7.8, rotation=0, ha='left', va='top')
for qx0, qv0, qx1, qv1 in zip(ev('always_on_actor', 'q_x'), ev('always_on_actor', 'q_v'),
                              ev('full_ace', 'q_x'), ev('full_ace', 'q_v')):
    ax.plot([qx0, qx1], [qv0, qv1], color='#C8C8C8', lw=0.55, zorder=2)
for key, lab, col, mk in METHODS:
    ax.scatter(ev(key, 'q_x'), ev(key, 'q_v'), s=13, marker=mk, facecolor=col, edgecolor='white',
               linewidth=0.4, alpha=0.95, zorder=4)
ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.06)
ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
ax.set_xticklabels(['0', '.5', '1']); ax.set_yticklabels(['0', '.5', '1'])
ax.set_xlabel(r'Shadow-audit $q_X$ (invalid)'); ax.set_ylabel(r'Shadow-audit $q_V$ (valid)', labelpad=2)
ax.set_title(r'\textbf{(c)} Held-out, per seed', loc='left', pad=4)
ax.set_aspect('auto')
tidy(ax, 'x')

# ---------------- (d) selectivity vs gross PnL
ax = fig.add_subplot(gs[3])
for key, lab, col, mk in METHODS:
    S, P = ev(key, 'selectivity'), ev(key, 'mean_per_dealer_gross_pnl')
    ax.scatter(S, P, s=13, marker=mk, facecolor=col, edgecolor='white', linewidth=0.4, alpha=0.9, zorder=3)
    ax.errorbar(S.mean(), P.mean(), xerr=S.std(ddof=1), yerr=P.std(ddof=1), fmt='none', ecolor=col,
                elinewidth=0.9, capsize=0, alpha=0.9, zorder=5)
    ax.scatter([S.mean()], [P.mean()], s=26, marker=mk, facecolor=col, edgecolor='#1A1A1A', linewidth=0.6, zorder=6)
ax.set_xlim(-0.13, 0.82); ax.set_ylim(0, 1400)
ax.set_xticks([0, .2, .4, .6, .8]); ax.set_xticklabels(['0', '.2', '.4', '.6', '.8'])
ax.set_yticks([0, 400, 800, 1200])
ax.set_xlabel(r'Selectivity $S=q_V-q_X$'); ax.set_ylabel('Live gross PnL per dealer', labelpad=2)
ax.set_title(r'\textbf{(d)} Held-out: $S$ vs.\ PnL', loc='left', pad=4)
tidy(ax, 'both')

# ---------------- shared legend
h = [Line2D([], [], color=col, marker=mk, ms=4.2, mfc=col, mec='white', mew=0.4, lw=1.5, label=lab)
     for _, lab, col, mk in METHODS]
fig.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8,
           handlelength=2.2, columnspacing=1.6, handletextpad=0.5, borderaxespad=0.2)
os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
fig.savefig(out_pdf)
print('saved', out_pdf)
for key, lab, col, mk in METHODS:
    print('%-20s S=%.3f PnL=%.1f qv=%.3f qx=%.3f' % (lab, ev(key, 'selectivity').mean(),
          ev(key, 'mean_per_dealer_gross_pnl').mean(), ev(key, 'q_v').mean(), ev(key, 'q_x').mean()))
