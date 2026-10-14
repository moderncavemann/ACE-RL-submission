"""Per-checkpoint mean and SD across seeds for Figure 4 of the paper, computed from the
per-seed outputs of the released runners:
  independent_context/run_main_experiment.py --formal      -> <indep>/learning_curves.csv
  shared_representation/run_experiment.py --formal         -> <shared>/learning_curves.csv
  coefficient_sensitivity/run_experiment.py --formal       -> <coef>/seed_results.csv
Usage (from the repository root, after the three runners have written their outputs):
  python paper/fig4_data_sd.py outputs/independent_context outputs/shared_representation outputs/coefficient_sensitivity outputs/fig4
paper/data/fig4/ holds the files used for the figure in the paper."""
import csv, sys, os
from collections import defaultdict
import numpy as np

indep, shared, coef, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)


def curves(path, methods, metric):
    acc = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r['method'] in methods:
            acc[(methods[r['method']], int(r['update']))].append(float(r[metric]))
    rows = []
    for (series, upd), v in sorted(acc.items()):
        v = np.array(v)
        rows.append(dict(series=series, policy_update=upd, n=v.size, mean=v.mean(), sd=v.std(ddof=1)))
    return rows


def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


write(f'{out}/panel_a.csv', curves(f'{indep}/learning_curves.csv',
      {'vanilla_pg': 'Vanilla PG', 'scalar_only_valid_shift': 'Scalar shift',
       'action_dependent_cost_advantage': 'Action-cost shaping', 'full_ace': 'Full ACE'}, 'q_V'))
write(f'{out}/panel_b.csv', curves(f'{shared}/learning_curves.csv',
      {'vanilla_shared': 'Vanilla shared MLP', 'always_on_shared': 'Always-on',
       'shuffled_validity_shared': 'Shuffled mask', 'full_ace_shared': 'Full ACE'}, 'selectivity'))
acc = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(f'{coef}/seed_results.csv')):
    for k in ('q_V', 'q_X', 'U'):
        acc[float(r['alpha'])][k].append(float(r[k]))
rows = []
for a in sorted(acc):
    row = dict(alpha=a, n=len(acc[a]['q_V']))
    for k in ('q_V', 'q_X', 'U'):
        v = np.array(acc[a][k]); row[f'{k}_mean'] = v.mean(); row[f'{k}_sd'] = v.std(ddof=1)
    rows.append(row)
write(f'{out}/panel_c.csv', rows)
for p in ('a', 'b'):
    last = [r for r in csv.DictReader(open(f'{out}/panel_{p}.csv')) if r['policy_update'] == '400']
    print(p, [(r['series'], r['n'], round(float(r['mean']), 4), round(float(r['sd']), 4)) for r in last])
print('c', [(r['alpha'], round(r['q_V_mean'], 4), round(r['q_V_sd'], 4), round(r['q_X_mean'], 4), round(r['U_mean'], 4), round(r['U_sd'], 4)) for r in rows])
