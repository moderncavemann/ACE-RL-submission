"""Extract the per-episode training history of the formal ten-dealer run (Figure 3a-b of the paper)
and check the logged audit invariants of every update.

Input : revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/seeds/seed_5600*/result.json
Output: training_history.csv with one row per policy seed, training episode and method.
Each history row is the market rollout consumed by that episode's update (a training rollout, not a held-out
evaluation). The script also reports, for every audited method, the largest logged audit gradient on critic
parameters and the largest audit loss on invalid rows across all updates; both are exactly zero for Full ACE-RL.

The per-seed logs (formal/seeds/) are not part of this repository; paper/data/training_history.csv is the output
of this script.
Usage (from the repository root): python paper/extract_training_history.py revision/experiments/outputs/abides_typed_audit_ippo_v2/formal outputs/training_history.csv
"""
import csv, json, os, sys

formal, out = sys.argv[1], sys.argv[2]
METHODS = ['vanilla_ippo', 'action_cost', 'always_on_actor', 'full_ace']
rows, checks = [], {m: {'critic': [], 'invalid': []} for m in METHODS}
for seed in range(560000, 560010):
    r = json.load(open(os.path.join(formal, 'seeds', 'seed_%d' % seed, 'result.json')))
    assert r['policy_seed'] == seed and len(r['training_history']) == 96
    for episode, h in enumerate(r['training_history'], 1):
        assert h['completed_response_episodes'] == episode
        for m in METHODS:
            p, u = h['methods'][m]['path_summary'], h['methods'][m]['update']
            rows.append(dict(policy_seed=seed, episode=episode, method=m, q_v=p['q_v'], q_x=p['q_x'],
                             selectivity=p['selectivity'], gross_pnl=p['mean_per_dealer_gross_pnl'],
                             agreement=p['mean_pairwise_action_agreement'],
                             entropy=p['mean_normalized_action_entropy']))
            if u.get('audit_critic_grad_max_abs') is not None:
                checks[m]['critic'].append(u['audit_critic_grad_max_abs'])
            if u.get('audit_invalid_loss_max_abs') is not None:
                checks[m]['invalid'].append(u['audit_invalid_loss_max_abs'])
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('wrote', len(rows), 'rows to', out)
for m in METHODS:
    c, v = checks[m]['critic'], checks[m]['invalid']
    if c:
        print('%-16s updates=%d  max audit gradient on critic=%r  max audit loss on invalid rows=%r'
              % (m, len(c), max(c), max(v) if v else None))
