"""Recompute the mean +- SD entries of Tables 4 and 5 of the paper from the released outputs.
Table 4: revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/rows.json (10 policy seeds).
Table 5: paired_contrasts.csv written by the independent-context and shared-representation runners.
Usage (from the repository root, after those two runners have written their outputs):
  python paper/table_stats.py revision/experiments/outputs/abides_typed_audit_ippo_v2/formal/rows.json outputs/independent_context/paired_contrasts.csv outputs/shared_representation/paired_contrasts.csv"""
import csv, json, sys
import numpy as np

rows = json.load(open(sys.argv[1]))
ARMS = [('vanilla_ippo', 'Vanilla IPPO'), ('action_cost', 'Action-cost shaping'),
        ('always_on_actor', 'Always-on'), ('full_ace', 'Full ACE-RL')]
METRICS = [('q_v', 'q_V', 3), ('q_x', 'q_X', 3), ('selectivity', 'S', 3), ('mean_per_dealer_gross_pnl', 'Gross PnL', 1),
           ('mean_quoted_half_spread', 'Half-spread', 2), ('mean_pairwise_action_agreement', 'Agreement', 3),
           ('mean_action_entropy', 'Entropy', 3)]
v = lambda arm, k: np.array([r[f'{arm}__{k}'] for r in rows], dtype=float)

print('Table 4, mean +- SD over %d seeds' % len(rows))
for arm, name in ARMS:
    print('  %-22s' % name, '  '.join('%s %.*f+-%.*f' % (lab, d, v(arm, k).mean(), d, v(arm, k).std(ddof=1))
                                      for k, lab, d in METRICS))
print('Table 4, paired differences (mean +- SD of the per-seed difference; seeds with a positive difference)')
for other, name in (('always_on_actor', 'Full - Always-on'), ('vanilla_ippo', 'Full - Vanilla')):
    parts = []
    for k, lab, d in METRICS:
        diff = v('full_ace', k) - v(other, k)
        parts.append('%s %+.*f+-%.*f (%d/10 >0)' % (lab, d, diff.mean(), d, diff.std(ddof=1), int((diff > 0).sum())))
    print('  %-18s' % name, '  '.join(parts))
gap = v('full_ace', 'effective_half_spread') - v('vanilla_ippo', 'effective_half_spread')
print('  effective half-spread, Full - Vanilla: %+.3f+-%.3f' % (gap.mean(), gap.std(ddof=1)))
print('  per-seed S: Full %.3f..%.3f, Always-on max %.3f' % (v('full_ace', 'selectivity').min(),
      v('full_ace', 'selectivity').max(), v('always_on_actor', 'selectivity').max()))

print('Table 5, mean +- SD of the per-seed difference')
# the two runners name their columns differently
pick = lambda r, *ks: next(r[k] for k in ks if k in r)
for path, keep in ((sys.argv[2], ('C1', 'C3')), (sys.argv[3], ('S3', 'S4'))):
    for r in csv.DictReader(open(path)):
        if r['contrast'] in keep:
            print('  %s %s-%s %s: %.3f+-%.3f  positive %s/%s  Holm p %.3g' % (
                r['contrast'], pick(r, 'left', 'minuend_method'), pick(r, 'right', 'subtrahend_method'), r['metric'],
                float(pick(r, 'mean_difference', 'mean')), float(pick(r, 'sd_difference', 'sd')),
                pick(r, 'positive_sign_count', 'expected_sign_count'), pick(r, 'n', 'n_pairs'),
                float(pick(r, 'holm_p_value', 'holm_adjusted_p_value'))))
