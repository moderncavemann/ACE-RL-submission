"""Sparse- and noisy-audit study of the paper (Section 5.7), run on the released independent-logit experiment.
The study was added after review and was not prespecified.

Uses the released code (experiments/independent_context/run_main_experiment.py) unchanged:
same seeds (1000-1049), streams, learner, alpha=0.30, 400 updates, batch 384, lr 0.08.
Two departures from Full ACE-RL, each drawn from a separate random stream so that contexts,
action-sampling draws, and reward shocks stay identical to the main assay:

  * audit fraction rho: each update audits a uniform random subset of round(rho*B) rows;
    the audit loss averages over the audited rows (Algorithm 1 with B_A = round(rho*B)).
  * validity errors eps: each row's validity flag is flipped independently with probability eps
    (false positives in blocked contexts, false negatives in the valid context).

rho = 1 and eps = 0 reproduce Full ACE-RL exactly. For the figure, the script also stores the
learning curves (q_V, q_X, S every 10 updates) and the released Vanilla PG, Full ACE-RL and
shuffled-mask runs on the same seeds as reference levels.
Usage (from the repository root): python paper/robustness.py experiments/independent_context outputs/robustness.json
paper/data/robustness.json holds the output used for Figure 5.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, sys.argv[1])
import run_main_experiment as M

cfg = M.Config()
RHOS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)
EPSS = (0.0, 0.05, 0.1, 0.2, 0.3)


def train(seed, stream, mode, level):
    logits = stream.initial_logits.copy()
    rng = np.random.default_rng([seed, 2026, 1 if mode == 'rho' else 2, int(round(level * 1000))])
    B = cfg.batch_size
    def point(lg):
        e0 = M.exact_endpoints(lg)
        return e0['q_V'], e0['q_X']
    qv0, qx0 = point(logits)
    curve = dict(q_V=[qv0], q_X=[qx0], S=[qv0 - qx0])
    for k in range(cfg.updates):
        contexts = stream.contexts[k]
        fin, _, _ = M.batch_gradients('vanilla_pg', logits, contexts, stream.action_uniforms[k],
                                      stream.reward_shocks[k], stream.shuffle_permutations[k], cfg)
        probs = M.softmax(logits)
        valid = contexts == 0
        if mode == 'rho':
            m = int(round(level * B))
            audited = np.zeros(B, dtype=bool)
            audited[rng.choice(B, size=m, replace=False)] = True
            active = audited & valid
            denom = m
        else:
            flip = rng.random(B) < level
            active = np.where(flip, ~valid, valid)
            denom = B
        audit = np.zeros((3, 3))
        for c in range(3):
            n = int(np.sum(active & (contexts == c)))
            if n:
                audit[c] = cfg.actor_alpha * n / denom * M.response_set_log_gradient(probs[c])
        logits = np.clip(logits + cfg.learning_rate * (fin + audit), -cfg.logit_guard, cfg.logit_guard)
        if (k + 1) % cfg.record_every == 0:
            qv, qx = point(logits)
            curve['q_V'].append(qv); curve['q_X'].append(qx); curve['S'].append(qv - qx)
    e = M.exact_endpoints(logits)
    return dict(q_V=e['q_V'], q_X=e['q_X'], S=e['q_V'] - e['q_X'], U=e['U'], curve=curve)


t0 = time.time()
res = {'rho': {str(r): [] for r in RHOS}, 'eps': {str(e): [] for e in EPSS}, 'vanilla': [], 'full': [], 'shuffled': []}


def released(seed, method, stream):
    traj, _, f = M.train_method(seed, method, stream, cfg)
    curve = dict(q_V=[t['q_V'] for t in traj], q_X=[t['q_X'] for t in traj], S=[t['q_V'] - t['q_X'] for t in traj])
    return dict(q_V=f['q_V'], q_X=f['q_X'], S=f['q_V'] - f['q_X'], U=f['U'], curve=curve)


for seed in cfg.seeds:
    stream = M.make_seed_stream(seed, cfg)
    res['vanilla'].append(released(seed, 'vanilla_pg', stream))
    res['full'].append(released(seed, 'full_ace', stream))
    res['shuffled'].append(released(seed, 'shuffled_validity_ace', stream))
    for r in RHOS:
        res['rho'][str(r)].append(train(seed, stream, 'rho', r))
    for e in EPSS:
        res['eps'][str(e)].append(train(seed, stream, 'eps', e))
print('seconds', round(time.time() - t0, 1))

# sanity: rho=1 and eps=0 must equal Full ACE-RL
for key, grp in (('rho', '1.0'), ('eps', '0.0')):
    d = max(abs(a['q_V'] - b['q_V']) + abs(a['q_X'] - b['q_X']) for a, b in zip(res[key][grp], res['full']))
    print('max |diff| vs full for', key, grp, d)

def summ(rows, k):
    v = np.array([r[k] for r in rows]); return v.mean(), v.std(ddof=1)
for label, rows in [('vanilla', res['vanilla']), ('full', res['full']), ('shuffled', res['shuffled'])]:
    print(label, ' '.join('%s=%.3f±%.3f' % (k, *summ(rows, k)) for k in ('q_V', 'q_X', 'S', 'U')))
for key, levels in (('rho', RHOS), ('eps', EPSS)):
    for l in levels:
        rows = res[key][str(l)]
        sv = np.array([r['S'] for r in rows]); svan = np.array([r['S'] for r in res['vanilla']])
        print(key, l, ' '.join('%s=%.3f±%.3f' % (k, *summ(rows, k)) for k in ('q_V', 'q_X', 'S', 'U')),
              ' S>vanilla in %d/50' % int(np.sum(sv > svan)))
os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
json.dump(res, open(sys.argv[2], 'w'))
