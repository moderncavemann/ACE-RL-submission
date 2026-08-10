# Prospective protocol: ACE-RL audit-coefficient sensitivity

## Status and purpose

This protocol is frozen before the sensitivity runner is implemented or any formal result is inspected. The study measures how the audit coefficient `alpha` changes response-set probability and financial utility in the same controlled independent-logit learner used by the paper's main signal-route assay. It is a coefficient analysis of that learner, not a new market-performance benchmark.

## Fixed learner and environment

- Contexts: `V` (valid competitive opportunity), `I` (inventory blocked), and `P` (price or risk blocked), sampled with probabilities `[0.50, 0.25, 0.25]`.
- Actions: `H` (Hold), `M` (moderate response), and `A` (aggressive response).
- Exact financial utilities: `V=[1.00,0.97,0.94]`, `I=[1.00,0.35,0.05]`, and `P=[1.00,0.55,0.20]`.
- Policy: one independent three-action softmax logit vector per context.
- Financial update: context-centered REINFORCE-style policy gradient with ordinary gradient ascent.
- Audit route: the set-valued response label `{M,A}` is active only in `V`; invalid contexts receive exact zero audit gradient.
- Formal seeds: integers `6000` through `6049` inclusive.
- Updates: `400`; batch size: `384`; learning rate: `0.08`.
- Initial-logit standard deviation: `0.15`; reward-noise standard deviation: `0.10`; float precision: `float64`.
- Every alpha value within a seed uses identical initial logits, contexts, action uniforms, and reward shocks.

## Prespecified coefficient grid

Run `alpha` in `[0.00, 0.10, 0.20, 0.30, 0.40, 0.50]`. The `alpha=0.00` row is the matched financial-only reference. No grid point may be added, removed, or retuned after the formal run is inspected.

## Endpoints and uncertainty

For every seed and coefficient, report final:

- `q_V = P(M or A | V)`;
- `q_X = [P(M or A | I) + P(M or A | P)] / 2`;
- selectivity `S = q_V - q_X`;
- exact expected financial utility `U` under the fixed context probabilities and utility table.

Report means and two-sided 95% Student-t intervals over all 50 seeds. Retain every planned seed and record all failures.

## Prespecified checks

1. **Valid-context direction.** For each positive coefficient, compute the paired difference `q_V(alpha)-q_V(0)`. A row passes when its two-sided 95% paired-t interval is strictly positive and at least 45/50 paired differences are positive.
2. **Invalid-context isolation.** For every coefficient and seed, `q_X(alpha)` must equal `q_X(0)` within absolute tolerance `1e-12`, because the policy uses independent context parameters and the audit route is active only in `V`.
3. **Dose ordering.** Report whether the mean `q_V` values are nondecreasing over the complete frozen grid. This is a descriptive diagnostic rather than an admissibility gate.
4. **Financial trade-off.** Report `U` for every grid point without imposing a preferred optimum or selecting a coefficient post hoc.

The paper may state only the checks supported by the complete formal artifacts. A failed check remains in the result ledger and narrows the corresponding statement.

## Artifact and replay contract

The formal run must write the configuration, source and protocol hashes, per-seed rows, summary rows, paired checks, a failure ledger, software versions, checksums, and a terminal receipt. A deterministic replay must reproduce the core CSV and JSON artifacts byte for byte. Engineering-smoke seeds are negative and cannot support scientific claims.
