# Prospective protocol: shared-representation contextual-selectivity assay

## Research question and evidence boundary

The existing three-context assay gives each context an independent logit vector. This prospective follow-up asks whether aligned validity labels remain more selective than dose-matched shuffled labels when all contexts share a neural representation and an output head, so that an audit update on a valid row can change predictions in invalid contexts. It is a component-level neural-policy assay. It cannot establish limit-order-book performance, collusion mitigation, equilibrium effects, profit improvement, legal intent, deployment safety, or superiority over constrained RL. It does not replace or modify any earlier frozen decision.

## Environment and shared policy

The latent contexts, context probabilities, actions, financial-utility matrix, reward-noise distribution, response set, and canonical validity decision match `multi_context_set_audit_v1`: `V` is a valid response opportunity with probability `0.50`, `I` is inventory-blocked with probability `0.25`, `P` is price/risk-blocked with probability `0.25`, and actions are `H`, `M`, and `A`. Utilities are `[1.00,0.97,0.94]` in `V`, `[1.00,0.35,0.05]` in `I`, and `[1.00,0.55,0.20]` in `P`. Realized training utility adds independent Normal(`0`,`0.10`) noise. The auditor returns `(z=1,R={M,A})` only for `V`.

All three contexts use one two-layer policy: a three-dimensional continuous observation is mapped through a fully shared affine--tanh layer of width `8` and one fully shared three-action output head. Context centers are `V=[1.0,0.0,0.0]`, `I=[0.8,1.0,0.2]`, and `P=[0.8,0.2,1.0]`. Each training row adds independent Normal(`0`,`0.08`) observation noise. Final endpoints are evaluated at the three noise-free centers. The architecture contains no context-specific parameters or heads.

Parameters are initialized from one matched stream per seed: input weights use Normal(`0`,`0.25`), hidden biases Normal(`0`,`0.05`), output weights Normal(`0`,`0.20`), and output biases Normal(`0`,`0.05`). The financial update is a complete-batch REINFORCE gradient centered separately within each latent context. The audit gradient is the complete-batch gradient of `log(P(M)+P(A))` on active rows. All parameters are updated by plain gradient ascent and clipped elementwise to `[-10,10]` only as a numerical guard.

## Compared methods

All arms receive identical initial parameters, latent contexts, noisy observations, action uniforms, financial shocks, and shuffle permutations within each seed.

1. `vanilla_shared`: financial policy gradient only.
2. `full_ace_shared`: financial policy gradient plus the response-set actor gradient on canonical valid rows.
3. `shuffled_validity_shared`: the canonical validity vector is permuted within each batch before applying the same response-set gradient; active-row count and coefficient exactly match `full_ace_shared`.
4. `always_on_shared`: the same response-set gradient is applied to every row; this larger-dose continuity baseline is secondary.

## Frozen run configuration

- Formal seeds: integers `3000` through `3099` (100 new streams).
- Updates: `400`.
- Batch size: `384`.
- Hidden width: `8`; activation: `tanh`.
- Actor learning rate: `0.03`.
- Audit coefficient: `0.30`.
- Training observation-noise standard deviation: `0.08`.
- Training reward-noise standard deviation: `0.10`.
- Curve checkpoints: update `0` and every `10` updates.
- No seed, endpoint, architecture, coefficient, update budget, contrast, or gate may change after any formal-seed output is inspected.
- Engineering smoke is restricted to seeds `-2` and `-1`, at most `10` updates, and may repair structural code errors only. Smoke results cannot tune the frozen configuration.

## Endpoints and nontrivial-sharing diagnostic

Primary endpoints are `q_V=P(M or A|V)`, `q_X=0.5[P(M or A|I)+P(M or A|P)]`, selectivity `S=q_V-q_X`, and exact expected financial utility `U=0.50 E[u|V]+0.25 E[u|I]+0.25 E[u|P]`. The shared-path spillover diagnostic is `Delta_X=abs(q_X(full_ace_shared)-q_X(vanilla_shared))`. Parameter sharing is considered behaviorally active when `Delta_X>1e-6`; this diagnostic prevents an interpretation that the shared architecture has silently reduced to independent context parameters.

## Frozen contrasts and decision rule

All `100 seeds x 4 methods = 400` seed-method runs must finish with finite parameters, valid probabilities, and zero silently dropped seeds. For each paired contrast report the mean, sample standard deviation, two-sided 95% paired t interval with 99 degrees of freedom, two-sided paired t-test p-value, and Holm-adjusted p-value across the five inferential contrasts:

- `S0`: `full_ace_shared - vanilla_shared` on `q_V`, expected positive.
- `S1`: `shuffled_validity_shared - full_ace_shared` on `q_X`, expected positive.
- `S2`: `always_on_shared - full_ace_shared` on `q_X`, expected positive.
- `S3`: `full_ace_shared - shuffled_validity_shared` on `S`, expected positive.
- `S4`: `full_ace_shared - shuffled_validity_shared` on `U`, expected positive.

Every 95% interval must lie strictly on the expected side of zero, every Holm-adjusted p-value must be below `0.05`, and at least `90/100` paired differences must have the expected sign. The nontrivial-sharing gate additionally requires `Delta_X>1e-6` in at least `95/100` seeds. A failure is reported as a failure; another contrast cannot rescue it.

## Required artifacts and execution policy

The runner must refuse formal execution if this protocol hash changes. It must preserve seed-level endpoints, learning curves, paired differences, paired contrasts, a header-preserving failure ledger, a machine-readable summary, an include-ready TeX table, a concise PDF/PNG figure, software versions, command, duration, code/protocol hashes, artifact hashes, and a checksum ledger. A second isolated execution must reproduce core scientific CSV/JSON artifacts byte for byte. Runtime timeout is 10 minutes. No automatic retry is permitted.

## Claim permitted only if every gate passes

The strongest permitted statement is: "In a frozen shared-MLP assay, aligned validity labels produce greater valid-versus-invalid response selectivity and higher exact expected financial utility than equal-dose shuffled labels, even though valid-row audit updates measurably propagate through shared parameters." This experiment cannot support a market-effect or anti-collusion claim.
