# Prospective protocol: multi-context set-valued audit main comparison and module ablations

## Research question and boundary

This controlled follow-up asks whether the complete ACE-RL audit contract supplies action direction, whether a context-aligned validity mask is more informative than an equal-dose shuffled mask, and whether parameter separation prevents audit gradients from changing a financial critic. It is an independent prospective follow-up to the earlier two-context diagnostic. It does not replace or repair that experiment's frozen `NO-GO` decision (`0.795634 < 0.80`; `14/50 < 45/50`). Neither study is evidence of limit-order-book effectiveness, collusion mitigation, profit improvement, equilibrium selection, deployment safety, or superiority over constrained RL.

## E-main: three-context, three-action set-valued audit assay

### Environment and audit contract

Each sample independently draws a context and one action. Context `V` is a valid competitive-response opportunity with probability `0.50`; context `I` is inventory-blocked with probability `0.25`; context `P` is price/risk-blocked with probability `0.25`. The actions are `H` (Hold/no response), `M` (moderate response), and `A` (aggressive response). Exact financial utilities are:

| Context | `H` | `M` | `A` |
|---|---:|---:|---:|
| `V`: valid opportunity | 1.00 | 0.97 | 0.94 |
| `I`: inventory-blocked | 1.00 | 0.35 | 0.05 |
| `P`: price/risk-blocked | 1.00 | 0.55 | 0.20 |

Training adds an independent Normal(`0`, `0.10`) shock to realized financial utility. Final evaluation uses the noise-free matrix. The policy has one three-action softmax logit vector per context, initialized independently from Normal(`0`, `0.15`). The canonical auditor returns `(z=1, R={M,A})` in `V` and `(z=0, R=empty)` in `I` and `P`. The copied audit observation is discarded and never changes the sampled context, action, financial utility, or subsequent sample.

### Compared methods

All methods use the same per-context centered REINFORCE estimator, complete-batch scaling, training budget, initial logits, context draws, action uniforms, and reward shocks within each seed.

1. `vanilla_pg`: policy gradient from financial utility only.
2. `action_dependent_cost_advantage`: financial utility minus `0.20` times a context-dependent action cost. Costs are `[1,0,0]` for `[H,M,A]` in `V` and `[0,1,1]` in `I/P`. This is a strong alternative signal route using the same context semantics; it is not an ACE-RL ablation and ACE-RL will not be ranked against it.
3. `full_ace`: financial policy gradient plus `0.30 * log(P(M)+P(A))` on canonical valid audit rows. Invalid rows contribute exact zero and remain in the full-batch denominator.
4. `scalar_only_valid_shift`: remove M1 action specificity. On the same `V` rows selected by the canonical auditor, subtract `0.40` from all three action utilities; no response-set actor label is applied. Per-context centering should cancel this action-independent shift.
5. `shuffled_validity_ace`: remove M2 contextual alignment while matching audit dose. For every batch, deterministically permute the canonical validity vector and use the permuted mask with response set `{M,A}`. The active-row count, coefficient, samples, and complete-batch denominator are exactly matched to `full_ace`.
6. `always_on_actor_regularization`: apply the same response-set actor label in all contexts. This continuity baseline removes the mask but has a larger audit dose, so it is secondary rather than the primary M2 ablation.

### Frozen run configuration

- Seeds: integers `1000` through `1049` (50 new initialization/training streams not used by the earlier experiments).
- Updates: `400`.
- Batch size: `384`.
- Actor learning rate: `0.08`, plain gradient ascent.
- Initial logit standard deviation: `0.15`.
- Training reward-noise standard deviation: `0.10`.
- Direct actor coefficient: `0.30`.
- Scalar shift: `0.40`.
- Action-cost weight: `0.20`.
- Numerical logit guard: `[-10,10]`.
- Curve checkpoints: update `0` and every `10` updates.
- No coefficient, seed, endpoint, contrast, or threshold may change after any test-seed output is inspected.
- An engineering smoke may use only seeds `-2` and `-1`, at most `10` updates, and may repair structural code errors only. Smoke outcomes cannot tune the frozen scientific configuration.

### Endpoints

Primary endpoints are `q_V = P(M|V)+P(A|V)`, `q_X = 0.5 * [(P(M|I)+P(A|I)) + (P(M|P)+P(A|P))]`, and exact expected financial utility `U = 0.50 E[u|V] + 0.25 E[u|I] + 0.25 E[u|P]`. `q_V` measures valid response-set mass, `q_X` measures response pressure in invalid contexts, and `U` is a financial guardrail rather than a dominance claim. Per-context probabilities, the selectivity gap `q_V-q_X`, response-set internal allocation, parameter trajectories, and gradient norms are secondary or diagnostic only.

### Frozen contrasts and decision rule

All `50 seeds x 6 methods = 300` seed-method runs must finish with finite logits, valid probabilities, and zero silently dropped seeds. For each of the following paired seed-level contrasts, report the mean, sample standard deviation, two-sided 95% paired t interval with 49 degrees of freedom, two-sided paired t-test p-value, and Holm-adjusted p-value across the five contrasts:

- `C0`: `full_ace - vanilla_pg` on `q_V`, expected positive.
- `C1`: `full_ace - scalar_only_valid_shift` on `q_V`, expected positive (M1).
- `C2`: `full_ace - shuffled_validity_ace` on `q_V`, expected positive (M2 valid opportunity).
- `C3`: `shuffled_validity_ace - full_ace` on `q_X`, expected positive (M2 invalid pressure).
- `C4`: `full_ace - shuffled_validity_ace` on `U`, expected positive (financial guardrail).

Each two-sided 95% interval must lie strictly on the expected side of zero and each Holm-adjusted p-value must be below `0.05`. At least `45/50` paired seed differences must have the expected sign for `C0` through `C3`. Two structural invariants are also required: `vanilla_pg` and `scalar_only_valid_shift` must match at every stored checkpoint to absolute tolerance `1e-12`, and `full_ace` must match `vanilla_pg` in both invalid-context logits and response masses at every stored checkpoint to absolute tolerance `1e-12`. Any failed condition is reported as a failed condition; selectivity, always-on results, or a favorable alternative baseline cannot rescue it.

## A3: randomized critic-isolation ablation

This ablation tests M3 structurally rather than inferring it from final performance. For seeds `2000` through `2099`, construct a one-hidden-layer tanh actor with three logits and a scalar critic over one-hot encodings of the three contexts. Hidden width is `8`. Each seed uses a 64-row audit batch, response set `{M,A}`, coefficient `1.0`, one audit-only SGD step with learning rate `0.05`, and matched initial tensors.

The `full_disjoint` variant gives actor and critic separate copies of the encoder and updates actor parameters only. The `shared_encoder_no_stop` variant uses one encoder for both heads and backpropagates the audit loss through that encoder; the value head itself is not stepped. For each seed record response-set probability before and after the step, audit-gradient and parameter-delta norms for every critic-used parameter, value-head delta, and maximum absolute critic-output change across all three contexts.

The structural gate requires `100/100` `full_disjoint` seeds to increase response-set probability while critic-used gradient norm, critic-used parameter delta, and critic-output change are each at most `1e-12`. It requires `100/100` shared variants to increase response-set probability, shared-encoder parameter delta above `1e-10`, value-head delta at most `1e-12`, and critic-output change above `1e-8` in at least `95/100` seeds. Report paired means and two-sided 95% t intervals, but do not interpret critic leakage as evidence of worse financial or market performance.

## Required artifacts and execution policy

The implementation must refuse to run if this protocol hash changes. Each runner must preserve raw seed rows, learning curves or per-seed traces, a header-preserving `failures.csv`, CSV/JSON summaries, paired contrasts, a concise PDF/PNG figure where applicable, include-ready TeX tables, software versions, command, duration, code/protocol hashes, artifact hashes, and a checksum ledger. A second execution in an isolated temporary directory must reproduce core scientific artifacts byte for byte. Runtime timeout is 10 minutes per runner. No automatic retry is permitted; a crash is reported before any rerun decision.

## Claim permitted only if all relevant gates pass

The strongest permitted statement is: "In a frozen controlled quote-choice assay, set-valued audit labels provide action direction, a dose-matched validity mask concentrates pressure on admissible contexts, and parameter separation prevents audit-to-critic leakage." The experiment cannot establish market-level effect, robust absolute target attainment, a legal interpretation of behavior, or superiority over action-dependent constrained learning.
