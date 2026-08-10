# ABIDES typed-audit IPPO v2 protocol

Date frozen: 2026-08-10
Status: frozen before v2 engineering smoke and scientific outcome inspection

## Version rationale and claim boundary

Formal v1 was terminated after five seeds triggered a redundant diagnostic that multiplied an already verified float32 scaled reward by 100 and compared the result in raw units. The direct, elementwise check of the value supplied to PPO passed; no scientific endpoint was inspected. The failed and interrupted v1 receipts remain under the v1 output directory. V2 removes only that algebraically redundant diagnostic, retains the direct learning-reward assertion, and uses fresh response-training, action-uniform, smoke, and evaluation namespaces.

This prospective experiment asks whether the ACE-RL typed audit-to-policy update remains context-selective and financially viable when ten market-making agents adapt together in the ABIDES price--time-priority market. It is an exploratory domain experiment with a parameter-sharing official IPPO backbone. Evidence is limited to the tested market, audit rule, training horizon, and seeds; it does not establish equilibrium selection, general collusion prevention, or absence of new coordination.

## Frozen backbone and environment

The policy and value networks use `marlbenchmark/on-policy` commit `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`. IPPO uses one parameter-shared recurrent actor across ten dealers and decentralized value inputs. Every arm for a policy seed starts from the same completed 384-episode official-IPPO base checkpoint from `abides_standard_learner_exact_risk_only_holdout_v1`; checkpoint paths and SHA-256 values are frozen in the JSON configuration. No response-policy checkpoint from v1 is reused.

The runtime is the pinned `ace-rl-abides-ace:abce-v1` image with image ID `sha256:351dd23c02b950cd986004ec2695901fe379e561375cc16f2548b041a3bcbff3`. The adapter explicitly preserves the base environment's `soft_rank` allocation mode while disabling order-flow routing in every arm.

## Typed audit contract and arms

Every decision row is selected for audit. A row is valid when public rolling volatility is at most 1.0 tick and the dealer's absolute inventory is at most 20 units. The valid response set is action indices `{0,1,2}`, corresponding to two-sided half-spreads of `{1,2,3}` ticks. Invalid rows have an empty typed response set and exact-zero audit loss. For measurement only, `q_V` and `q_X` both evaluate probability assigned to the fixed `{0,1,2}` set; this measurement set is not an invalid-row training label.

The discarded shadow observation replaces the focal dealer's previous quote with the widest acceptable three-tick quote and updates the observed cross-dealer mean quote accordingly. It does not call `env.step`, alter the live kernel, or change financial reward. The set loss is the complete-batch mean of `-z log sum_{a in R} pi(a | o_tilde)`.

All arms start from the identical base checkpoint and receive identical market paths and action uniforms: (1) `vanilla_ippo`, the official PPO actor and financial-return critic objectives; (2) `action_cost`, PPO with scaled learning reward `r_fin/100 - 0.20 z 1[a not in R]`, so both actor advantages and value targets consume the explicit cost; (3) `always_on_actor`, the ACE set loss with every row treated as valid and `alpha=0.30`; and (4) `full_ace`, the same set loss and coefficient with the frozen validity rule. Raw financial reward and PnL are recorded before the action-cost transformation. For the two actor-regularized arms, financial rewards and value targets remain unchanged. Actor and critic modules and optimizers are disjoint.

## Training, evaluation, and endpoints

Formal policy seeds are `560000--560009`. Each arm receives 96 response-training episodes of 100 decisions. Formal response paths use namespace `700000000`; evaluation uses 20 fresh common paths `701000000--701000019`. Training and evaluation action-uniform namespaces are `702000` and `703000`. Ten CPU workers may run policy seeds concurrently.

Engineering smoke uses only policy seed `560000`, two response episodes, response namespace `710000000`, evaluation seed `711000000`, and action namespaces `710100` and `710200`. Smoke evidence is never pooled with formal results. All v2 paths are disjoint from v1 and earlier routing experiments.

Audit endpoints are `q_V`, `q_X`, selectivity `q_V-q_X`, valid-audit rate, and invalid-row audit contribution. Market endpoints are dealer financial reward, mark-to-market PnL before inventory penalty, inventory penalty, mean absolute inventory, cross-dealer inventory variance, quoted half-spread, realized execution half-spread, fill ratio, volume, and customer execution cost. Coordination diagnostics are quote dispersion, best-quote tie share, pairwise action agreement, and cross-dealer action entropy.

Primary descriptive contrasts are `full_ace - vanilla_ippo` and `full_ace - always_on_actor` for selectivity, PnL, inventory risk, quoted spread, and action agreement. `action_cost` supplies a distinct cost-based operating point. Results are averaged over evaluation paths within policy seed, then summarized with paired means and two-sided 95% t intervals across the ten policy seeds. No seed is excluded.

## Execution gates and stopping rule

Before formal execution, unit tests must establish finite response-set mass, exact-zero invalid-row loss, increased response-set probability after a valid audit-only step, unchanged critic parameters and outputs under audit-only backpropagation, and exact equality between the frozen scaled reward formula and the value stored for PPO. The engineering smoke must complete every arm, preserve common path and action-uniform digests, produce finite endpoints, retain both audit denominators, disable routing throughout, and report zero failures.

Any exception, non-finite value, checkpoint or Docker hash mismatch, incomplete seed, common-path mismatch, or common-uniform mismatch is retained as a failure. There is no outcome-triggered retry, seed replacement, coefficient change, threshold change, training extension, or post-outcome arm removal. A failed formal v2 run remains failed; any repair requires a new protocol version and fresh response/evaluation paths.
