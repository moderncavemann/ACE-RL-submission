# Original ACE-RL LOB audit-contract protocol

Status: frozen before implementation and formal execution.

## Source

The assay uses the included original ACE-RL price--time-priority order-book implementation at `ace_rl_project/envs/lob_env.py`. Its SHA-256 is `ff4fa85bbc08cf3fa275cd29e859fa6bbdbd714a43c63352f8b4aae362cfded8`. The runner refuses a different source.

## Decision states

Seeds 5000--5099 each instantiate one seven-action logit vector and three fixed decision states. The action grid is one through seven half-spread ticks. A valid low-risk state returns `z=1` and response set `{1,2,3}` ticks. An inventory-headroom block and a risk-limit block each return `z=0` and the empty set.

## Shadow audit

For a valid tuple, the runner copies the decision state, replaces only the focal agent's copied quote with the widest acceptable quote, runs matching on the copy, requires zero fills, and discards the copy. Each candidate response must be in the action grid, satisfy the price, inventory and risk checks, remain non-crossing and tick-representable, and produce zero copied-book fills.

## Actor update

Initial logits are sampled from `Normal(0,0.1)`. One audit-only gradient-descent step of size 0.1 minimizes the negative log probability of the response set when `z=1`. Loss, logits and probabilities remain exactly unchanged when `z=0`.

## Gates

All 300 typed rows must match their expected validity, reason and response set. Every candidate response must pass every action check. The recorded live-state digest must remain unchanged. Response-set probability must increase on all 100 valid rows, while all 200 invalid rows must have zero loss and zero policy change. Every copied state must be discarded with zero fills, and an isolated replay must be byte-identical for every core CSV and JSON artifact.
