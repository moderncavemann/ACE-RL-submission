"""Independent checks for the frozen shared-representation assay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("shared_assay", ROOT / "run_experiment.py")
assert SPEC is not None and SPEC.loader is not None
ASSAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASSAY
SPEC.loader.exec_module(ASSAY)


def test_frozen_protocol_hash() -> None:
    assert ASSAY.sha256(ROOT / "PROTOCOL.md") == ASSAY.EXPECTED_PROTOCOL_SHA256


def test_shuffled_mask_preserves_exact_audit_dose() -> None:
    cfg = ASSAY.Config()
    for seed in (3000, 3049, 3099):
        stream = ASSAY.make_stream(seed, cfg)
        for contexts, permutation in zip(stream.contexts, stream.shuffle_indices, strict=True):
            canonical = contexts == 0
            assert int(np.sum(canonical)) == int(np.sum(canonical[permutation]))


def test_combined_gradient_matches_finite_difference() -> None:
    cfg = ASSAY.Config(seeds=(-2,), updates=1, batch_size=32)
    stream = ASSAY.make_stream(-2, cfg)
    params = stream.initial.copy()
    observations = stream.observations[0]
    contexts = stream.contexts[0]
    hidden, probabilities = ASSAY.forward(params, observations)
    actions = ASSAY.sample_actions(probabilities, stream.action_uniforms[0])
    returns = ASSAY.UTILITY[contexts, actions] + stream.reward_shocks[0]
    advantages = ASSAY.centered_advantages(contexts, returns)
    mask = (contexts == 0).astype(np.float64)

    def objective(candidate: ASSAY.Parameters) -> float:
        _, probs = ASSAY.forward(candidate, observations)
        financial = np.mean(advantages * np.log(probs[np.arange(len(actions)), actions]))
        audit = cfg.audit_alpha * np.mean(mask * np.log(np.sum(probs[:, 1:], axis=1)))
        return float(financial + audit)

    before = params.copy()
    ASSAY.update_parameters(
        params,
        observations,
        contexts,
        stream.action_uniforms[0],
        stream.reward_shocks[0],
        stream.shuffle_indices[0],
        "full_ace_shared",
        cfg,
    )
    analytic = {
        name: (getattr(params, name) - getattr(before, name)) / cfg.learning_rate
        for name in ("w1", "b1", "w2", "b2")
    }
    epsilon = 1e-6
    probes = (("w1", (0, 0)), ("w1", (2, 7)), ("b1", (3,)), ("w2", (0, 0)), ("w2", (7, 2)), ("b2", (1,)))
    for name, index in probes:
        plus = before.copy()
        minus = before.copy()
        getattr(plus, name)[index] += epsilon
        getattr(minus, name)[index] -= epsilon
        numerical = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        assert np.isclose(analytic[name][index], numerical, rtol=2e-5, atol=2e-7)

