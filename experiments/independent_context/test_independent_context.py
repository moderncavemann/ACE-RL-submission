#!/usr/bin/env python3
"""Structural unit tests for the frozen E-main implementation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parent / "run_main_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_main_experiment", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("could not load run_main_experiment.py")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class MainExperimentTests(unittest.TestCase):
    def test_response_set_gradient_matches_finite_difference(self) -> None:
        logits = np.asarray([0.2, -0.1, 0.4], dtype=np.float64)
        probability = experiment.softmax(logits)
        analytic = experiment.response_set_log_gradient(probability)
        epsilon = 1e-6
        finite_difference = np.zeros(3, dtype=np.float64)
        for index in range(3):
            positive = logits.copy()
            negative = logits.copy()
            positive[index] += epsilon
            negative[index] -= epsilon
            positive_probability = experiment.softmax(positive)
            negative_probability = experiment.softmax(negative)
            finite_difference[index] = (
                np.log(positive_probability[1:].sum())
                - np.log(negative_probability[1:].sum())
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, finite_difference, atol=1e-10, rtol=0.0)

    def test_scalar_shift_cancels_after_context_centering(self) -> None:
        cfg = experiment.Config(seeds=(-2, -1), updates=10)
        stream = experiment.make_seed_stream(-2, cfg)
        vanilla, _, _ = experiment.batch_gradients(
            "vanilla_pg",
            stream.initial_logits,
            stream.contexts[0],
            stream.action_uniforms[0],
            stream.reward_shocks[0],
            stream.shuffle_permutations[0],
            cfg,
        )
        shifted, _, _ = experiment.batch_gradients(
            "scalar_only_valid_shift",
            stream.initial_logits,
            stream.contexts[0],
            stream.action_uniforms[0],
            stream.reward_shocks[0],
            stream.shuffle_permutations[0],
            cfg,
        )
        np.testing.assert_allclose(vanilla, shifted, atol=1e-15, rtol=0.0)

    def test_full_ace_audit_gradient_is_valid_context_only(self) -> None:
        cfg = experiment.Config(seeds=(-2, -1), updates=10)
        stream = experiment.make_seed_stream(-1, cfg)
        _, audit, counts = experiment.batch_gradients(
            "full_ace",
            stream.initial_logits,
            stream.contexts[0],
            stream.action_uniforms[0],
            stream.reward_shocks[0],
            stream.shuffle_permutations[0],
            cfg,
        )
        self.assertGreater(np.linalg.norm(audit[0]), 0.0)
        np.testing.assert_array_equal(audit[1:], np.zeros((2, 3)))
        self.assertEqual(counts["active_audit_rows"], counts["canonical_valid_rows"])

    def test_shuffled_mask_is_exactly_dose_matched(self) -> None:
        cfg = experiment.Config(seeds=(-2, -1), updates=10)
        stream = experiment.make_seed_stream(-2, cfg)
        for update in range(cfg.updates):
            _, _, counts = experiment.batch_gradients(
                "shuffled_validity_ace",
                stream.initial_logits,
                stream.contexts[update],
                stream.action_uniforms[update],
                stream.reward_shocks[update],
                stream.shuffle_permutations[update],
                cfg,
            )
            self.assertEqual(
                counts["active_audit_rows"], counts["canonical_valid_rows"]
            )


if __name__ == "__main__":
    unittest.main()
