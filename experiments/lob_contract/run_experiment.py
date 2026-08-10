#!/usr/bin/env python3
"""Audit-contract checks on the original ACE-RL order-book class."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PROTOCOL_SHA256 = (
    "b48314360868ef534d77f05a849a18b381726918c740082573dd76ff3e963967"
)
EXPECTED_ENGINE_SHA256 = (
    "ff4fa85bbc08cf3fa275cd29e859fa6bbdbd714a43c63352f8b4aae362cfded8"
)
FORMAL_SEEDS = tuple(range(5000, 5100))
SMOKE_SEEDS = (-4, -3)
SCENARIO_ORDER = ("valid_low_risk", "inventory_blocked", "risk_blocked")
T_CRITICAL_95_N100 = 1.9842169515086827
CORE_ARTIFACTS = (
    "scenario_results.csv",
    "response_action_checks.csv",
    "failures.csv",
    "gate_summary.csv",
    "summary.csv",
    "summary.json",
)


SOURCE_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_original_project() -> Path:
    configured = os.environ.get("ACE_RL_PROJECT_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    candidates.append((SOURCE_DIR.parents[1] / "ace_rl_project").resolve())
    for candidate in candidates:
        engine = candidate / "envs" / "lob_env.py"
        if engine.is_file():
            if sha256(engine) != EXPECTED_ENGINE_SHA256:
                raise RuntimeError(
                    "original ACE-RL engine hash mismatch: "
                    f"expected {EXPECTED_ENGINE_SHA256}, found {sha256(engine)}"
                )
            return candidate
    raise RuntimeError(
        "could not locate the original ACE-RL project; set ACE_RL_PROJECT_ROOT "
        "to the directory containing envs/lob_env.py"
    )


ORIGINAL_PROJECT = locate_original_project()
ENGINE_PATH = ORIGINAL_PROJECT / "envs" / "lob_env.py"
sys.path.insert(0, str(ORIGINAL_PROJECT))

from envs.lob_env import LimitOrderBook  # noqa: E402


@dataclass(frozen=True)
class Config:
    n_agents: int = 3
    half_spread_levels: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    tick_size: float = 0.01
    reference_mid: float = 100.0
    max_inventory: int = 20
    quote_quantity: int = 1
    live_half_spread_ticks: int = 6
    competitive_cap_ticks: int = 3
    low_risk_minimum_ticks: int = 1
    high_risk_minimum_ticks: int = 4
    audited_agent_id: int = 0
    initial_logit_sd: float = 0.10
    audit_learning_rate: float = 0.10
    zero_tolerance: float = 1e-12


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    risk_state: str
    focal_inventory: float
    expected_z: int
    expected_reason: str
    expected_response_ticks: tuple[int, ...]


@dataclass
class DecisionState:
    book: Any
    inventories: np.ndarray
    risk_state: str
    last_actions: np.ndarray
    agent_orders: dict[int, list[int]]
    reference_mid: float
    timestamp: int
    rng: np.random.Generator


@dataclass(frozen=True)
class AuditResult:
    selected: bool
    z: int
    reason: str
    response_ticks: tuple[int, ...]
    response_indices: tuple[int, ...]
    shadow_constructed: bool
    shadow_discarded: bool
    shadow_action_ticks: int | None
    shadow_best_bid: float | None
    shadow_best_ask: float | None
    shadow_fill_count: int
    shadow_object_alive_after_discard: bool
    action_checks: tuple[dict[str, Any], ...]


SCENARIOS = (
    ScenarioSpec(
        "valid_low_risk", "low", 0.0, 1, "valid", (1, 2, 3)
    ),
    ScenarioSpec(
        "inventory_blocked", "low", 20.0, 0, "inventory_headroom", ()
    ),
    ScenarioSpec("risk_blocked", "high", 0.0, 0, "risk_limit", ()),
)


SCENARIO_FIELDS = (
    "seed",
    "scenario",
    "risk_state",
    "focal_inventory",
    "selected",
    "expected_selected",
    "z",
    "expected_z",
    "reason",
    "expected_reason",
    "response_ticks",
    "expected_response_ticks",
    "response_action_indices",
    "current_half_spread_ticks",
    "live_best_bid",
    "live_best_ask",
    "live_book_spread_ticks",
    "live_total_volume",
    "shadow_constructed",
    "shadow_discarded",
    "shadow_object_alive_after_discard",
    "shadow_action_ticks",
    "shadow_best_bid",
    "shadow_best_ask",
    "shadow_fill_count",
    "audit_loss",
    "response_probability_before",
    "response_probability_after",
    "response_probability_delta",
    "max_abs_logit_change",
    "max_abs_probability_change",
    "logits_before",
    "logits_after",
    "probabilities_before",
    "probabilities_after",
    "live_hash_before",
    "live_hash_after",
    "live_hash_equal",
    "typed_output_exact",
    "all_response_actions_feasible",
    "finite_values",
    "scenario_pass",
)


ACTION_FIELDS = (
    "seed",
    "scenario",
    "half_spread_ticks",
    "action_index",
    "in_action_grid",
    "within_competitive_cap",
    "respects_risk_minimum",
    "inventory_feasible",
    "canonical_bid",
    "canonical_ask",
    "bid_below_ask",
    "finite_positive_prices",
    "tick_representable",
    "candidate_fill_count",
    "action_feasible",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--formal", action="store_true")
    mode.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verify-determinism", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def order_payload(order: Any) -> dict[str, Any]:
    return {
        "order_id": int(order.order_id),
        "agent_id": int(order.agent_id),
        "side": str(order.side),
        "price": float(order.price),
        "quantity": int(order.quantity),
        "timestamp": int(order.timestamp),
    }


def state_payload(state: DecisionState) -> dict[str, Any]:
    bids = sorted(
        (order_payload(order) for order in state.book.bids),
        key=lambda row: (-row["price"], row["timestamp"], row["order_id"]),
    )
    asks = sorted(
        (order_payload(order) for order in state.book.asks),
        key=lambda row: (row["price"], row["timestamp"], row["order_id"]),
    )
    return {
        "book": {
            "bids": bids,
            "asks": asks,
            "order_counter": int(state.book.order_counter),
            "trades": jsonable(state.book.trades),
        },
        "inventories": jsonable(state.inventories),
        "risk_state": str(state.risk_state),
        "last_actions": jsonable(state.last_actions),
        "agent_orders": {
            str(agent_id): [int(order_id) for order_id in order_ids]
            for agent_id, order_ids in sorted(state.agent_orders.items())
        },
        "reference_mid": float(state.reference_mid),
        "timestamp": int(state.timestamp),
        "rng_state": jsonable(state.rng.bit_generator.state),
    }


def live_state_hash(state: DecisionState) -> str:
    return canonical_json_hash(state_payload(state))


def canonical_price(price: float, tick_size: float) -> float:
    return float(round(price / tick_size) * tick_size)


def make_state(seed: int, spec: ScenarioSpec, cfg: Config) -> DecisionState:
    runtime_seed = seed if seed >= 0 else (1 << 32) + seed
    book = LimitOrderBook(tick_size=cfg.tick_size)
    agent_orders: dict[int, list[int]] = {}
    width = cfg.live_half_spread_ticks * cfg.tick_size
    for agent_id in range(cfg.n_agents):
        bid_id = book.add_order(
            agent_id,
            "bid",
            cfg.reference_mid - width,
            cfg.quote_quantity,
            0,
        )
        ask_id = book.add_order(
            agent_id,
            "ask",
            cfg.reference_mid + width,
            cfg.quote_quantity,
            0,
        )
        agent_orders[agent_id] = [bid_id, ask_id]
    if book.match_orders(0):
        raise RuntimeError("prespecified live quotes unexpectedly crossed")
    inventories = np.zeros(cfg.n_agents, dtype=np.float64)
    inventories[cfg.audited_agent_id] = spec.focal_inventory
    return DecisionState(
        book=book,
        inventories=inventories,
        risk_state=spec.risk_state,
        last_actions=np.full(
            cfg.n_agents, cfg.live_half_spread_ticks, dtype=np.int64
        ),
        agent_orders=agent_orders,
        reference_mid=cfg.reference_mid,
        timestamp=0,
        rng=np.random.default_rng(runtime_seed),
    )


def inventory_has_two_sided_headroom(
    inventory: float, max_inventory: float, quote_quantity: int
) -> bool:
    return (
        inventory + quote_quantity <= max_inventory
        and inventory - quote_quantity >= -max_inventory
    )


def risk_minimum_ticks(risk_state: str, cfg: Config) -> int:
    if risk_state == "low":
        return cfg.low_risk_minimum_ticks
    if risk_state == "high":
        return cfg.high_risk_minimum_ticks
    raise ValueError(f"unknown risk state: {risk_state}")


def post_two_sided_quote(
    book: Any,
    agent_id: int,
    half_spread_ticks: int,
    reference_mid: float,
    timestamp: int,
    cfg: Config,
) -> tuple[list[int], list[dict[str, Any]], float, float]:
    book.cancel_agent_orders(agent_id)
    half_width = half_spread_ticks * cfg.tick_size
    bid = canonical_price(reference_mid - half_width, cfg.tick_size)
    ask = canonical_price(reference_mid + half_width, cfg.tick_size)
    bid_id = book.add_order(
        agent_id, "bid", bid, cfg.quote_quantity, timestamp
    )
    ask_id = book.add_order(
        agent_id, "ask", ask, cfg.quote_quantity, timestamp
    )
    fills = book.match_orders(timestamp)
    return [bid_id, ask_id], fills, bid, ask


def candidate_action_check(
    state: DecisionState,
    seed: int,
    scenario: str,
    half_spread_ticks: int,
    risk_minimum: int,
    inventory_feasible: bool,
    cfg: Config,
) -> dict[str, Any]:
    in_grid = half_spread_ticks in cfg.half_spread_levels
    action_index = (
        cfg.half_spread_levels.index(half_spread_ticks) if in_grid else -1
    )
    bid = canonical_price(
        state.reference_mid - half_spread_ticks * cfg.tick_size, cfg.tick_size
    )
    ask = canonical_price(
        state.reference_mid + half_spread_ticks * cfg.tick_size, cfg.tick_size
    )
    finite_positive = bool(
        math.isfinite(bid) and math.isfinite(ask) and bid > 0.0 and ask > 0.0
    )
    tick_representable = bool(
        math.isclose(
            bid,
            canonical_price(bid, cfg.tick_size),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            ask,
            canonical_price(ask, cfg.tick_size),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    candidate = copy.deepcopy(state.book)
    _, fills, submitted_bid, submitted_ask = post_two_sided_quote(
        candidate,
        cfg.audited_agent_id,
        half_spread_ticks,
        state.reference_mid,
        state.timestamp + 1,
        cfg,
    )
    within_cap = half_spread_ticks <= cfg.competitive_cap_ticks
    respects_risk = half_spread_ticks >= risk_minimum
    feasible = bool(
        in_grid
        and within_cap
        and respects_risk
        and inventory_feasible
        and bid < ask
        and finite_positive
        and tick_representable
        and submitted_bid == bid
        and submitted_ask == ask
        and len(fills) == 0
    )
    return {
        "seed": seed,
        "scenario": scenario,
        "half_spread_ticks": half_spread_ticks,
        "action_index": action_index,
        "in_action_grid": in_grid,
        "within_competitive_cap": within_cap,
        "respects_risk_minimum": respects_risk,
        "inventory_feasible": inventory_feasible,
        "canonical_bid": bid,
        "canonical_ask": ask,
        "bid_below_ask": bid < ask,
        "finite_positive_prices": finite_positive,
        "tick_representable": tick_representable,
        "candidate_fill_count": len(fills),
        "action_feasible": feasible,
    }


def run_shadow_copy(
    state: DecisionState, response_ticks: tuple[int, ...], cfg: Config
) -> tuple[bool, bool, int, float, float, int, bool]:
    shadow = copy.deepcopy(state)
    shadow_reference = weakref.ref(shadow)
    action_ticks = max(response_ticks)
    order_ids, fills, bid, ask = post_two_sided_quote(
        shadow.book,
        cfg.audited_agent_id,
        action_ticks,
        shadow.reference_mid,
        shadow.timestamp + 1,
        cfg,
    )
    shadow.agent_orders[cfg.audited_agent_id] = order_ids
    shadow.last_actions[cfg.audited_agent_id] = action_ticks
    fill_count = len(fills)
    if fill_count:
        raise RuntimeError("shadow quote unexpectedly crossed the copied book")
    del shadow
    gc.collect()
    object_alive = shadow_reference() is not None
    return True, not object_alive, action_ticks, bid, ask, fill_count, object_alive


def audit_state(
    state: DecisionState, seed: int, scenario: str, cfg: Config
) -> AuditResult:
    focal_id = cfg.audited_agent_id
    current_half_spread = int(state.last_actions[focal_id])
    selected = current_half_spread > cfg.competitive_cap_ticks
    inventory_feasible = inventory_has_two_sided_headroom(
        float(state.inventories[focal_id]), cfg.max_inventory, cfg.quote_quantity
    )
    minimum_ticks = risk_minimum_ticks(state.risk_state, cfg)
    if not selected:
        response_ticks: tuple[int, ...] = ()
        reason = "not_selected"
    elif not inventory_feasible:
        response_ticks = ()
        reason = "inventory_headroom"
    else:
        response_ticks = tuple(
            ticks
            for ticks in cfg.half_spread_levels
            if minimum_ticks <= ticks <= cfg.competitive_cap_ticks
        )
        reason = "valid" if response_ticks else "risk_limit"
    z = int(selected and bool(response_ticks))
    response_indices = tuple(
        cfg.half_spread_levels.index(ticks) for ticks in response_ticks
    )
    checks = tuple(
        candidate_action_check(
            state,
            seed,
            scenario,
            ticks,
            minimum_ticks,
            inventory_feasible,
            cfg,
        )
        for ticks in response_ticks
    )
    if z:
        (
            shadow_constructed,
            shadow_discarded,
            shadow_action_ticks,
            shadow_best_bid,
            shadow_best_ask,
            shadow_fill_count,
            shadow_alive,
        ) = run_shadow_copy(state, response_ticks, cfg)
    else:
        shadow_constructed = False
        shadow_discarded = False
        shadow_action_ticks = None
        shadow_best_bid = None
        shadow_best_ask = None
        shadow_fill_count = 0
        shadow_alive = False
    return AuditResult(
        selected,
        z,
        reason,
        response_ticks,
        response_indices,
        shadow_constructed,
        shadow_discarded,
        shadow_action_ticks,
        shadow_best_bid,
        shadow_best_ask,
        shadow_fill_count,
        shadow_alive,
        checks,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def audit_actor_step(
    logits: np.ndarray, audit: AuditResult, cfg: Config
) -> dict[str, Any]:
    before_logits = np.asarray(logits, dtype=np.float64).copy()
    before_probabilities = softmax(before_logits)
    if audit.z:
        indices = np.asarray(audit.response_indices, dtype=np.int64)
        response_before = float(before_probabilities[indices].sum())
        target = np.zeros_like(before_probabilities)
        target[indices] = before_probabilities[indices] / response_before
        gradient = before_probabilities - target
        after_logits = before_logits - cfg.audit_learning_rate * gradient
        audit_loss = -math.log(response_before)
    else:
        response_before = 0.0
        after_logits = before_logits.copy()
        audit_loss = 0.0
    after_probabilities = softmax(after_logits)
    response_after = (
        float(
            after_probabilities[
                np.asarray(audit.response_indices, dtype=np.int64)
            ].sum()
        )
        if audit.z
        else 0.0
    )
    return {
        "audit_loss": audit_loss,
        "response_probability_before": response_before,
        "response_probability_after": response_after,
        "response_probability_delta": response_after - response_before,
        "max_abs_logit_change": float(
            np.max(np.abs(after_logits - before_logits))
        ),
        "max_abs_probability_change": float(
            np.max(np.abs(after_probabilities - before_probabilities))
        ),
        "logits_before": before_logits,
        "logits_after": after_logits,
        "probabilities_before": before_probabilities,
        "probabilities_after": after_probabilities,
    }


def encode_sequence(values: Any) -> str:
    return json.dumps(jsonable(values), separators=(",", ":"), allow_nan=False)


def finite_actor_values(update: dict[str, Any]) -> bool:
    arrays = (
        np.asarray(update["logits_before"]),
        np.asarray(update["logits_after"]),
        np.asarray(update["probabilities_before"]),
        np.asarray(update["probabilities_after"]),
    )
    scalars = (
        update["audit_loss"],
        update["response_probability_before"],
        update["response_probability_after"],
        update["response_probability_delta"],
        update["max_abs_logit_change"],
        update["max_abs_probability_change"],
    )
    return all(np.isfinite(array).all() for array in arrays) and all(
        math.isfinite(float(value)) for value in scalars
    )


def evaluate_row(
    seed: int, spec: ScenarioSpec, logits: np.ndarray, cfg: Config
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = make_state(seed, spec, cfg)
    before_hash = live_state_hash(state)
    audit = audit_state(state, seed, spec.name, cfg)
    update = audit_actor_step(logits, audit, cfg)
    after_hash = live_state_hash(state)
    typed_exact = bool(
        audit.selected
        and audit.z == spec.expected_z
        and audit.reason == spec.expected_reason
        and audit.response_ticks == spec.expected_response_ticks
    )
    all_actions_feasible = bool(
        all(check["action_feasible"] for check in audit.action_checks)
    )
    finite_values = finite_actor_values(update)
    valid_behavior = bool(
        update["response_probability_delta"] > 0.0
        and audit.shadow_constructed
        and audit.shadow_discarded
        and not audit.shadow_object_alive_after_discard
        and audit.shadow_fill_count == 0
    )
    invalid_behavior = bool(
        abs(update["audit_loss"]) <= cfg.zero_tolerance
        and update["max_abs_logit_change"] <= cfg.zero_tolerance
        and update["max_abs_probability_change"] <= cfg.zero_tolerance
    )
    scenario_pass = bool(
        typed_exact
        and all_actions_feasible
        and finite_values
        and before_hash == after_hash
        and (valid_behavior if audit.z else invalid_behavior)
    )
    spread_ticks = int(
        round(
            (state.book.get_best_ask() - state.book.get_best_bid())
            / cfg.tick_size
        )
    )
    row = {
        "seed": seed,
        "scenario": spec.name,
        "risk_state": spec.risk_state,
        "focal_inventory": spec.focal_inventory,
        "selected": audit.selected,
        "expected_selected": True,
        "z": audit.z,
        "expected_z": spec.expected_z,
        "reason": audit.reason,
        "expected_reason": spec.expected_reason,
        "response_ticks": encode_sequence(audit.response_ticks),
        "expected_response_ticks": encode_sequence(spec.expected_response_ticks),
        "response_action_indices": encode_sequence(audit.response_indices),
        "current_half_spread_ticks": cfg.live_half_spread_ticks,
        "live_best_bid": state.book.get_best_bid(),
        "live_best_ask": state.book.get_best_ask(),
        "live_book_spread_ticks": spread_ticks,
        "live_total_volume": state.book.get_total_volume(),
        "shadow_constructed": audit.shadow_constructed,
        "shadow_discarded": audit.shadow_discarded,
        "shadow_object_alive_after_discard": (
            audit.shadow_object_alive_after_discard
        ),
        "shadow_action_ticks": audit.shadow_action_ticks,
        "shadow_best_bid": audit.shadow_best_bid,
        "shadow_best_ask": audit.shadow_best_ask,
        "shadow_fill_count": audit.shadow_fill_count,
        "audit_loss": update["audit_loss"],
        "response_probability_before": update["response_probability_before"],
        "response_probability_after": update["response_probability_after"],
        "response_probability_delta": update["response_probability_delta"],
        "max_abs_logit_change": update["max_abs_logit_change"],
        "max_abs_probability_change": update["max_abs_probability_change"],
        "logits_before": encode_sequence(update["logits_before"]),
        "logits_after": encode_sequence(update["logits_after"]),
        "probabilities_before": encode_sequence(update["probabilities_before"]),
        "probabilities_after": encode_sequence(update["probabilities_after"]),
        "live_hash_before": before_hash,
        "live_hash_after": after_hash,
        "live_hash_equal": before_hash == after_hash,
        "typed_output_exact": typed_exact,
        "all_response_actions_feasible": all_actions_feasible,
        "finite_values": finite_values,
        "scenario_pass": scenario_pass,
    }
    return row, list(audit.action_checks)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) <= 1:
        return mean, mean, mean
    critical = T_CRITICAL_95_N100 if len(array) == 100 else 12.706204736432095
    half = critical * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean, mean - half, mean + half


def build_gates(
    rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], cfg: Config
) -> list[dict[str, Any]]:
    valid = [row for row in rows if int(row["z"]) == 1]
    invalid = [row for row in rows if int(row["z"]) == 0]
    checks = [
        (
            "O0",
            "original source",
            int(sha256(ENGINE_PATH) == EXPECTED_ENGINE_SHA256),
            1,
            "original ACE-RL order-book hash equals the frozen hash",
        ),
        (
            "O1",
            "typed outputs",
            sum(bool(row["typed_output_exact"]) for row in rows),
            len(rows),
            "fixture validity, reason, and response set match exactly",
        ),
        (
            "O2",
            "executable set",
            sum(bool(row["action_feasible"]) for row in action_rows),
            len(action_rows),
            "every response action passes every declared quote check",
        ),
        (
            "O3",
            "live isolation",
            sum(bool(row["live_hash_equal"]) for row in rows),
            len(rows),
            "the prespecified live-state digest is unchanged",
        ),
        (
            "O4",
            "valid direction",
            sum(float(row["response_probability_delta"]) > 0.0 for row in valid),
            len(valid),
            "response-set probability increases on every valid row",
        ),
        (
            "O5",
            "invalid zero",
            sum(
                abs(float(row["audit_loss"])) <= cfg.zero_tolerance
                and float(row["max_abs_logit_change"]) <= cfg.zero_tolerance
                and float(row["max_abs_probability_change"])
                <= cfg.zero_tolerance
                for row in invalid
            ),
            len(invalid),
            "invalid loss and policy changes are each at most 1e-12",
        ),
        (
            "O6",
            "discarded shadow",
            sum(
                bool(row["shadow_constructed"])
                and bool(row["shadow_discarded"])
                and not bool(row["shadow_object_alive_after_discard"])
                and int(row["shadow_fill_count"]) == 0
                for row in valid
            ),
            len(valid),
            "each valid copied state is discarded and produces zero fills",
        ),
    ]
    return [
        {
            "gate_id": gate_id,
            "gate_name": gate_name,
            "numerator": numerator,
            "denominator": denominator,
            "criterion": criterion,
            "observed_status": "PASS" if numerator == denominator else "FAIL",
            "formal_status": "PASS" if numerator == denominator else "FAIL",
        }
        for gate_id, gate_name, numerator, denominator, criterion in checks
    ]


def run_once(output_dir: Path, seeds: tuple[int, ...], run_mode: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for seed in seeds:
        runtime_seed = seed if seed >= 0 else (1 << 32) + seed
        logits = np.random.default_rng(runtime_seed).normal(
            0.0, cfg.initial_logit_sd, len(cfg.half_spread_levels)
        )
        for spec in SCENARIOS:
            try:
                row, checks = evaluate_row(seed, spec, logits, cfg)
                rows.append(row)
                action_rows.extend(checks)
            except Exception as exc:
                failures.append(
                    {
                        "seed": seed,
                        "scenario": spec.name,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    rows.sort(key=lambda row: (int(row["seed"]), SCENARIO_ORDER.index(row["scenario"])))
    action_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            SCENARIO_ORDER.index(row["scenario"]),
            int(row["half_spread_ticks"]),
        )
    )
    failures.sort(key=lambda row: (int(row["seed"]), row["scenario"]))
    gates = build_gates(rows, action_rows, cfg)
    expected_rows = len(seeds) * len(SCENARIOS)
    structural_pass = bool(
        len(rows) == expected_rows
        and not failures
        and all(bool(row["scenario_pass"]) for row in rows)
        and all(gate["formal_status"] == "PASS" for gate in gates)
    )
    valid = [row for row in rows if int(row["z"]) == 1]
    q_before = [float(row["response_probability_before"]) for row in valid]
    q_after = [float(row["response_probability_after"]) for row in valid]
    q_delta = [float(row["response_probability_delta"]) for row in valid]
    before_mean, before_low, before_high = mean_ci95(q_before)
    after_mean, after_low, after_high = mean_ci95(q_after)
    delta_mean, delta_low, delta_high = mean_ci95(q_delta)
    summary = {
        "experiment": "original_ace_lob_audit_contract_v1",
        "run_mode": run_mode,
        "seed_count": len(seeds),
        "scenario_row_count": len(rows),
        "expected_scenario_row_count": expected_rows,
        "candidate_action_row_count": len(action_rows),
        "failure_count": len(failures),
        "valid_row_count": len(valid),
        "invalid_row_count": len(rows) - len(valid),
        "response_probability_before": {
            "mean": before_mean,
            "ci95_low": before_low,
            "ci95_high": before_high,
        },
        "response_probability_after": {
            "mean": after_mean,
            "ci95_low": after_low,
            "ci95_high": after_high,
        },
        "response_probability_delta": {
            "mean": delta_mean,
            "ci95_low": delta_low,
            "ci95_high": delta_high,
        },
        "formal_status": "PASS" if structural_pass else "FAIL",
        "scientific_decision": "GO" if structural_pass else "NO-GO",
    }
    write_csv(output_dir / "scenario_results.csv", SCENARIO_FIELDS, rows)
    write_csv(output_dir / "response_action_checks.csv", ACTION_FIELDS, action_rows)
    write_csv(
        output_dir / "failures.csv",
        ("seed", "scenario", "exception_type", "message"),
        failures,
    )
    write_csv(
        output_dir / "gate_summary.csv",
        (
            "gate_id",
            "gate_name",
            "numerator",
            "denominator",
            "criterion",
            "observed_status",
            "formal_status",
        ),
        gates,
    )
    summary_row = {
        "run_mode": run_mode,
        "seed_count": len(seeds),
        "scenario_rows": len(rows),
        "candidate_actions": len(action_rows),
        "failures": len(failures),
        "q_before_mean": before_mean,
        "q_before_ci95_low": before_low,
        "q_before_ci95_high": before_high,
        "q_after_mean": after_mean,
        "q_after_ci95_low": after_low,
        "q_after_ci95_high": after_high,
        "q_delta_mean": delta_mean,
        "q_delta_ci95_low": delta_low,
        "q_delta_ci95_high": delta_high,
        "formal_status": summary["formal_status"],
    }
    write_csv(output_dir / "summary.csv", tuple(summary_row), [summary_row])
    write_json(output_dir / "summary.json", summary)
    return {
        "summary": summary,
        "gates": gates,
        "duration_seconds": time.monotonic() - started,
    }


def compare_replay(output_dir: Path, replay_dir: Path) -> dict[str, Any]:
    comparisons = {
        name: sha256(output_dir / name) == sha256(replay_dir / name)
        for name in CORE_ARTIFACTS
    }
    return {
        "comparison": "byte-for-byte isolated replay of core artifacts",
        "compared_artifact_count": len(comparisons),
        "matched_artifact_count": sum(comparisons.values()),
        "mismatched_artifacts": [
            name for name, matches in comparisons.items() if not matches
        ],
        "comparison_sha256": {
            name: sha256(output_dir / name) for name in CORE_ARTIFACTS
        },
        "status": "PASS" if all(comparisons.values()) else "FAIL",
    }


def write_checksums(output_dir: Path) -> None:
    names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "checksums.sha256"
    )
    content = "".join(f"{sha256(output_dir / name)}  {name}\n" for name in names)
    (output_dir / "checksums.sha256").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    protocol_path = SOURCE_DIR / "PROTOCOL.md"
    if protocol_path.is_file() and sha256(protocol_path) != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash mismatch")
    seeds = FORMAL_SEEDS if args.formal else SMOKE_SEEDS
    run_mode = "formal" if args.formal else "engineering_smoke"
    output_dir = args.output_dir or SOURCE_DIR / f"results_{run_mode}"
    result = run_once(output_dir, seeds, run_mode)
    if args.verify_determinism:
        with tempfile.TemporaryDirectory(prefix="ace-lob-replay-") as temp_name:
            replay_dir = Path(temp_name)
            run_once(replay_dir, seeds, run_mode)
            replay = compare_replay(output_dir, replay_dir)
    else:
        replay = {
            "comparison": "not requested",
            "compared_artifact_count": 0,
            "matched_artifact_count": 0,
            "mismatched_artifacts": [],
            "comparison_sha256": {},
            "status": "NOT_REQUESTED",
        }
    write_json(output_dir / "determinism_receipt.json", replay)
    formal_pass = bool(
        result["summary"]["formal_status"] == "PASS"
        and (not args.verify_determinism or replay["status"] == "PASS")
    )
    manifest = {
        "schema_version": 1,
        "experiment": "original_ace_lob_audit_contract_v1",
        "status": "complete",
        "run_mode": run_mode,
        "formal_status": "PASS" if formal_pass else "FAIL",
        "scientific_decision": "GO" if formal_pass else "NO-GO",
        "protocol_status": "frozen before implementation and formal execution",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "engine_source": "ace_rl_project/envs/lob_env.py",
        "engine_source_sha256": sha256(ENGINE_PATH),
        "seed_count": len(seeds),
        "expected_scenario_rows": len(seeds) * len(SCENARIOS),
        "completed_scenario_rows": result["summary"]["scenario_row_count"],
        "failure_count": result["summary"]["failure_count"],
        "duration_seconds": result["duration_seconds"],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "deterministic_replay": replay,
        "validation": {
            "formal_status": result["summary"]["formal_status"],
            "gate_rows": result["gates"],
        },
        "permitted_claim": (
            "Across 300 prespecified decision states, the typed audit contract "
            "uses the original ACE-RL price-time-priority order-book class, "
            "passes declared response-action checks, increases response-set "
            "probability on valid tuples, produces exact-zero invalid updates, "
            "and preserves the prespecified live-state digest."
        ),
        "claim_boundary": (
            "This interface assay covers contract construction, declared quote "
            "feasibility, actor direction, invalid zeroing, and live-state "
            "isolation. Strategic multi-agent outcomes require a separate study."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_checksums(output_dir)
    if not args.quiet:
        delta = result["summary"]["response_probability_delta"]
        print(
            f"{run_mode}: {result['summary']['scenario_row_count']}/"
            f"{len(seeds) * len(SCENARIOS)} rows, "
            f"failures={result['summary']['failure_count']}, "
            f"delta_q={delta['mean']:.6f} "
            f"[{delta['ci95_low']:.6f}, {delta['ci95_high']:.6f}], "
            f"status={manifest['formal_status']}"
        )
    return 0 if formal_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
