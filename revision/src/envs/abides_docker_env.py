"""Host-side client for the persistent ABIDES Docker worker."""

from __future__ import annotations

import json
import select
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ABIDESDockerConfig:
    image: str = "ace-rl-abides-ace:f9cbe51"
    platform: str = "linux/amd64"
    worker_path: str = "/opt/ace-abides/worker.py"
    request_timeout_seconds: float = 120.0


class ABIDESDockerDealerEnv:
    """A fail-closed reset/step facade over one persistent container."""

    def __init__(
        self,
        market_config: Mapping[str, Any],
        docker_config: ABIDESDockerConfig = ABIDESDockerConfig(),
    ) -> None:
        self.market_config = dict(market_config)
        self.docker_config = docker_config
        self._container_name = f"ace-rl-abides-{uuid.uuid4().hex}"
        self._stderr = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                self._container_name,
                "--platform",
                docker_config.platform,
                "--entrypoint",
                "python",
                docker_config.image,
                docker_config.worker_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            bufsize=1,
        )
        self._closed = False
        self._episode_open = False
        self._worker_healthy = True

    def reset(self, seed: int) -> dict[str, Any]:
        response = self._request(
            {"cmd": "reset", "config": self.market_config, "seed": int(seed)}
        )
        self._episode_open = True
        state = response["state"]
        self._validate_state(state)
        return state

    def step(
        self,
        actions: Sequence[int],
        gate_active: bool = False,
        routing_enabled: bool = True,
        routing_strength: float | None = None,
        routing_mechanism: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._episode_open:
            raise RuntimeError("reset must be called before step")
        payload = {
            "cmd": "step",
            "actions": [int(value) for value in actions],
            "gate_active": bool(gate_active),
            "routing_enabled": bool(routing_enabled),
        }
        if routing_strength is not None:
            payload["routing_strength"] = float(routing_strength)
        if routing_mechanism is not None:
            payload["routing_mechanism"] = dict(routing_mechanism)
        response = self._request(payload)
        state = response["state"]
        self._validate_state(state)
        return state

    def close_episode(self) -> None:
        if self._closed or not self._episode_open:
            return
        self._request({"cmd": "close_episode"})
        self._episode_open = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None and self._worker_healthy:
                try:
                    self._request({"cmd": "shutdown"})
                except (OSError, RuntimeError, TimeoutError):
                    self._worker_healthy = False
        finally:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
            if self._process.poll() is None:
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=5)
            try:
                subprocess.run(
                    ["docker", "rm", "-f", self._container_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._stderr.close()
            self._closed = True
            self._episode_open = False

    def _request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("ABIDES Docker worker is closed")
        if self._process.poll() is not None:
            raise RuntimeError(self._worker_failure("worker exited before request"))
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(
            json.dumps(dict(payload), separators=(",", ":")) + "\n"
        )
        self._process.stdin.flush()
        ready, _, _ = select.select(
            [self._process.stdout],
            [],
            [],
            self.docker_config.request_timeout_seconds,
        )
        if not ready:
            self._worker_healthy = False
            raise TimeoutError(
                self._worker_failure(
                    f"worker response exceeded {self.docker_config.request_timeout_seconds}s"
                )
            )
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(self._worker_failure("worker returned no response"))
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError(self._worker_failure(str(response.get("error"))))
        return response

    def _worker_failure(self, message: str) -> str:
        self._stderr.flush()
        self._stderr.seek(0)
        details = self._stderr.read().strip()
        suffix = f"; stderr={details[-4000:]}" if details else ""
        return f"{message}{suffix}"

    def _validate_state(self, state: Mapping[str, Any]) -> None:
        if state.get("backend") != "abides":
            raise RuntimeError("worker response is not from ABIDES")
        observations = state.get("observations")
        expected = int(self.market_config.get("n_agents", 10))
        if not isinstance(observations, list) or len(observations) != expected:
            raise RuntimeError(
                f"worker returned {len(observations) if isinstance(observations, list) else 'invalid'} "
                f"agent observations; expected {expected}"
            )

    def __enter__(self) -> "ABIDESDockerDealerEnv":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
