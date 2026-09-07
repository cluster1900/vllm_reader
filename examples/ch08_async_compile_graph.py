#!/usr/bin/env python3
"""Dependency-free teaching models for vLLM async/compile/CUDA-graph paths.

The code models contracts visible in the pinned vLLM source revision. It does
not emulate CUDA, torch.compile, attention kernels, or their performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Iterable


class CompilationMode(IntEnum):
    NONE = 0
    STOCK_TORCH_COMPILE = 1
    DYNAMO_TRACE_ONCE = 2
    VLLM_COMPILE = 3


class GraphMode(Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (2, 0)
    FULL_AND_PIECEWISE = (2, 1)

    def separate_routine(self) -> bool:
        return isinstance(self.value, tuple)

    def decode_mode(self) -> "GraphMode":
        return GraphMode(self.value[0]) if self.separate_routine() else self

    def mixed_mode(self) -> "GraphMode":
        return GraphMode(self.value[1]) if self.separate_routine() else self

    def runtime_modes(self) -> frozenset["GraphMode"]:
        if self.separate_routine():
            return frozenset((self.decode_mode(), self.mixed_mode()))
        return frozenset((self,))


class AttentionGraphSupport(IntEnum):
    NEVER = 0
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    UNIFORM_BATCH = 2
    ALWAYS = 3


@dataclass(frozen=True)
class StageTimes:
    cpu_prepare_ms: float
    gpu_execute_ms: float

    def __post_init__(self) -> None:
        if self.cpu_prepare_ms < 0 or self.gpu_execute_ms < 0:
            raise ValueError("stage times must be non-negative")


@dataclass(frozen=True)
class TimelineItem:
    step: int
    resource: str
    start_ms: float
    end_ms: float


def synchronous_timeline(num_steps: int, times: StageTimes) -> list[TimelineItem]:
    """Build a serialized CPU-then-GPU timeline."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    now = 0.0
    result: list[TimelineItem] = []
    for step in range(num_steps):
        result.append(
            TimelineItem(step, "cpu", now, now + times.cpu_prepare_ms)
        )
        now += times.cpu_prepare_ms
        result.append(
            TimelineItem(step, "gpu", now, now + times.gpu_execute_ms)
        )
        now += times.gpu_execute_ms
    return result


def overlapped_timeline(num_steps: int, times: StageTimes) -> list[TimelineItem]:
    """Build a two-stage pipeline: CPU prepares while the prior GPU step runs."""

    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    cpu_available = 0.0
    gpu_available = 0.0
    result: list[TimelineItem] = []
    for step in range(num_steps):
        cpu_start = cpu_available
        cpu_end = cpu_start + times.cpu_prepare_ms
        cpu_available = cpu_end
        result.append(TimelineItem(step, "cpu", cpu_start, cpu_end))

        gpu_start = max(cpu_end, gpu_available)
        gpu_end = gpu_start + times.gpu_execute_ms
        gpu_available = gpu_end
        result.append(TimelineItem(step, "gpu", gpu_start, gpu_end))
    return result


def makespan(items: Iterable[TimelineItem]) -> float:
    return max((item.end_ms for item in items), default=0.0)


@dataclass
class AsyncRequestState:
    """The optimistic/confirmed boundary represented by output placeholders."""

    num_computed_tokens: int
    num_output_placeholders: int = 0
    delivered_tokens: list[int] | None = None

    def __post_init__(self) -> None:
        if self.num_computed_tokens < 0 or self.num_output_placeholders < 0:
            raise ValueError("token counts must be non-negative")
        if self.delivered_tokens is None:
            self.delivered_tokens = []

    @property
    def confirmed_tokens(self) -> int:
        return self.num_computed_tokens - self.num_output_placeholders

    def schedule_decode(self, sampled_tokens: int = 1, spec_tokens: int = 0) -> None:
        if sampled_tokens < 0 or spec_tokens < 0:
            raise ValueError("scheduled token counts must be non-negative")
        in_flight = sampled_tokens + spec_tokens
        self.num_computed_tokens += in_flight
        self.num_output_placeholders += in_flight

    def apply_output(self, token_ids: Iterable[int], *, stale: bool = False) -> None:
        token_ids = list(token_ids)
        assert self.delivered_tokens is not None
        self.delivered_tokens.extend(token_ids)
        if stale:
            return
        self.num_output_placeholders -= len(token_ids)
        if self.num_output_placeholders < 0:
            raise AssertionError("output placeholders underflow")

    def preempt(self, rollback_to: int) -> None:
        if rollback_to < 0 or rollback_to > self.num_computed_tokens:
            raise ValueError("invalid rollback point")
        self.num_computed_tokens = rollback_to
        self.num_output_placeholders = 0


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0


def pad_to_capture_size(num_tokens: int, capture_sizes: Iterable[int]) -> int | None:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    for size in sorted(set(capture_sizes)):
        if size >= num_tokens:
            return size
    return None


class GraphDispatcher:
    """Small MRV1-like dispatcher with FULL > PIECEWISE > NONE priority."""

    def __init__(
        self,
        configured_mode: GraphMode,
        capture_sizes: Iterable[int],
        *,
        max_num_reqs: int,
        decode_query_len: int = 1,
    ) -> None:
        self.configured_mode = configured_mode
        self.capture_sizes = tuple(sorted(set(capture_sizes)))
        self.max_num_reqs = max_num_reqs
        self.decode_query_len = decode_query_len
        if max_num_reqs <= 0 or decode_query_len <= 0:
            raise ValueError("request and decode sizes must be positive")

    def _descriptor(
        self, padded: int, *, uniform_decode: bool, mode: GraphMode
    ) -> BatchDescriptor:
        if mode == GraphMode.PIECEWISE:
            return BatchDescriptor(num_tokens=padded, num_reqs=None)
        if uniform_decode and self.configured_mode.separate_routine():
            if padded % self.decode_query_len:
                raise ValueError("uniform decode graph must align to query length")
            return BatchDescriptor(
                num_tokens=padded,
                num_reqs=min(padded // self.decode_query_len, self.max_num_reqs),
                uniform=True,
            )
        return BatchDescriptor(
            num_tokens=padded,
            num_reqs=min(padded, self.max_num_reqs),
            uniform=False,
        )

    def dispatch(
        self,
        num_tokens: int,
        *,
        uniform_decode: bool,
        allow_full: bool = True,
        allow_piecewise: bool = True,
    ) -> tuple[GraphMode, BatchDescriptor]:
        padded = pad_to_capture_size(num_tokens, self.capture_sizes)
        if padded is None or self.configured_mode == GraphMode.NONE:
            return GraphMode.NONE, BatchDescriptor(num_tokens)

        desired = (
            self.configured_mode.decode_mode()
            if uniform_decode and self.configured_mode.separate_routine()
            else self.configured_mode.mixed_mode()
            if self.configured_mode.separate_routine()
            else self.configured_mode
        )
        if desired == GraphMode.FULL and allow_full:
            return GraphMode.FULL, self._descriptor(
                padded, uniform_decode=uniform_decode, mode=GraphMode.FULL
            )
        if desired == GraphMode.PIECEWISE and allow_piecewise:
            return GraphMode.PIECEWISE, self._descriptor(
                padded, uniform_decode=False, mode=GraphMode.PIECEWISE
            )
        return GraphMode.NONE, BatchDescriptor(num_tokens)


def resolve_graph_mode(
    requested: GraphMode,
    support: AttentionGraphSupport,
    *,
    piecewise_available: bool,
    speculative_query_len: int = 1,
) -> GraphMode:
    """Teach the important direction of current backend-driven downgrades."""

    if requested == GraphMode.NONE:
        return GraphMode.NONE
    if support == AttentionGraphSupport.NEVER:
        return GraphMode.PIECEWISE if piecewise_available else GraphMode.NONE
    if (
        speculative_query_len > 1
        and support < AttentionGraphSupport.UNIFORM_BATCH
        and requested.decode_mode() == GraphMode.FULL
    ):
        return GraphMode.PIECEWISE if piecewise_available else GraphMode.NONE
    if requested.mixed_mode() == GraphMode.FULL and support < AttentionGraphSupport.ALWAYS:
        return (
            GraphMode.FULL_AND_PIECEWISE
            if piecewise_available
            else GraphMode.FULL_DECODE_ONLY
        )
    return requested


class StaticBuffer:
    """A stable-address buffer into which changing request data is copied."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.values = [0] * capacity

    @property
    def address(self) -> int:
        return id(self.values)

    def stage(self, values: Iterable[int]) -> int:
        values = list(values)
        if len(values) > len(self.values):
            raise ValueError("input exceeds static buffer capacity")
        self.values[: len(values)] = values
        self.values[len(values) :] = [0] * (len(self.values) - len(values))
        return len(values)


@dataclass(frozen=True)
class CapturedGraph:
    input_address: int
    captured_size: int

    @classmethod
    def capture(cls, buffer: StaticBuffer) -> "CapturedGraph":
        return cls(buffer.address, len(buffer.values))

    def replay(self, buffer: StaticBuffer, actual_size: int) -> int:
        if buffer.address != self.input_address:
            raise ValueError("CUDA Graph replay requires the captured input address")
        if not 0 <= actual_size <= self.captured_size:
            raise ValueError("runtime shape exceeds captured padded shape")
        return sum(buffer.values[:actual_size])


def compile_cache_key(factors: dict[str, Any], traced_files: dict[str, str]) -> str:
    """Hash graph-affecting config plus traced-file fingerprints."""

    payload = {
        "factors": factors,
        "traced_files": traced_files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def demo() -> None:
    times = StageTimes(cpu_prepare_ms=2.0, gpu_execute_ms=5.0)
    sync = makespan(synchronous_timeline(4, times))
    overlap = makespan(overlapped_timeline(4, times))
    print(f"four steps: sync={sync:.1f}ms overlap={overlap:.1f}ms")

    dispatcher = GraphDispatcher(
        GraphMode.FULL_AND_PIECEWISE, [1, 2, 4, 8], max_num_reqs=8
    )
    for uniform in (True, False):
        mode, desc = dispatcher.dispatch(3, uniform_decode=uniform)
        print(f"uniform={uniform}: {mode.name} {desc}")

    stable = StaticBuffer(4)
    graph = CapturedGraph.capture(stable)
    count = stable.stage([7, 8, 9])
    print("replay result:", graph.replay(stable, count))


if __name__ == "__main__":
    demo()
