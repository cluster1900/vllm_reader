#!/usr/bin/env python3
"""Dependency-free teaching models for vLLM distributed inference.

The code mirrors topology and algebraic contracts in the pinned source tree.
It does not start processes, initialize NCCL, or predict real communication
performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


Number = int | float
Vector = list[float]
Matrix = list[Vector]


@dataclass(frozen=True)
class RankCoordinate:
    dp: int
    pp: int
    pcp: int
    tp: int


@dataclass(frozen=True)
class ParallelTopology:
    """Rank layout matching ``DP x PP x PCP x TP`` in parallel_state.py."""

    tp: int = 1
    pp: int = 1
    dp: int = 1
    pcp: int = 1
    dcp: int = 1

    def __post_init__(self) -> None:
        if min(self.tp, self.pp, self.dp, self.pcp, self.dcp) < 1:
            raise ValueError("parallel sizes must be positive")
        if self.pcp == 1:
            if self.tp % self.dcp:
                raise ValueError("TP must be divisible by DCP when PCP is disabled")
        elif self.dcp not in (1, self.pcp, self.tp * self.pcp):
            raise ValueError("DCP must be 1, PCP, or TP*PCP when PCP is enabled")

    @property
    def worker_world_size(self) -> int:
        """Workers managed by one DP EngineCore: TP * PP * PCP."""

        return self.tp * self.pp * self.pcp

    @property
    def world_size_across_dp(self) -> int:
        return self.worker_world_size * self.dp

    def rank(self, coordinate: RankCoordinate) -> int:
        self._validate_coordinate(coordinate)
        return (
            ((coordinate.dp * self.pp + coordinate.pp) * self.pcp + coordinate.pcp)
            * self.tp
            + coordinate.tp
        )

    def coordinate(self, rank: int) -> RankCoordinate:
        if not 0 <= rank < self.world_size_across_dp:
            raise ValueError("rank is outside the topology")
        tp = rank % self.tp
        rank //= self.tp
        pcp = rank % self.pcp
        rank //= self.pcp
        pp = rank % self.pp
        dp = rank // self.pp
        return RankCoordinate(dp=dp, pp=pp, pcp=pcp, tp=tp)

    def _validate_coordinate(self, coordinate: RankCoordinate) -> None:
        for value, size, name in (
            (coordinate.dp, self.dp, "dp"),
            (coordinate.pp, self.pp, "pp"),
            (coordinate.pcp, self.pcp, "pcp"),
            (coordinate.tp, self.tp, "tp"),
        ):
            if not 0 <= value < size:
                raise ValueError(f"{name} coordinate is outside the topology")

    def _all_coordinates(self) -> Iterable[RankCoordinate]:
        for dp in range(self.dp):
            for pp in range(self.pp):
                for pcp in range(self.pcp):
                    for tp in range(self.tp):
                        yield RankCoordinate(dp, pp, pcp, tp)

    def groups(self, axis: str) -> list[list[int]]:
        """Build TP/PP/DP/PCP/DCP/EP groups from the rank tensor."""

        if axis == "tp":
            return [
                [self.rank(RankCoordinate(dp, pp, pcp, tp)) for tp in range(self.tp)]
                for dp in range(self.dp)
                for pp in range(self.pp)
                for pcp in range(self.pcp)
            ]
        if axis == "pcp":
            return [
                [self.rank(RankCoordinate(dp, pp, pcp, tp)) for pcp in range(self.pcp)]
                for dp in range(self.dp)
                for pp in range(self.pp)
                for tp in range(self.tp)
            ]
        if axis == "pp":
            return [
                [self.rank(RankCoordinate(dp, pp, pcp, tp)) for pp in range(self.pp)]
                for dp in range(self.dp)
                for pcp in range(self.pcp)
                for tp in range(self.tp)
            ]
        if axis == "dp":
            return [
                [self.rank(RankCoordinate(dp, pp, pcp, tp)) for dp in range(self.dp)]
                for pp in range(self.pp)
                for pcp in range(self.pcp)
                for tp in range(self.tp)
            ]
        if axis == "ep":
            return [
                [
                    self.rank(RankCoordinate(dp, pp, pcp, tp))
                    for dp in range(self.dp)
                    for pcp in range(self.pcp)
                    for tp in range(self.tp)
                ]
                for pp in range(self.pp)
            ]
        if axis == "dcp":
            traversal = [
                self.rank(RankCoordinate(dp, pp, pcp, tp))
                for dp in range(self.dp)
                for pp in range(self.pp)
                for tp in range(self.tp)
                for pcp in range(self.pcp)
            ]
            return [
                traversal[i : i + self.dcp]
                for i in range(0, len(traversal), self.dcp)
            ]
        raise ValueError(f"unknown parallel axis: {axis}")


def _validate_matrix(matrix: Matrix) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be non-empty")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    return len(matrix), width


def matmul(inputs: Matrix, weight: Matrix) -> Matrix:
    rows, input_size = _validate_matrix(inputs)
    weight_input_size, output_size = _validate_matrix(weight)
    if input_size != weight_input_size:
        raise ValueError("matrix dimensions do not align")
    return [
        [
            sum(inputs[row][k] * weight[k][column] for k in range(input_size))
            for column in range(output_size)
        ]
        for row in range(rows)
    ]


def split_columns(weight: Matrix, parts: int) -> list[Matrix]:
    _, output_size = _validate_matrix(weight)
    if output_size % parts:
        raise ValueError("output size must be divisible by TP")
    width = output_size // parts
    return [
        [row[part * width : (part + 1) * width] for row in weight]
        for part in range(parts)
    ]


def column_parallel_linear(inputs: Matrix, weight: Matrix, parts: int) -> list[Matrix]:
    """Return rank-local ``X @ A_i`` outputs before optional all-gather."""

    return [matmul(inputs, shard) for shard in split_columns(weight, parts)]


def all_gather_columns(local_outputs: Sequence[Matrix]) -> Matrix:
    if not local_outputs:
        raise ValueError("at least one rank output is required")
    rows = len(local_outputs[0])
    if any(len(output) != rows for output in local_outputs):
        raise ValueError("rank outputs must have equal row counts")
    return [
        [value for output in local_outputs for value in output[row]]
        for row in range(rows)
    ]


def row_parallel_linear(inputs: Matrix, weight: Matrix, parts: int) -> list[Matrix]:
    """Return partial ``X_i @ A_i`` outputs before the all-reduce sum."""

    _, input_size = _validate_matrix(inputs)
    weight_input_size, _ = _validate_matrix(weight)
    if input_size != weight_input_size or input_size % parts:
        raise ValueError("input size must align and be divisible by TP")
    width = input_size // parts
    outputs: list[Matrix] = []
    for part in range(parts):
        start = part * width
        end = start + width
        input_shard = [row[start:end] for row in inputs]
        weight_shard = weight[start:end]
        outputs.append(matmul(input_shard, weight_shard))
    return outputs


def all_reduce_sum(local_outputs: Sequence[Matrix]) -> Matrix:
    if not local_outputs:
        raise ValueError("at least one rank output is required")
    rows, columns = _validate_matrix(local_outputs[0])
    for output in local_outputs[1:]:
        if _validate_matrix(output) != (rows, columns):
            raise ValueError("all-reduce inputs must have identical shapes")
    return [
        [sum(output[row][column] for output in local_outputs) for column in range(columns)]
        for row in range(rows)
    ]


def pipeline_partitions(num_layers: int, pp_size: int) -> list[tuple[int, int]]:
    """Match vLLM's default ``get_pp_indices`` balancing rule."""

    if num_layers < 0 or pp_size < 1:
        raise ValueError("invalid layer or PP count")
    base = num_layers // pp_size
    partitions = [base] * pp_size
    remaining = num_layers % pp_size
    for i in range(2, remaining + 2):
        partitions[-i] += 1
    result: list[tuple[int, int]] = []
    start = 0
    for size in partitions:
        result.append((start, start + size))
        start += size
    return result


def pcp_dual_chunk_assignment(num_tokens: int, pcp_size: int) -> list[list[int]]:
    """Assign the two mirrored prefill chunks used by MRV2 PCPManager."""

    if num_tokens < 0 or pcp_size < 1:
        raise ValueError("invalid token or PCP count")
    num_chunks = 2 * pcp_size
    chunk_size = math.ceil(num_tokens / num_chunks) if num_tokens else 0
    result: list[list[int]] = []
    for rank in range(pcp_size):
        tokens: list[int] = []
        for chunk in (rank, num_chunks - 1 - rank):
            start = chunk * chunk_size
            tokens.extend(range(start, min(start + chunk_size, num_tokens)))
        result.append(tokens)
    return result


def dcp_token_owner(token_position: int, dcp_size: int, interleave: int = 1) -> int:
    if token_position < 0 or dcp_size < 1 or interleave < 1:
        raise ValueError("invalid DCP position or size")
    return (token_position // interleave) % dcp_size


def lse_weighted_combine(
    partial_outputs: Sequence[Sequence[Number]], lses: Sequence[Number]
) -> Vector:
    """Combine attention shard outputs with numerically stable LSE weights."""

    if not partial_outputs or len(partial_outputs) != len(lses):
        raise ValueError("one LSE is required per partial output")
    width = len(partial_outputs[0])
    if any(len(output) != width for output in partial_outputs):
        raise ValueError("partial outputs must have equal widths")
    finite_lses = [float(value) if math.isfinite(value) else -math.inf for value in lses]
    maximum = max(finite_lses)
    if maximum == -math.inf:
        return [0.0] * width
    raw_weights = [math.exp(value - maximum) for value in finite_lses]
    denominator = sum(raw_weights)
    weights = [value / denominator for value in raw_weights]
    return [
        sum(weights[rank] * float(partial_outputs[rank][i]) for rank in range(len(weights)))
        for i in range(width)
    ]


def expert_map(
    num_experts: int, ep_size: int, ep_rank: int, strategy: str = "linear"
) -> list[int]:
    """Map global expert IDs to local IDs, using -1 for non-local experts."""

    if num_experts < 0 or ep_size < 1 or not 0 <= ep_rank < ep_size:
        raise ValueError("invalid expert topology")
    owners: list[int]
    if strategy == "linear":
        base, remainder = divmod(num_experts, ep_size)
        start = ep_rank * base + min(ep_rank, remainder)
        count = base + (1 if ep_rank < remainder else 0)
        owners = list(range(start, start + count))
    elif strategy == "round_robin":
        owners = list(range(ep_rank, num_experts, ep_size))
    else:
        raise ValueError("unknown expert placement strategy")
    result = [-1] * num_experts
    for local_id, global_id in enumerate(owners):
        result[global_id] = local_id
    return result


@dataclass(frozen=True)
class RoutedToken:
    source_rank: int
    source_index: int
    expert_id: int
    expert_rank: int
    value: float
    weight: float


def dispatch_tokens(
    tokens_by_rank: Sequence[Sequence[float]],
    expert_ids_by_rank: Sequence[Sequence[int]],
    router_weights_by_rank: Sequence[Sequence[float]],
    num_experts: int,
    ep_size: int,
) -> list[list[RoutedToken]]:
    """Build the conceptual all-to-all buckets for top-1 routing."""

    if not (
        len(tokens_by_rank) == len(expert_ids_by_rank) == len(router_weights_by_rank)
    ):
        raise ValueError("routing inputs must have one entry per source rank")
    maps = [expert_map(num_experts, ep_size, rank) for rank in range(ep_size)]
    result: list[list[RoutedToken]] = [[] for _ in range(ep_size)]
    for source_rank, tokens in enumerate(tokens_by_rank):
        ids = expert_ids_by_rank[source_rank]
        weights = router_weights_by_rank[source_rank]
        if not (len(tokens) == len(ids) == len(weights)):
            raise ValueError("each source rank needs aligned token routing data")
        for source_index, (value, expert_id, weight) in enumerate(zip(tokens, ids, weights)):
            if not 0 <= expert_id < num_experts:
                raise ValueError("expert id is outside the global expert space")
            destination = next(rank for rank, mapping in enumerate(maps) if mapping[expert_id] >= 0)
            result[destination].append(
                RoutedToken(
                    source_rank,
                    source_index,
                    expert_id,
                    destination,
                    float(value),
                    float(weight),
                )
            )
    return result


def combine_expert_outputs(
    dispatched: Sequence[Sequence[RoutedToken]], expert_scales: Sequence[float]
) -> list[list[float]]:
    """Compute local experts and restore values to source-rank token order."""

    max_index: dict[int, int] = {}
    for bucket in dispatched:
        for item in bucket:
            max_index[item.source_rank] = max(max_index.get(item.source_rank, -1), item.source_index)
    outputs = [[0.0] * (max_index[rank] + 1) for rank in range(max(max_index, default=-1) + 1)]
    for bucket in dispatched:
        for item in bucket:
            outputs[item.source_rank][item.source_index] += (
                item.value * expert_scales[item.expert_id] * item.weight
            )
    return outputs


def choose_dp_engine(
    engine_stats: Sequence[tuple[int, int, float]],
    inflight: Sequence[int] | None = None,
    client_count: int = 1,
    start_index: int = 0,
) -> int:
    """Approximate DPLBAsyncMPClient's minimum-score routing rule."""

    if not engine_stats:
        raise ValueError("at least one engine is required")
    if inflight is None:
        inflight = [0] * len(engine_stats)
    if len(inflight) != len(engine_stats) or client_count < 1:
        raise ValueError("invalid inflight counters or client count")
    best_index = 0
    best_score = math.inf
    for offset in range(len(engine_stats)):
        index = (start_index + offset) % len(engine_stats)
        waiting, running, kv_usage = engine_stats[index]
        score = max(client_count * inflight[index], waiting + running)
        if waiting:
            score += waiting * 6.0 * max(0.0, kv_usage - 0.5)
        if score < best_score:
            best_score = score
            best_index = index
    return best_index


def collective_time_us(
    payload_bytes: int, latency_us: float, bandwidth_gbps: float, phases: int = 1
) -> float:
    """Simple alpha-beta teaching model, not a collective implementation."""

    if payload_bytes < 0 or latency_us < 0 or bandwidth_gbps <= 0 or phases < 1:
        raise ValueError("invalid communication model")
    transfer_us = payload_bytes * 8 / bandwidth_gbps / 1_000
    return phases * latency_us + transfer_us


def demo() -> None:
    topology = ParallelTopology(tp=2, pp=2, dp=2, pcp=1, dcp=1)
    print("workers per DP engine:", topology.worker_world_size)
    print("workers across DP:", topology.world_size_across_dp)
    for axis in ("tp", "pp", "dp", "ep"):
        print(f"{axis.upper()} groups:", topology.groups(axis))
    print("PP partitions for 10 layers:", pipeline_partitions(10, 3))
    print("PCP dual chunks:", pcp_dual_chunk_assignment(16, 4))


if __name__ == "__main__":
    demo()
