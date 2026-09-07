#!/usr/bin/env python3
"""Dependency-free teaching model for paged KV addressing and attention.

This is not a performance model and does not reproduce a CUDA kernel. It keeps
the contracts that matter for reading vLLM V1: allocation blocks may expand to
smaller kernel blocks, a request block table maps logical blocks to physical
blocks, slot mappings drive KV writes, and the attention read path follows the
block table instead of requiring a contiguous per-request cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Iterable, Sequence


PAD_SLOT_ID = -1
Vector = tuple[float, ...]


def expand_manager_blocks(
    block_ids: Sequence[int],
    manager_block_size: int,
    kernel_block_size: int,
) -> list[int]:
    """Expand allocation-level IDs into the IDs consumed by a kernel."""

    if manager_block_size < 1 or kernel_block_size < 1:
        raise ValueError("block sizes must be positive")
    if manager_block_size % kernel_block_size:
        raise ValueError("kernel_block_size must divide manager_block_size")
    blocks_per_manager_block = manager_block_size // kernel_block_size
    return [
        block_id * blocks_per_manager_block + offset
        for block_id in block_ids
        for offset in range(blocks_per_manager_block)
    ]


def slot_for_position(
    block_table: Sequence[int], position: int, kernel_block_size: int
) -> int:
    """Translate one logical token position into a flat physical cache slot."""

    if position < 0:
        raise ValueError("position must be non-negative")
    if kernel_block_size < 1:
        raise ValueError("kernel_block_size must be positive")
    logical_block = position // kernel_block_size
    if logical_block >= len(block_table):
        raise IndexError("block table does not cover this position")
    block_offset = position % kernel_block_size
    physical_block = block_table[logical_block]
    return physical_block * kernel_block_size + block_offset


def build_query_start_loc(query_lengths: Sequence[int]) -> list[int]:
    """Build cumulative ragged-query boundaries such as [0, 3, 4, 6]."""

    starts = [0]
    for length in query_lengths:
        if length < 0:
            raise ValueError("query lengths must be non-negative")
        starts.append(starts[-1] + length)
    return starts


def build_slot_mapping(
    block_tables: Sequence[Sequence[int]],
    positions_by_request: Sequence[Sequence[int]],
    kernel_block_size: int,
    *,
    padded_tokens: int | None = None,
    enabled: bool = True,
) -> tuple[list[int], list[int]]:
    """Flatten request positions and compute the corresponding write slots."""

    if len(block_tables) != len(positions_by_request):
        raise ValueError("need one block table per request")
    query_start_loc = build_query_start_loc(
        [len(positions) for positions in positions_by_request]
    )
    mappings = []
    for block_table, positions in zip(block_tables, positions_by_request):
        for position in positions:
            mappings.append(
                slot_for_position(block_table, position, kernel_block_size)
                if enabled
                else PAD_SLOT_ID
            )
    if padded_tokens is None:
        padded_tokens = len(mappings)
    if padded_tokens < len(mappings):
        raise ValueError("padded_tokens cannot be smaller than actual tokens")
    mappings.extend([PAD_SLOT_ID] * (padded_tokens - len(mappings)))
    return query_start_loc, mappings


def make_cache(num_blocks: int, block_size: int) -> list[Vector | None]:
    if num_blocks < 1 or block_size < 1:
        raise ValueError("cache dimensions must be positive")
    return [None] * (num_blocks * block_size)


def scatter_cache(
    cache: list[Vector | None], values: Sequence[Vector], slot_mapping: Sequence[int]
) -> None:
    """Write token values through slot_mapping; PAD slots have no side effect."""

    if len(values) > len(slot_mapping):
        raise ValueError("slot mapping is shorter than values")
    for value, slot in zip(values, slot_mapping):
        if slot == PAD_SLOT_ID:
            continue
        if not 0 <= slot < len(cache):
            raise IndexError(f"slot {slot} is outside the cache")
        cache[slot] = tuple(float(x) for x in value)


def gather_sequence(
    cache: Sequence[Vector | None],
    block_table: Sequence[int],
    seq_len: int,
    block_size: int,
) -> list[Vector]:
    """Read a logical sequence from non-contiguous physical cache blocks."""

    result = []
    for position in range(seq_len):
        slot = slot_for_position(block_table, position, block_size)
        value = cache[slot]
        if value is None:
            raise ValueError(f"logical position {position} points to an empty slot")
        result.append(value)
    return result


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    return sum(a * b for a, b in zip(left, right))


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("softmax requires at least one value")
    maximum = max(values)
    numerators = [exp(value - maximum) for value in values]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def contiguous_attention(
    query: Vector,
    keys: Sequence[Vector],
    values: Sequence[Vector],
    *,
    scale: float | None = None,
) -> Vector:
    """Small single-head attention reference over contiguous logical K/V."""

    if len(keys) != len(values) or not keys:
        raise ValueError("keys and values must have the same non-zero length")
    if scale is None:
        scale = 1.0 / sqrt(len(query))
    weights = _softmax([_dot(query, key) * scale for key in keys])
    value_size = len(values[0])
    if any(len(value) != value_size for value in values):
        raise ValueError("value dimensions differ")
    return tuple(
        sum(weight * value[index] for weight, value in zip(weights, values))
        for index in range(value_size)
    )


def paged_attention(
    query: Vector,
    key_cache: Sequence[Vector | None],
    value_cache: Sequence[Vector | None],
    block_table: Sequence[int],
    seq_len: int,
    block_size: int,
    *,
    scale: float | None = None,
) -> Vector:
    """Reference attention whose K/V are gathered through a block table."""

    keys = gather_sequence(key_cache, block_table, seq_len, block_size)
    values = gather_sequence(value_cache, block_table, seq_len, block_size)
    return contiguous_attention(query, keys, values, scale=scale)


@dataclass(frozen=True)
class BackendRequirements:
    dtype: str
    kv_cache_dtype: str
    head_size: int
    block_size: int
    compute_capability: int
    sliding_window: bool = False
    non_causal: bool = False


@dataclass(frozen=True)
class BackendCandidate:
    name: str
    priority: int
    dtypes: frozenset[str]
    kv_cache_dtypes: frozenset[str]
    head_size_multiple: int
    max_head_size: int
    kernel_block_multiple: int
    min_compute_capability: int
    supports_sliding_window: bool = False
    supports_non_causal: bool = False

    def rejection_reasons(self, requirements: BackendRequirements) -> list[str]:
        reasons = []
        if requirements.dtype not in self.dtypes:
            reasons.append("dtype not supported")
        if requirements.kv_cache_dtype not in self.kv_cache_dtypes:
            reasons.append("kv_cache_dtype not supported")
        if (
            requirements.head_size % self.head_size_multiple
            or requirements.head_size > self.max_head_size
        ):
            reasons.append("head_size not supported")
        if requirements.block_size % self.kernel_block_multiple:
            reasons.append("block_size not supported")
        if requirements.compute_capability < self.min_compute_capability:
            reasons.append("compute capability not supported")
        if requirements.sliding_window and not self.supports_sliding_window:
            reasons.append("sliding window not supported")
        if requirements.non_causal and not self.supports_non_causal:
            reasons.append("non-causal attention not supported")
        return reasons


def select_backend(
    candidates: Iterable[BackendCandidate], requirements: BackendRequirements
) -> tuple[BackendCandidate, dict[str, list[str]]]:
    """Validate all candidates, then choose the valid one with best priority."""

    valid = []
    rejected = {}
    for candidate in candidates:
        reasons = candidate.rejection_reasons(requirements)
        if reasons:
            rejected[candidate.name] = reasons
        else:
            valid.append(candidate)
    if not valid:
        details = "; ".join(f"{name}: {reasons}" for name, reasons in rejected.items())
        raise ValueError(f"no valid backend: {details}")
    return min(valid, key=lambda candidate: candidate.priority), rejected


def demo() -> None:
    block_size = 4
    block_table = [2, 0, 3]
    positions = list(range(10))
    _, slots = build_slot_mapping([block_table], [positions], block_size)
    key_cache = make_cache(4, block_size)
    value_cache = make_cache(4, block_size)
    keys = [(float(i), 1.0) for i in positions]
    values = [(float(i), float(i * 10)) for i in positions]
    scatter_cache(key_cache, keys, slots)
    scatter_cache(value_cache, values, slots)

    print("block table:", block_table)
    print("positions:  ", positions)
    print("slot mapping:", slots)
    print("gathered K: ", gather_sequence(key_cache, block_table, 10, block_size))
    print(
        "attention: ",
        paged_attention(
            (0.1, 0.2), key_cache, value_cache, block_table, 10, block_size
        ),
    )


if __name__ == "__main__":
    demo()
