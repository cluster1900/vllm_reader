#!/usr/bin/env python3
"""Dependency-free teaching model for vLLM V1 KV-cache block management.

The model demonstrates scheduler-side metadata only. It does not allocate GPU
tensors. It keeps the important contracts used in Chapter 5: a reserved null
block, request block tables, reference-counted prefix sharing, a free queue
that doubles as an eviction queue, full-block cache keys, optional partial-tail
keys, copy-on-write, and allocation failure without partial mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil


CacheKey = tuple[int, ...]


def kv_bytes_per_token(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype_bytes: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Estimate decoder-only KV bytes per token on one TP rank.

    This teaching formula assumes KV heads divide evenly across TP ranks.
    Some real models replicate KV heads when there are fewer KV heads than TP
    ranks, and quantized layouts can carry scale/zero-point metadata.
    """

    values = (num_layers, num_kv_heads, head_size, dtype_bytes, tensor_parallel_size)
    if any(value < 1 for value in values):
        raise ValueError("all dimensions must be positive")
    if num_kv_heads % tensor_parallel_size:
        raise ValueError("teaching formula requires KV heads divisible by TP")
    local_kv_heads = num_kv_heads // tensor_parallel_size
    return 2 * num_layers * local_kv_heads * head_size * dtype_bytes


def block_bytes(bytes_per_token: int, block_size: int) -> int:
    if bytes_per_token < 1 or block_size < 1:
        raise ValueError("bytes_per_token and block_size must be positive")
    return bytes_per_token * block_size


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    cache_keys: set[CacheKey] = field(default_factory=set)
    is_null: bool = False

    @property
    def cached(self) -> bool:
        return bool(self.cache_keys)


@dataclass
class RequestBlocks:
    request_id: str
    tokens: list[int]
    blocks: list[Block]
    computed_tokens: int = 0


@dataclass(frozen=True)
class CowCopy:
    src_block_id: int
    dst_block_id: int


@dataclass(frozen=True)
class AllocationResult:
    computed_tokens: int
    block_ids: tuple[int, ...]
    new_block_ids: tuple[int, ...]
    cow_copies: tuple[CowCopy, ...]


class TeachingBlockPool:
    """A small block pool whose free list also carries eviction priority."""

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        hash_block_size: int | None = None,
        enable_caching: bool = True,
    ) -> None:
        if num_blocks < 2:
            raise ValueError("need at least one null block and one usable block")
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if hash_block_size is None:
            hash_block_size = block_size
        if hash_block_size < 1 or block_size % hash_block_size:
            raise ValueError("hash_block_size must divide block_size")

        self.block_size = block_size
        self.hash_block_size = hash_block_size
        self.enable_caching = enable_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.null_block = self.blocks[0]
        self.null_block.is_null = True
        self.free_queue: list[Block] = list(self.blocks[1:])
        self.cache_map: dict[CacheKey, list[Block]] = {}
        self.requests: dict[str, RequestBlocks] = {}
        self.cow_history: list[CowCopy] = []
        self.eviction_history: list[int] = []

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_queue)

    @property
    def usage(self) -> float:
        usable = len(self.blocks) - 1
        return (usable - self.num_free_blocks) / usable

    def _insert_key(self, block: Block, key: CacheKey) -> None:
        if not self.enable_caching or block.is_null or key in block.cache_keys:
            return
        block.cache_keys.add(key)
        self.cache_map.setdefault(key, []).append(block)

    def _evict_metadata(self, block: Block) -> None:
        if not block.cache_keys:
            return
        for key in tuple(block.cache_keys):
            candidates = self.cache_map[key]
            candidates.remove(block)
            if not candidates:
                del self.cache_map[key]
        block.cache_keys.clear()
        self.eviction_history.append(block.block_id)

    def evict(self, block_ids: set[int]) -> None:
        for block_id in block_ids:
            if block_id <= 0 or block_id >= len(self.blocks):
                raise ValueError(f"invalid block id: {block_id}")
            self._evict_metadata(self.blocks[block_id])

    def _allocate_new(self, count: int) -> list[Block]:
        if count > self.num_free_blocks:
            raise ValueError(f"need {count} blocks, only {self.num_free_blocks} free")
        allocated = self.free_queue[:count]
        del self.free_queue[:count]
        for block in allocated:
            self._evict_metadata(block)
            if block.ref_count != 0:
                raise AssertionError("free queue contains referenced block")
            block.ref_count = 1
        return allocated

    def _touch(self, blocks: list[Block]) -> None:
        for block in blocks:
            if block.ref_count == 0:
                self.free_queue.remove(block)
            block.ref_count += 1

    def _free_one(self, block: Block) -> None:
        if block.is_null:
            return
        if block.ref_count < 1:
            raise AssertionError("double free")
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.cached and self.enable_caching:
                self.free_queue.append(block)
            else:
                self.free_queue.insert(0, block)

    def _release_blocks(self, blocks: list[Block]) -> None:
        # Tail blocks are passed first, matching the request-free call site.
        uncached: list[Block] = []
        cached: list[Block] = []
        for block in blocks:
            if block.is_null:
                continue
            if block.ref_count < 1:
                raise AssertionError("double free")
            block.ref_count -= 1
            if block.ref_count == 0:
                (cached if block.cached and self.enable_caching else uncached).append(
                    block
                )
        self.free_queue = uncached + self.free_queue + cached

    def _cached_candidate(self, key: CacheKey) -> Block | None:
        candidates = self.cache_map.get(key)
        return candidates[0] if candidates else None

    def lookup_prefix(
        self,
        tokens: list[int],
        *,
        max_cache_hit_length: int | None = None,
        allow_partial: bool = False,
    ) -> tuple[list[Block], int, bool]:
        """Return a contiguous cached prefix and whether its tail is partial."""

        if not self.enable_caching or not tokens:
            return [], 0, False
        if max_cache_hit_length is None:
            max_cache_hit_length = len(tokens) - 1
        max_cache_hit_length = max(0, min(max_cache_hit_length, len(tokens)))

        hit_blocks: list[Block] = []
        hit_tokens = 0
        full_limit = (max_cache_hit_length // self.block_size) * self.block_size
        for boundary in range(self.block_size, full_limit + 1, self.block_size):
            block = self._cached_candidate(tuple(tokens[:boundary]))
            if block is None:
                break
            hit_blocks.append(block)
            hit_tokens = boundary

        partial = False
        if allow_partial:
            block_end = min(
                ((hit_tokens // self.block_size) + 1) * self.block_size,
                max_cache_hit_length,
            )
            for boundary in range(
                hit_tokens + self.hash_block_size,
                block_end + 1,
                self.hash_block_size,
            ):
                if boundary % self.block_size == 0:
                    continue
                block = self._cached_candidate(tuple(tokens[:boundary]))
                if block is not None:
                    if len(hit_blocks) == boundary // self.block_size:
                        hit_blocks.append(block)
                    else:
                        hit_blocks[-1] = block
                    hit_tokens = boundary
                    partial = True
        return hit_blocks, hit_tokens, partial

    def start_request(
        self,
        request_id: str,
        tokens: list[int],
        *,
        allow_partial: bool = False,
    ) -> AllocationResult | None:
        if request_id in self.requests:
            raise ValueError(f"duplicate request id: {request_id}")
        if not tokens:
            raise ValueError("request must contain at least one token")

        hit_blocks, hit_tokens, partial = self.lookup_prefix(
            tokens, allow_partial=allow_partial
        )
        total_blocks = ceil(len(tokens) / self.block_size)
        normal_new = max(total_blocks - len(hit_blocks), 0)
        cow_needed = int(partial and hit_tokens < len(tokens))

        evictable_hits = sum(block.ref_count == 0 for block in hit_blocks)
        if normal_new + cow_needed > self.num_free_blocks - evictable_hits:
            return None

        self._touch(hit_blocks)
        new_blocks: list[Block] = []
        cow_copies: list[CowCopy] = []
        request_blocks = list(hit_blocks)

        if cow_needed:
            source = request_blocks[-1]
            target = self._allocate_new(1)[0]
            request_blocks[-1] = target
            copy = CowCopy(source.block_id, target.block_id)
            cow_copies.append(copy)
            self.cow_history.append(copy)
            self._free_one(source)
            new_blocks.append(target)

        if normal_new:
            allocated = self._allocate_new(normal_new)
            request_blocks.extend(allocated)
            new_blocks.extend(allocated)

        self.requests[request_id] = RequestBlocks(
            request_id=request_id,
            tokens=list(tokens),
            blocks=request_blocks,
            computed_tokens=hit_tokens,
        )
        return AllocationResult(
            computed_tokens=hit_tokens,
            block_ids=tuple(block.block_id for block in request_blocks),
            new_block_ids=tuple(block.block_id for block in new_blocks),
            cow_copies=tuple(cow_copies),
        )

    def extend_request(self, request_id: str, new_tokens: list[int]) -> AllocationResult | None:
        request = self.requests[request_id]
        if not new_tokens:
            raise ValueError("new_tokens cannot be empty")
        old_len = len(request.tokens)
        target_tokens = request.tokens + list(new_tokens)
        old_block_count = ceil(old_len / self.block_size)
        new_block_count = ceil(len(target_tokens) / self.block_size)

        cow_needed = 0
        if old_len % self.block_size and request.blocks:
            tail = request.blocks[-1]
            cow_needed = int(tail.ref_count > 1 or tail.cached)
        normal_new = new_block_count - old_block_count
        if cow_needed + normal_new > self.num_free_blocks:
            return None

        new_blocks: list[Block] = []
        copies: list[CowCopy] = []
        if cow_needed:
            source = request.blocks[-1]
            target = self._allocate_new(1)[0]
            request.blocks[-1] = target
            copy = CowCopy(source.block_id, target.block_id)
            copies.append(copy)
            self.cow_history.append(copy)
            self._free_one(source)
            new_blocks.append(target)
        if normal_new:
            allocated = self._allocate_new(normal_new)
            request.blocks.extend(allocated)
            new_blocks.extend(allocated)
        request.tokens = target_tokens
        return AllocationResult(
            computed_tokens=request.computed_tokens,
            block_ids=tuple(block.block_id for block in request.blocks),
            new_block_ids=tuple(block.block_id for block in new_blocks),
            cow_copies=tuple(copies),
        )

    def mark_computed(
        self,
        request_id: str,
        num_computed_tokens: int,
        *,
        register_partial: bool = False,
    ) -> None:
        request = self.requests[request_id]
        if not 0 <= num_computed_tokens <= len(request.tokens):
            raise ValueError("computed length outside request token range")
        request.computed_tokens = num_computed_tokens
        if not self.enable_caching:
            return

        num_full = num_computed_tokens // self.block_size
        for index in range(num_full):
            boundary = (index + 1) * self.block_size
            self._insert_key(request.blocks[index], tuple(request.tokens[:boundary]))

        remainder = num_computed_tokens % self.block_size
        if register_partial and remainder:
            if num_computed_tokens % self.hash_block_size:
                raise ValueError("partial boundary must align to hash_block_size")
            block_index = num_computed_tokens // self.block_size
            self._insert_key(
                request.blocks[block_index],
                tuple(request.tokens[:num_computed_tokens]),
            )

    def finish_request(self, request_id: str) -> None:
        request = self.requests.pop(request_id)
        self._release_blocks(list(reversed(request.blocks)))

    def reset_prefix_cache(self) -> bool:
        if self.requests:
            return False
        for block in self.blocks:
            block.cache_keys.clear()
        self.cache_map.clear()
        return True

    def block_table(self, request_id: str) -> tuple[int, ...]:
        return tuple(block.block_id for block in self.requests[request_id].blocks)

    def free_queue_ids(self) -> tuple[int, ...]:
        return tuple(block.block_id for block in self.free_queue)


def demo() -> None:
    pool = TeachingBlockPool(num_blocks=7, block_size=4, hash_block_size=2)
    first = pool.start_request("A", [1, 2, 3, 4, 5, 6])
    assert first is not None
    pool.mark_computed("A", 6, register_partial=True)
    pool.finish_request("A")
    print("after A:", pool.free_queue_ids())

    second = pool.start_request("B", [1, 2, 3, 4, 5, 6, 9], allow_partial=True)
    assert second is not None
    print("B hit:", second.computed_tokens, "blocks:", second.block_ids)
    print("B CoW:", second.cow_copies)
    print("free queue:", pool.free_queue_ids())


if __name__ == "__main__":
    demo()
