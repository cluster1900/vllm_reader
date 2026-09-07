import unittest

from examples.ch05_kv_cache_blocks import (
    TeachingBlockPool,
    block_bytes,
    kv_bytes_per_token,
)


class KVCapacityTest(unittest.TestCase):
    def test_llama3_8b_style_capacity(self):
        per_token = kv_bytes_per_token(
            num_layers=32,
            num_kv_heads=8,
            head_size=128,
            dtype_bytes=2,
        )
        self.assertEqual(per_token, 128 * 1024)
        self.assertEqual(block_bytes(per_token, 16), 2 * 1024 * 1024)
        self.assertEqual(per_token * 8192, 1024**3)

    def test_tp_splits_kv_heads_in_teaching_formula(self):
        single = kv_bytes_per_token(
            num_layers=32,
            num_kv_heads=8,
            head_size=128,
            dtype_bytes=2,
        )
        tp2 = kv_bytes_per_token(
            num_layers=32,
            num_kv_heads=8,
            head_size=128,
            dtype_bytes=2,
            tensor_parallel_size=2,
        )
        self.assertEqual(tp2, single // 2)


class BlockLifecycleTest(unittest.TestCase):
    def test_full_prefix_is_shared_by_reference_count(self):
        pool = TeachingBlockPool(num_blocks=6, block_size=4)
        a = pool.start_request("A", list(range(9)))
        self.assertIsNotNone(a)
        pool.mark_computed("A", 8)

        b = pool.start_request("B", list(range(8)) + [99])
        self.assertIsNotNone(b)
        self.assertEqual(b.computed_tokens, 8)
        self.assertEqual(b.block_ids[:2], a.block_ids[:2])
        self.assertEqual(pool.blocks[a.block_ids[0]].ref_count, 2)

        pool.finish_request("A")
        self.assertEqual(pool.blocks[a.block_ids[0]].ref_count, 1)
        self.assertNotIn(a.block_ids[0], pool.free_queue_ids())

    def test_finished_cached_blocks_remain_evictable(self):
        pool = TeachingBlockPool(num_blocks=4, block_size=4)
        result = pool.start_request("A", list(range(8)))
        self.assertIsNotNone(result)
        pool.mark_computed("A", 8)
        pool.finish_request("A")

        self.assertEqual(pool.num_free_blocks, 3)
        self.assertIn(tuple(range(4)), pool.cache_map)
        self.assertEqual(pool.blocks[result.block_ids[0]].ref_count, 0)

        other = pool.start_request("X", [20] * 12)
        self.assertIsNotNone(other)
        self.assertTrue(pool.eviction_history)

    def test_uncached_tail_is_reused_before_cached_blocks(self):
        pool = TeachingBlockPool(num_blocks=4, block_size=4)
        result = pool.start_request("A", list(range(6)))
        self.assertIsNotNone(result)
        pool.mark_computed("A", 4)
        pool.finish_request("A")

        first_free = pool.free_queue_ids()[0]
        self.assertEqual(first_free, result.block_ids[1])
        other = pool.start_request("B", [9])
        self.assertEqual(other.new_block_ids, (result.block_ids[1],))

    def test_all_hit_prompt_recomputes_last_token_boundary(self):
        pool = TeachingBlockPool(num_blocks=5, block_size=4)
        first = pool.start_request("A", list(range(8)))
        pool.mark_computed("A", 8)
        pool.finish_request("A")

        replay = pool.start_request("B", list(range(8)))

        self.assertEqual(replay.computed_tokens, 4)
        self.assertNotEqual(replay.block_ids[1], first.block_ids[1])

    def test_allocation_failure_does_not_create_request(self):
        pool = TeachingBlockPool(num_blocks=3, block_size=4)
        self.assertIsNone(pool.start_request("too-large", list(range(12))))
        self.assertNotIn("too-large", pool.requests)
        self.assertEqual(pool.num_free_blocks, 2)

    def test_partial_hit_uses_copy_on_write(self):
        pool = TeachingBlockPool(
            num_blocks=6, block_size=4, hash_block_size=2
        )
        owner = pool.start_request("owner", [1, 2, 3, 4, 5, 6])
        pool.mark_computed("owner", 6, register_partial=True)
        partial_source = owner.block_ids[1]
        pool.finish_request("owner")

        hit = pool.start_request(
            "consumer", [1, 2, 3, 4, 5, 6, 7], allow_partial=True
        )

        self.assertEqual(hit.computed_tokens, 6)
        self.assertEqual(hit.cow_copies[0].src_block_id, partial_source)
        self.assertNotEqual(hit.block_ids[1], partial_source)
        self.assertIn(tuple([1, 2, 3, 4, 5, 6]), pool.cache_map)

    def test_owner_continuing_cached_partial_tail_also_cows(self):
        pool = TeachingBlockPool(
            num_blocks=6, block_size=4, hash_block_size=2
        )
        owner = pool.start_request("owner", [1, 2, 3, 4, 5, 6])
        pool.mark_computed("owner", 6, register_partial=True)

        extension = pool.extend_request("owner", [7])

        self.assertEqual(extension.cow_copies[0].src_block_id, owner.block_ids[1])
        self.assertNotEqual(extension.block_ids[1], owner.block_ids[1])

    def test_reset_cache_requires_no_active_requests(self):
        pool = TeachingBlockPool(num_blocks=4, block_size=4)
        pool.start_request("A", [1, 2, 3, 4])
        pool.mark_computed("A", 4)
        self.assertFalse(pool.reset_prefix_cache())
        pool.finish_request("A")
        self.assertTrue(pool.reset_prefix_cache())
        self.assertEqual(pool.cache_map, {})


if __name__ == "__main__":
    unittest.main()
