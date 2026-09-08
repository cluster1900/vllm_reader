import unittest

from examples.ch06_paged_addressing import (
    PAD_SLOT_ID,
    BackendCandidate,
    BackendRequirements,
    build_query_start_loc,
    build_slot_mapping,
    contiguous_attention,
    expand_manager_blocks,
    gather_sequence,
    make_cache,
    paged_attention,
    scatter_cache,
    select_backend,
    slot_for_position,
)


class AddressingTest(unittest.TestCase):
    def test_manager_blocks_expand_for_smaller_kernel_pages(self):
        self.assertEqual(expand_manager_blocks([0, 3], 32, 16), [0, 1, 6, 7])

    def test_kernel_block_must_divide_manager_block(self):
        with self.assertRaises(ValueError):
            expand_manager_blocks([1], 24, 16)

    def test_non_contiguous_block_table_maps_positions(self):
        table = [2, 5, 3]
        self.assertEqual([slot_for_position(table, p, 4) for p in range(10)],
                         [8, 9, 10, 11, 20, 21, 22, 23, 12, 13])

    def test_ragged_query_boundaries(self):
        self.assertEqual(build_query_start_loc([3, 1, 2]), [0, 3, 4, 6])

    def test_slot_mapping_is_flattened_and_padded(self):
        starts, slots = build_slot_mapping(
            [[2, 5], [3]], [[4, 5], [0]], 4, padded_tokens=5
        )
        self.assertEqual(starts, [0, 2, 3])
        self.assertEqual(slots, [20, 21, 12, PAD_SLOT_ID, PAD_SLOT_ID])

    def test_disabled_slot_mapping_is_all_padding(self):
        _, slots = build_slot_mapping([[5]], [[0, 1]], 4, enabled=False)
        self.assertEqual(slots, [PAD_SLOT_ID, PAD_SLOT_ID])


class CacheAndAttentionTest(unittest.TestCase):
    def test_scatter_then_gather_restores_logical_order(self):
        table = [2, 5, 3]
        positions = list(range(10))
        _, slots = build_slot_mapping([table], [positions], 4)
        values = [(float(i),) for i in positions]
        cache = make_cache(6, 4)
        scatter_cache(cache, values, slots)
        self.assertEqual(gather_sequence(cache, table, 10, 4), values)

    def test_padding_slot_has_no_side_effect(self):
        cache = make_cache(2, 4)
        scatter_cache(cache, [(1.0,), (2.0,)], [PAD_SLOT_ID, 3])
        self.assertEqual(cache[3], (2.0,))
        self.assertEqual(sum(value is not None for value in cache), 1)

    def test_paged_attention_matches_contiguous_reference(self):
        table = [2, 0]
        keys = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, -1.0),
                (0.5, 0.5), (-1.0, 2.0)]
        values = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0),
                  (9.0, 10.0), (11.0, 12.0)]
        _, slots = build_slot_mapping([table], [range(len(keys))], 4)
        key_cache = make_cache(3, 4)
        value_cache = make_cache(3, 4)
        scatter_cache(key_cache, keys, slots)
        scatter_cache(value_cache, values, slots)

        expected = contiguous_attention((0.25, 0.75), keys, values)
        actual = paged_attention(
            (0.25, 0.75), key_cache, value_cache, table, len(keys), 4
        )
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right)

    def test_two_requests_can_share_the_same_prefix_blocks(self):
        cache = make_cache(5, 4)
        shared = [(float(i),) for i in range(4)]
        scatter_cache(cache, shared, [4, 5, 6, 7])
        request_a = [1, 3]
        request_b = [1, 4]
        self.assertEqual(gather_sequence(cache, request_a, 4, 4), shared)
        self.assertEqual(gather_sequence(cache, request_b, 4, 4), shared)


class BackendSelectionTest(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            BackendCandidate(
                "FAST", 0, frozenset({"fp16", "bf16"}),
                frozenset({"auto", "fp8"}), 8, 256, 16, 80,
                supports_sliding_window=True, supports_non_causal=True,
            ),
            BackendCandidate(
                "FALLBACK", 1, frozenset({"fp16", "bf16"}),
                frozenset({"auto"}), 1, 512, 8, 70,
                supports_sliding_window=True,
            ),
        ]

    def test_best_valid_priority_wins(self):
        req = BackendRequirements("fp16", "auto", 128, 16, 90)
        selected, rejected = select_backend(self.candidates, req)
        self.assertEqual(selected.name, "FAST")
        self.assertEqual(rejected, {})

    def test_invalid_higher_priority_backend_falls_back_with_reason(self):
        req = BackendRequirements("fp16", "auto", 17, 16, 90)
        selected, rejected = select_backend(self.candidates, req)
        self.assertEqual(selected.name, "FALLBACK")
        self.assertIn("head_size not supported", rejected["FAST"])

    def test_explicit_feature_can_leave_no_valid_backend(self):
        req = BackendRequirements(
            "fp16", "auto", 17, 16, 90, non_causal=True
        )
        with self.assertRaisesRegex(ValueError, "no valid backend"):
            select_backend(self.candidates, req)


if __name__ == "__main__":
    unittest.main()
