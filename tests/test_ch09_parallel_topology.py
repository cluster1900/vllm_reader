import math
import unittest

from examples.ch09_parallel_topology import (
    ParallelTopology,
    RankCoordinate,
    all_gather_columns,
    all_reduce_sum,
    choose_dp_engine,
    collective_time_us,
    column_parallel_linear,
    combine_expert_outputs,
    dcp_token_owner,
    dispatch_tokens,
    expert_map,
    lse_weighted_combine,
    matmul,
    pcp_dual_chunk_assignment,
    pipeline_partitions,
    row_parallel_linear,
)


class TopologyTest(unittest.TestCase):
    def test_world_sizes(self):
        topology = ParallelTopology(tp=4, pp=3, dp=4, dcp=2)
        self.assertEqual(topology.worker_world_size, 12)
        self.assertEqual(topology.world_size_across_dp, 48)

    def test_rank_round_trip(self):
        topology = ParallelTopology(tp=2, pcp=2, dcp=2)
        coordinate = RankCoordinate(dp=0, pp=0, pcp=1, tp=0)
        self.assertEqual(topology.coordinate(topology.rank(coordinate)), coordinate)

    def test_tp_and_pp_groups_match_rank_layout(self):
        topology = ParallelTopology(tp=2, pp=2)
        self.assertEqual(topology.groups("tp"), [[0, 1], [2, 3]])
        self.assertEqual(topology.groups("pp"), [[0, 2], [1, 3]])

    def test_pcp_and_dcp_groups(self):
        topology = ParallelTopology(tp=2, pcp=2, dcp=2)
        self.assertEqual(topology.groups("pcp"), [[0, 2], [1, 3]])
        self.assertEqual(topology.groups("dcp"), [[0, 2], [1, 3]])

    def test_ep_flattens_dp_pcp_tp_per_pipeline_stage(self):
        topology = ParallelTopology(tp=2, pp=2, dp=2)
        self.assertEqual(topology.groups("ep"), [[0, 1, 4, 5], [2, 3, 6, 7]])

    def test_invalid_dcp_is_rejected(self):
        with self.assertRaises(ValueError):
            ParallelTopology(tp=3, dcp=2)

    def test_pcp_with_dp_is_rejected_by_pinned_config(self):
        with self.assertRaisesRegex(ValueError, "PCP does not support"):
            ParallelTopology(tp=2, dp=2, pcp=2, dcp=2)


class TensorParallelTest(unittest.TestCase):
    def setUp(self):
        self.inputs = [[1.0, 2.0, 3.0, 4.0]]
        self.weight = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]

    def test_column_parallel_gather_matches_dense(self):
        local = column_parallel_linear(self.inputs, self.weight, 2)
        self.assertEqual(all_gather_columns(local), matmul(self.inputs, self.weight))

    def test_row_parallel_reduce_matches_dense(self):
        local = row_parallel_linear(self.inputs, self.weight, 2)
        self.assertEqual(all_reduce_sum(local), matmul(self.inputs, self.weight))


class PipelineAndContextTest(unittest.TestCase):
    def test_pp_remainder_avoids_last_stage(self):
        self.assertEqual(pipeline_partitions(10, 3), [(0, 3), (3, 7), (7, 10)])

    def test_pcp_dual_chunks_cover_prefill_once(self):
        assignments = pcp_dual_chunk_assignment(16, 4)
        flattened = sorted(token for rank in assignments for token in rank)
        self.assertEqual(flattened, list(range(16)))
        self.assertIn(0, assignments[0])
        self.assertIn(15, assignments[0])

    def test_dcp_interleaved_owner(self):
        owners = [dcp_token_owner(i, 2, interleave=2) for i in range(8)]
        self.assertEqual(owners, [0, 0, 1, 1, 0, 0, 1, 1])

    def test_lse_combine_uses_partition_mass_not_plain_average(self):
        result = lse_weighted_combine([[10.0], [20.0]], [0.0, math.log(3.0)])
        self.assertAlmostEqual(result[0], 17.5)

    def test_empty_lse_shards_produce_zero(self):
        self.assertEqual(
            lse_weighted_combine([[10.0], [20.0]], [-math.inf, -math.inf]),
            [0.0],
        )


class ExpertParallelTest(unittest.TestCase):
    def test_linear_expert_map(self):
        self.assertEqual(expert_map(5, 2, 0), [0, 1, 2, -1, -1])
        self.assertEqual(expert_map(5, 2, 1), [-1, -1, -1, 0, 1])

    def test_round_robin_expert_map(self):
        self.assertEqual(expert_map(6, 2, 1, "round_robin"), [-1, 0, -1, 1, -1, 2])

    def test_dispatch_and_combine_restore_source_order(self):
        dispatched = dispatch_tokens(
            [[2.0, 3.0], [5.0]],
            [[0, 3], [1]],
            [[1.0, 0.5], [0.25]],
            num_experts=4,
            ep_size=2,
        )
        self.assertEqual([item.expert_id for item in dispatched[0]], [0, 1])
        self.assertEqual([item.expert_id for item in dispatched[1]], [3])
        outputs = combine_expert_outputs(dispatched, [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(outputs, [[20.0, 60.0], [25.0]])


class DataParallelAndCostTest(unittest.TestCase):
    def test_dp_router_selects_lower_load(self):
        stats = [(2, 1, 0.3), (0, 0, 0.9), (1, 0, 0.2)]
        self.assertEqual(choose_dp_engine(stats), 1)

    def test_kv_pressure_penalizes_waiting_queue(self):
        stats = [(1, 1, 1.0), (0, 3, 0.2)]
        self.assertEqual(choose_dp_engine(stats), 1)

    def test_collective_model_has_latency_and_payload_terms(self):
        small = collective_time_us(1_000, latency_us=5, bandwidth_gbps=100)
        large = collective_time_us(1_000_000, latency_us=5, bandwidth_gbps=100)
        self.assertGreater(large, small)
        self.assertEqual(collective_time_us(0, 5, 100, phases=2), 10)


if __name__ == "__main__":
    unittest.main()
