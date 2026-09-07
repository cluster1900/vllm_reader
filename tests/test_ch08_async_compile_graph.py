import unittest

from examples.ch08_async_compile_graph import (
    AsyncRequestState,
    AttentionGraphSupport,
    CapturedGraph,
    CompilationMode,
    GraphDispatcher,
    GraphMode,
    StageTimes,
    StaticBuffer,
    compile_cache_key,
    makespan,
    overlapped_timeline,
    pad_to_capture_size,
    resolve_graph_mode,
    synchronous_timeline,
)


class TimelineTest(unittest.TestCase):
    def test_overlap_reduces_repeated_step_makespan(self):
        times = StageTimes(cpu_prepare_ms=2, gpu_execute_ms=5)
        sync = makespan(synchronous_timeline(4, times))
        overlap = makespan(overlapped_timeline(4, times))
        self.assertEqual(sync, 28)
        self.assertEqual(overlap, 22)

    def test_single_step_has_no_pipeline_gain(self):
        times = StageTimes(cpu_prepare_ms=3, gpu_execute_ms=7)
        self.assertEqual(
            makespan(synchronous_timeline(1, times)),
            makespan(overlapped_timeline(1, times)),
        )


class PlaceholderTest(unittest.TestCase):
    def test_schedule_moves_optimistic_not_confirmed_boundary(self):
        state = AsyncRequestState(num_computed_tokens=10)
        state.schedule_decode(sampled_tokens=1, spec_tokens=3)
        self.assertEqual(state.num_computed_tokens, 14)
        self.assertEqual(state.num_output_placeholders, 4)
        self.assertEqual(state.confirmed_tokens, 10)

    def test_normal_output_decrements_placeholders(self):
        state = AsyncRequestState(14, 4)
        state.apply_output([101, 102])
        self.assertEqual(state.num_output_placeholders, 2)
        self.assertEqual(state.confirmed_tokens, 12)

    def test_stale_output_does_not_underflow_after_preemption(self):
        state = AsyncRequestState(14, 4)
        state.preempt(10)
        state.apply_output([101], stale=True)
        self.assertEqual(state.num_output_placeholders, 0)
        self.assertEqual(state.confirmed_tokens, 10)


class ModeTest(unittest.TestCase):
    def test_compilation_and_graph_modes_are_independent_enums(self):
        self.assertEqual(CompilationMode.VLLM_COMPILE.value, 3)
        self.assertNotEqual(CompilationMode.VLLM_COMPILE.name, GraphMode.FULL.name)

    def test_composite_mode_resolves_to_runtime_modes(self):
        mode = GraphMode.FULL_AND_PIECEWISE
        self.assertEqual(mode.decode_mode(), GraphMode.FULL)
        self.assertEqual(mode.mixed_mode(), GraphMode.PIECEWISE)
        self.assertEqual(
            mode.runtime_modes(), frozenset((GraphMode.FULL, GraphMode.PIECEWISE))
        )

    def test_backend_support_can_downgrade_full(self):
        self.assertEqual(
            resolve_graph_mode(
                GraphMode.FULL,
                AttentionGraphSupport.NEVER,
                piecewise_available=True,
            ),
            GraphMode.PIECEWISE,
        )
        self.assertEqual(
            resolve_graph_mode(
                GraphMode.FULL,
                AttentionGraphSupport.NEVER,
                piecewise_available=False,
            ),
            GraphMode.NONE,
        )

    def test_mixed_full_downgrades_but_keeps_decode_full(self):
        self.assertEqual(
            resolve_graph_mode(
                GraphMode.FULL,
                AttentionGraphSupport.UNIFORM_SINGLE_TOKEN_DECODE,
                piecewise_available=True,
            ),
            GraphMode.FULL_AND_PIECEWISE,
        )


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.dispatcher = GraphDispatcher(
            GraphMode.FULL_AND_PIECEWISE,
            [1, 2, 4, 8],
            max_num_reqs=8,
        )

    def test_padding_uses_smallest_fitting_capture_size(self):
        self.assertEqual(pad_to_capture_size(3, [1, 2, 4, 8]), 4)
        self.assertIsNone(pad_to_capture_size(9, [1, 2, 4, 8]))

    def test_uniform_decode_prefers_full(self):
        mode, desc = self.dispatcher.dispatch(3, uniform_decode=True)
        self.assertEqual(mode, GraphMode.FULL)
        self.assertEqual(desc.num_tokens, 4)
        self.assertTrue(desc.uniform)

    def test_mixed_batch_uses_piecewise(self):
        mode, desc = self.dispatcher.dispatch(3, uniform_decode=False)
        self.assertEqual(mode, GraphMode.PIECEWISE)
        self.assertIsNone(desc.num_reqs)

    def test_disabling_full_falls_back_to_none_for_decode_only_route(self):
        mode, _ = self.dispatcher.dispatch(
            2, uniform_decode=True, allow_full=False
        )
        self.assertEqual(mode, GraphMode.NONE)

    def test_size_beyond_capture_range_falls_back(self):
        mode, desc = self.dispatcher.dispatch(9, uniform_decode=False)
        self.assertEqual(mode, GraphMode.NONE)
        self.assertEqual(desc.num_tokens, 9)


class StaticBufferTest(unittest.TestCase):
    def test_runtime_data_can_change_inside_one_stable_buffer(self):
        buffer = StaticBuffer(4)
        graph = CapturedGraph.capture(buffer)
        n = buffer.stage([1, 2])
        self.assertEqual(graph.replay(buffer, n), 3)
        n = buffer.stage([7, 8, 9])
        self.assertEqual(graph.replay(buffer, n), 24)

    def test_replacing_buffer_breaks_address_contract(self):
        original = StaticBuffer(4)
        graph = CapturedGraph.capture(original)
        replacement = StaticBuffer(4)
        replacement.stage([1])
        with self.assertRaisesRegex(ValueError, "captured input address"):
            graph.replay(replacement, 1)


class CompileCacheTest(unittest.TestCase):
    def test_cache_key_is_order_independent_but_source_sensitive(self):
        first = compile_cache_key(
            {"mode": 3, "dtype": "bf16"}, {"model.py": "hash-a"}
        )
        reordered = compile_cache_key(
            {"dtype": "bf16", "mode": 3}, {"model.py": "hash-a"}
        )
        changed = compile_cache_key(
            {"dtype": "bf16", "mode": 3}, {"model.py": "hash-b"}
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
