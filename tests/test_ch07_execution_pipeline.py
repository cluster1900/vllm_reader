import unittest

from examples.ch07_execution_pipeline import (
    IntermediateTensors,
    MultiprocExecutorModel,
    NewRequest,
    PersistentModelRunner,
    SamplingParams,
    SchedulerStep,
    choose_loader,
    prepare_batch,
    process_logits,
    resolve_model_class,
    sample_token,
    select_executor_backend,
)


class SelectionTest(unittest.TestCase):
    def test_executor_backend_selection(self):
        self.assertEqual(select_executor_backend("uni"), "UniProcExecutor")
        self.assertEqual(select_executor_backend("mp"), "MultiprocExecutor")
        self.assertEqual(
            select_executor_backend("ray", use_ray_v2=True), "RayExecutorV2"
        )

    def test_registry_uses_first_supported_architecture(self):
        registry = {"LlamaForCausalLM": "llama:LlamaForCausalLM"}
        self.assertEqual(
            resolve_model_class(["Unknown", "LlamaForCausalLM"], registry),
            ("LlamaForCausalLM", "llama:LlamaForCausalLM"),
        )

    def test_loader_selection(self):
        self.assertEqual(choose_loader("auto"), "DefaultModelLoader")
        self.assertEqual(choose_loader("dummy"), "DummyModelLoader")
        with self.assertRaises(ValueError):
            choose_loader("made-up")


class PersistentRunnerTest(unittest.TestCase):
    def test_ragged_input_preparation(self):
        runner = PersistentModelRunner()
        step = SchedulerStep(
            {"A": 2, "B": 1},
            (NewRequest("A", (3, 4)), NewRequest("B", (8,))),
        )
        runner._update_requests(step)
        batch = prepare_batch(runner.requests, step.scheduled_tokens)
        self.assertEqual(batch.input_ids, [3, 4, 8])
        self.assertEqual(batch.positions, [0, 1, 0])
        self.assertEqual(batch.query_start_loc, [0, 2, 3])

    def test_execute_must_be_consumed_before_next_execute(self):
        runner = PersistentModelRunner()
        step = SchedulerStep({"A": 1}, (NewRequest("A", (2,)),))
        runner.execute_model(step)
        with self.assertRaisesRegex(RuntimeError, "sample_tokens"):
            runner.execute_model(step)
        runner.sample_tokens()

    def test_zero_work_produces_empty_sample_output(self):
        runner = PersistentModelRunner()
        self.assertIsNone(runner.execute_model(SchedulerStep({})))
        self.assertEqual(runner.sample_tokens().req_ids, [])

    def test_preemption_releases_row_and_readd_gets_a_row(self):
        runner = PersistentModelRunner(max_num_reqs=1)
        runner.execute_model(
            SchedulerStep({"A": 1}, (NewRequest("A", (1,)),))
        )
        runner.sample_tokens()
        runner.execute_model(
            SchedulerStep(
                {"B": 1},
                (NewRequest("B", (2,)),),
                preempted_req_ids=frozenset({"A"}),
            )
        )
        self.assertEqual(runner.requests["B"].row, 0)

    def test_non_last_pipeline_rank_returns_intermediate(self):
        runner = PersistentModelRunner(is_last_pp_rank=False)
        result = runner.execute_model(
            SchedulerStep({"A": 2}, (NewRequest("A", (3, 4)),))
        )
        self.assertEqual(result, IntermediateTensors([3.0, 5.0]))
        self.assertEqual(runner.sample_tokens().req_ids, [])


class SamplingTest(unittest.TestCase):
    def test_greedy_sampling_is_argmax(self):
        self.assertEqual(
            sample_token([0.0, 3.0, 2.0], [], SamplingParams(), sample_position=0),
            1,
        )

    def test_penalties_are_applied_before_filtering(self):
        processed = process_logits(
            [0.0, 4.0, 3.0],
            [1, 1],
            SamplingParams(
                temperature=1.0, presence_penalty=1.0, frequency_penalty=1.0
            ),
        )
        self.assertEqual(processed, [0.0, 1.0, 3.0])

    def test_top_k_masks_all_but_k_candidates(self):
        processed = process_logits(
            [0.0, 1.0, 2.0, 3.0], [], SamplingParams(temperature=1, top_k=2)
        )
        self.assertEqual(processed[:2], [float("-inf"), float("-inf")])
        self.assertEqual(processed[2:], [2.0, 3.0])

    def test_seeded_sampling_is_deterministic(self):
        params = SamplingParams(temperature=1.0, seed=99)
        first = sample_token([1.0, 1.0, 1.0], [], params, sample_position=4)
        second = sample_token([1.0, 1.0, 1.0], [], params, sample_position=4)
        self.assertEqual(first, second)


class ExecutorTest(unittest.TestCase):
    def test_all_workers_receive_the_same_step(self):
        executor = MultiprocExecutorModel(world_size=3)
        step = SchedulerStep({"A": 1}, (NewRequest("A", (2,)),))
        executor.execute_model(step)
        self.assertTrue(all(worker.received_steps == [step] for worker in executor.workers))

    def test_only_selected_rank_result_is_observed(self):
        executor = MultiprocExecutorModel(world_size=2, output_rank=1)
        step = SchedulerStep({"A": 1}, (NewRequest("A", (2,)),))
        executor.execute_model(step)
        output = executor.sample_tokens()
        self.assertEqual(output.req_ids, ["A"])
        self.assertEqual(output.req_id_to_index, {"A": 0})
        self.assertEqual(
            output.sampled_token_ids,
            [executor.workers[1].model_runner.requests["A"].output_token_ids[-1:]],
        )


if __name__ == "__main__":
    unittest.main()
