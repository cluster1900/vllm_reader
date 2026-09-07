import unittest

from examples.ch03_engine_core_loop import (
    ClientKind,
    EngineCoreOutput,
    MessageType,
    PauseState,
    ShutdownState,
    TeachingEngineCore,
    TeachingEngineCoreProc,
    select_client,
)


class ClientSelectionTest(unittest.TestCase):
    def test_client_selection_matrix(self):
        self.assertEqual(
            select_client(multiprocess=False, asyncio_mode=False), ClientKind.INPROC
        )
        self.assertEqual(
            select_client(multiprocess=True, asyncio_mode=False), ClientKind.SYNC_MP
        )
        self.assertEqual(
            select_client(multiprocess=True, asyncio_mode=True), ClientKind.ASYNC_MP
        )
        with self.assertRaises(NotImplementedError):
            select_client(multiprocess=False, asyncio_mode=True)


class EngineCoreStepTest(unittest.TestCase):
    def test_normal_step_waits_for_future_and_finishes(self):
        core = TeachingEngineCore()
        core.add_request("r0", max_tokens=1)

        outputs, model_executed = core.step()

        self.assertTrue(model_executed)
        self.assertEqual(core.executor.launched_batches, [1])
        self.assertEqual(outputs[0][0].token_ids, (100,))
        self.assertEqual(outputs[0][0].finish_reason, "length")

    def test_abort_between_launch_and_update_wins(self):
        core = TeachingEngineCore()
        core.add_request("r0", max_tokens=2)

        outputs, _ = core.step(lambda engine: engine.queue_abort("r0"))

        self.assertEqual(outputs[0], (EngineCoreOutput("r0", (), "abort"),))
        self.assertEqual(core.scheduler.requests["r0"].settled_tokens, 0)

    def test_batch_queue_fills_then_drains_fifo(self):
        core = TeachingEngineCore(max_concurrent_batches=2)
        core.add_request("r0", max_tokens=3)

        first, _ = core.step_with_batch_queue()
        second, _ = core.step_with_batch_queue()
        third, _ = core.step_with_batch_queue()
        fourth, _ = core.step_with_batch_queue()

        self.assertIsNone(first)
        self.assertEqual(second[0][0].token_ids, (100,))
        self.assertEqual(third[0][0].token_ids, (101,))
        self.assertEqual(fourth[0][0].token_ids, (102,))
        self.assertEqual(fourth[0][0].finish_reason, "length")
        self.assertEqual(core.executor.launched_batches, [1, 2, 3])

    def test_pause_keep_freezes_and_resume_releases(self):
        core = TeachingEngineCore()
        core.add_request("old", max_tokens=1)
        core.pause("keep")
        core.add_request("new", max_tokens=1)

        self.assertEqual(core.scheduler.pause_state, PauseState.PAUSED_ALL)
        self.assertEqual(core.step(), ({}, False))

        core.resume()
        outputs, _ = core.step()
        self.assertEqual(outputs[0][0].request_id, "old")


class EngineCoreProcTest(unittest.TestCase):
    def test_executor_failure_message_is_fatal(self):
        proc = TeachingEngineCoreProc(TeachingEngineCore())
        proc.submit(MessageType.EXECUTOR_FAILED)

        with self.assertRaisesRegex(RuntimeError, "Executor failed"):
            proc.run_once()

    def test_zero_timeout_shutdown_aborts_and_stops(self):
        proc = TeachingEngineCoreProc(TeachingEngineCore(), shutdown_timeout=0)
        proc.submit(MessageType.ADD, ("r0", 4))
        proc.run_once()
        proc.request_shutdown()

        keep_running = proc.run_once()

        self.assertFalse(keep_running)
        self.assertEqual(proc.shutdown_state, ShutdownState.SHUTTING_DOWN)
        self.assertEqual(proc.output_queue[-1].finish_reason, "abort")


if __name__ == "__main__":
    unittest.main()
