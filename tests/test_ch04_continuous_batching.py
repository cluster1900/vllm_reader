import unittest

from examples.ch04_continuous_batching import (
    Request,
    RequestStatus,
    SchedulingPolicy,
    TeachingScheduler,
)


class ContinuousBatchingTest(unittest.TestCase):
    def test_new_request_joins_before_older_request_finishes(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=5,
            max_num_seqs=2,
            kv_capacity_tokens=64,
            long_prefill_token_threshold=4,
            reserve_full_prompt=False,
        )
        scheduler.add_request(Request("long", 8, 3, arrival_step=0))
        scheduler.add_request(Request("short", 1, 1, arrival_step=1))

        scheduler.step()
        output = scheduler.step()

        self.assertIn("short", output.num_scheduled_tokens)
        self.assertNotEqual(scheduler.requests["long"].status, RequestStatus.FINISHED)

    def test_chunked_prefill_shares_budget_with_decode(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=6,
            max_num_seqs=3,
            kv_capacity_tokens=64,
            long_prefill_token_threshold=4,
            reserve_full_prompt=False,
        )
        scheduler.add_request(Request("decode", 1, 3))
        first = scheduler.step()
        self.assertEqual(first.num_scheduled_tokens, {"decode": 1})
        scheduler.add_request(Request("prefill", 20, 1, arrival_step=1))

        second = scheduler.step()

        self.assertEqual(second.num_scheduled_tokens["decode"], 1)
        self.assertEqual(second.num_scheduled_tokens["prefill"], 4)
        self.assertLessEqual(second.token_budget_used, 6)

    def test_token_input_and_sequence_budgets_hold(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=10,
            max_num_batched_tokens=8,
            max_num_seqs=2,
            kv_capacity_tokens=100,
            draft_slots_per_request=1,
            reserve_full_prompt=False,
        )
        for i in range(3):
            scheduler.add_request(Request(str(i), 10, 1))

        output = scheduler.step()

        self.assertLessEqual(output.token_budget_used, 10)
        self.assertLessEqual(output.input_budget_used, 8)
        self.assertLessEqual(len(scheduler.running), 2)
        self.assertEqual(len(output.scheduled), 1)

    def test_priority_uses_lower_number_then_arrival(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=20,
            max_num_seqs=4,
            kv_capacity_tokens=100,
            policy=SchedulingPolicy.PRIORITY,
        )
        scheduler.add_request(Request("low", 2, 1, priority=3))
        scheduler.add_request(Request("later", 2, 1, priority=0, arrival_step=0))
        scheduler.add_request(Request("earlier", 2, 1, priority=0, arrival_step=0))

        output = scheduler.step()

        self.assertEqual(
            [item.request_id for item in output.scheduled],
            ["earlier", "later", "low"],
        )

    def test_kv_pressure_preempts_worst_priority_and_recomputes(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=20,
            max_num_seqs=2,
            kv_capacity_tokens=9,
            policy=SchedulingPolicy.PRIORITY,
            reserve_full_prompt=False,
        )
        low = Request("low", 4, 3, priority=5)
        high = Request("high", 4, 3, arrival_step=1, priority=0)
        scheduler.add_request(low)
        scheduler.add_request(high)

        scheduler.step()
        scheduler.step()
        output = scheduler.step()

        self.assertIn("low", output.preempted)
        self.assertEqual(low.status, RequestStatus.PREEMPTED)
        self.assertEqual(low.num_computed_tokens, 0)
        self.assertEqual(low.num_preemptions, 1)
        self.assertEqual(high.status, RequestStatus.RUNNING)

        scheduler.run()
        self.assertEqual(low.status, RequestStatus.FINISHED)
        self.assertGreaterEqual(low.num_preemptions, 1)

    def test_fcfs_preempts_running_tail(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=20,
            max_num_seqs=2,
            kv_capacity_tokens=9,
            policy=SchedulingPolicy.FCFS,
            reserve_full_prompt=False,
        )
        first = Request("first", 4, 3)
        second = Request("second", 4, 3, arrival_step=1)
        scheduler.add_request(first)
        scheduler.add_request(second)

        scheduler.step()
        scheduler.step()
        output = scheduler.step()

        self.assertIn("second", output.preempted)
        self.assertEqual(second.status, RequestStatus.PREEMPTED)
        self.assertNotEqual(first.status, RequestStatus.PREEMPTED)

    def test_without_chunking_head_request_is_not_bypassed(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=8,
            max_num_seqs=3,
            kv_capacity_tokens=100,
            enable_chunked_prefill=False,
        )
        scheduler.add_request(Request("long", 10, 1))
        scheduler.add_request(Request("short", 2, 1))

        output = scheduler.step()

        self.assertEqual(output.scheduled, ())
        self.assertEqual([r.request_id for r in scheduler.waiting], ["long", "short"])


if __name__ == "__main__":
    unittest.main()
