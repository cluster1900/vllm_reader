"""Re-compute the worked numbers of the ch04 formulas.

The helpers below are the formulas of book/04-scheduler/README.md written as code
(notation: glossary/notation.md).  Each test replays the chapter's worked example
step by step.  ``ReadmeSyncTest`` reads the chapter and asserts that the numbers
produced by these helpers are the ones printed in the README, so a changed number
in either place fails the test.
"""

import unittest
from pathlib import Path

from examples.ch04_continuous_batching import Request, TeachingScheduler


def n_target(n_prompt, n_out, n_spec):
    return n_prompt + n_out + n_spec


def n_new(target, computed, placeholders=0):
    return target + placeholders - computed


def n_chunk(remaining, budget, budget_in, c_threshold=None, c_other=None):
    """min(...) of 4.10; None means "this cap is not active"."""
    caps = [remaining, budget, budget_in]
    caps += [c for c in (c_threshold, c_other) if c is not None]
    return min(caps)


def mm_split(computed, new, pos_mm, n_mm):
    return computed + new < pos_mm + n_mm


def mm_rollback(computed, pos_mm, delta=0):
    return max(0, pos_mm - (computed + delta))


def drafts_unused(computed, new, n_tokens, n_out, a_async, n_max):
    return computed + new < n_tokens or n_out + 1 + a_async >= n_max


def reject_rollback(computed, placeholders, n_spec, k):
    rejected = n_spec - (k - 1)
    return computed - rejected, placeholders - rejected


def stop_trim(computed, placeholders, k, j):
    return computed - (k - j), placeholders - k


def defer(i_pf, n_step, cap_bound, has_decode):
    throttle = i_pf > 1 and n_step % i_pf != 0
    return throttle and not cap_bound and has_decode


class Ch04FormulaWalkthroughTest(unittest.TestCase):
    def test_target_and_new(self):
        self.assertEqual(n_target(4, 1, 2), 7)
        self.assertEqual(n_target(4, 1, 0), 5)
        self.assertEqual(n_new(4, 0), 4)
        self.assertEqual(n_new(5, 4), 1)
        self.assertEqual(n_new(7, 4), 3)

    def test_async_placeholders_walkthrough(self):
        tokens, computed, placeholders = 4, 0, 0
        self.assertEqual(n_new(tokens, computed), 4)
        computed, placeholders = computed + 4, placeholders + 1  # step 1 scheduled
        self.assertEqual(n_new(tokens, computed, placeholders), 1)
        computed, placeholders = computed + 1, placeholders + 1  # step 2 scheduled
        self.assertEqual((computed, placeholders), (5, 2))
        tokens, placeholders = tokens + 1, placeholders - 1  # step 1 output arrives
        self.assertEqual(n_new(tokens, computed, placeholders), 1)

    def test_token_budget_and_chunk_example(self):
        c_step = 1024  # input budget starts at 1024 as well and drops in step
        sizes = []
        budget = c_step
        for remaining in (800, 800, 800):
            size = n_chunk(remaining, budget, budget, 400)
            sizes.append(size)
            budget -= size
        self.assertEqual(sizes, [400, 400, 224])
        self.assertLessEqual(sum(sizes), c_step)
        # third round: R0 and R1 decode one token each, R2 has 352 left
        budget = c_step - 1 - 1
        self.assertEqual(800 - 224 - 224, 352)
        self.assertEqual(n_chunk(352, budget, budget, 400), 352)

    def test_input_budget_with_draft_slots(self):
        token_budget = input_budget = 8
        draft_slots = 2
        scheduled = 0
        for _ in range(4):
            if input_budget <= draft_slots:
                break
            size = min(1, token_budget, input_budget - draft_slots)
            token_budget -= size
            input_budget -= size + draft_slots
            scheduled += 1
        self.assertEqual((scheduled, token_budget, input_budget), (2, 6, 2))

    def test_adaptive_threshold_example(self):
        self.assertEqual(1024 // 2, 512)
        self.assertEqual(max(100, 1024 // 2), 512)

    def test_prefix_hit_example(self):
        n_tokens, hit_local, hit_ext = 100, 4 * 16, 16
        computed = hit_local + hit_ext
        self.assertEqual(computed, 80)
        self.assertLessEqual(computed, n_tokens)
        self.assertEqual(n_tokens - computed, 20)
        self.assertEqual((100 + 30) - 0, 130)

    def test_mm_split_examples(self):
        # chapter example: text TTTT + 10 image tokens
        self.assertTrue(mm_split(4, 5, 4, 10))
        self.assertFalse(mm_split(4, 10, 4, 10))
        self.assertEqual(mm_rollback(0, 4), 4)
        self.assertEqual(mm_rollback(0, 4, delta=1), 3)
        self.assertEqual(mm_rollback(4, 4), 0)
        self.assertTrue(mm_split(7, 5, 4, 10))
        self.assertEqual(mm_rollback(7, 4), 0)

    def test_mm_split_matches_upstream_boundary_test(self):
        # tests/v1/core/test_scheduler.py::test_no_mm_input_chunking_at_item_boundary
        def ask(computed, new, pos_mm=400, n_mm=800):
            if mm_split(computed, new, pos_mm, n_mm):
                return mm_rollback(computed, pos_mm)
            return new

        self.assertEqual(
            [ask(0, 500), ask(400, 500), ask(400, 800), ask(600, 500), ask(600, 600)],
            [400, 0, 800, 0, 600],
        )

    def test_drafts_unused_examples(self):
        self.assertTrue(drafts_unused(0, 64, 100, 0, 0, 10))
        self.assertFalse(49 + 1 < 50)
        self.assertTrue(drafts_unused(49, 1, 50, 9, 0, 10))
        self.assertFalse(drafts_unused(49, 1, 50, 9, 0, 20))

    def test_confirmed_progress_example(self):
        pos_mm, n_mm = 4, 10
        confirmed = 16 - max(0, 4)
        self.assertEqual(confirmed, 12)
        self.assertFalse(pos_mm + n_mm <= confirmed)
        self.assertTrue(pos_mm + n_mm <= 16 - max(0, 0))

    def test_rejection_rollback_walkthrough(self):
        tokens, computed, placeholders, n_spec = 20, 19, 0, 3
        self.assertEqual(n_new(tokens + n_spec, computed, placeholders), 4)
        computed, placeholders = computed + 4, placeholders + 1 + n_spec
        self.assertEqual((computed, placeholders), (23, 4))
        computed, placeholders = reject_rollback(computed, placeholders, n_spec, k=3)
        self.assertEqual((computed, placeholders), (22, 3))
        computed, placeholders = stop_trim(computed, placeholders, k=3, j=2)
        self.assertEqual((computed, placeholders), (21, 0))
        tokens += 2
        self.assertEqual(computed, tokens - 1)

    def test_prefill_throttle_table(self):
        got = [defer(4, n, False, True) for n in range(6)]
        self.assertEqual(got, [False, True, True, True, False, True])
        self.assertFalse(defer(4, 2, True, True))
        self.assertFalse(defer(4, 2, False, False))
        released = [n for n in range(9) if n % 4 == 0]
        self.assertEqual(released, [0, 4, 8])  # third chunk runs at step 8, done at step 9

    def test_section_4_11_trace_matches_teaching_scheduler(self):
        scheduler = TeachingScheduler(
            max_num_scheduled_tokens=8,
            max_num_seqs=3,
            kv_capacity_tokens=64,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=6,
            reserve_full_prompt=False,
        )
        scheduler.add_request(Request("A", prompt_tokens=12, max_new_tokens=3))
        scheduler.add_request(Request("B", prompt_tokens=3, max_new_tokens=2, arrival_step=1))
        scheduler.add_request(Request("C", prompt_tokens=2, max_new_tokens=2, arrival_step=2))
        rows = []
        while scheduler.has_work():
            output = scheduler.step()
            rows.append(scheduler.format_step(output).split(" preempted=")[0])
        self.assertEqual(
            rows,
            [
                "step=00 work=[A:p6]",
                "step=01 work=[A:p6, B:p2]",
                "step=02 work=[A:d1, B:p1, C:p2]",
                "step=03 work=[A:d1, B:d1, C:d1]",
            ],
        )


    def test_encoder_free_and_truncate_examples(self):
        # 4.16.2: release iff pos + n_mm <= computed - placeholders
        def released(pos, n_mm, computed, placeholders):
            return pos + n_mm <= computed - placeholders

        self.assertFalse(released(4, 10, 16, 3))
        self.assertTrue(released(4, 10, 16, 1))
        # truncate: min(N_new, pos - computed)
        self.assertEqual(min(6, 4 - 0), 4)


class ReadmeSyncTest(unittest.TestCase):
    """The README text must contain the numbers the helpers compute."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.readme = (root / "book" / "04-scheduler" / "README.md").read_text(encoding="utf-8")

    def assertInReadme(self, text):
        self.assertIn(text, self.readme)

    def test_n_target_example(self):
        t = n_target(4, 1, 2)
        self.assertEqual(t, 7)
        self.assertInReadme(f"$N_{{\\mathrm{{target}}}}=5+2={t}$")

    def test_chunk_examples(self):
        self.assertInReadme(f"$\\min(800,1024,1024,400)={n_chunk(800, 1024, 1024, 400)}$")
        self.assertInReadme(f"$\\min(800,624,624,400)={n_chunk(800, 624, 624, 400)}$")
        self.assertInReadme(f"$\\min(800,224,224,400)={n_chunk(800, 224, 224, 400)}$")
        self.assertInReadme(f"$\\min(352,1022,1022,400)={n_chunk(352, 1022, 1022, 400)}$")

    def test_mm_rollback_examples(self):
        for computed, pos, delta, shown in [(0, 4, 0, "4-0"), (0, 4, 1, "4-1"), (4, 4, 0, "4-4"), (7, 4, 0, "4-7")]:
            value = mm_rollback(computed, pos, delta)
            self.assertInReadme(f"\\max(0,{shown})={value}$")

    def test_hit_example(self):
        self.assertEqual(4 * 16 + 16, 80)
        self.assertInReadme("$N_{\\mathrm{computed}}=64+16=80$ token")
        self.assertInReadme(f"$N_{{\\mathrm{{new}}}}=100-80={100 - 80}$")


if __name__ == "__main__":
    unittest.main()
