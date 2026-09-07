import unittest

from examples.ch02_request_lifecycle import LifecycleDemo


class RequestLifecycleTest(unittest.TestCase):
    def test_offline_returns_final_outputs_in_request_order(self):
        demo = LifecycleDemo()
        demo.submit("req-B", "介绍 vLLM", max_tokens=2)
        demo.submit("req-A", "请 介绍 vLLM", max_tokens=3)

        outputs = demo.run_offline()

        self.assertEqual([output.request_id for output in outputs], ["req-A", "req-B"])
        self.assertEqual(outputs[0].text, "它让推理")
        self.assertEqual(outputs[1].text, "它让")
        self.assertTrue(all(output.finished for output in outputs))

    def test_online_returns_incremental_deltas(self):
        demo = LifecycleDemo()
        demo.submit("chatcmpl-1", "请 介绍 vLLM", max_tokens=3)

        chunks = [demo.tick_online()[0] for _ in range(3)]

        self.assertEqual([chunk.text for chunk in chunks], ["它", "让", "推理"])
        self.assertFalse(chunks[0].finished)
        self.assertTrue(chunks[-1].finished)

    def test_abort_uses_internal_id_but_returns_external_id(self):
        demo = LifecycleDemo()
        internal_id = demo.submit("chatcmpl-2", "请 介绍 vLLM", max_tokens=4)
        demo.tick_online()

        demo.core.abort(internal_id)
        chunks = demo.tick_online()

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].request_id, "chatcmpl-2")
        self.assertEqual(chunks[0].finish_reason, "abort")

    def test_frontend_registers_state_before_core_submission(self):
        demo = LifecycleDemo()
        internal_id = demo.submit("req-1", "介绍 vLLM", max_tokens=1)

        self.assertIn(internal_id, demo.output_processor.states)
        self.assertIn(internal_id, demo.core.requests)

    def test_input_length_validation(self):
        demo = LifecycleDemo()
        demo.input_processor.max_model_len = 4

        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            demo.submit("too-long", "请 介绍 vLLM", max_tokens=2)


if __name__ == "__main__":
    unittest.main()
