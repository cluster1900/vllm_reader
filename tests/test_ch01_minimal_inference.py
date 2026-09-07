from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ch01_minimal_inference.py"
)
SPEC = importlib.util.spec_from_file_location("ch01_minimal_inference", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MinimalInferenceTest(unittest.TestCase):
    def test_prefill_matches_full_recompute(self) -> None:
        model = MODULE.ToyDecoder()
        token_ids = [0, 1, 2, 3]
        cached_logits, _ = model.prefill(token_ids)
        full_logits = model.forward_full(token_ids)
        for cached, full in zip(cached_logits, full_logits):
            self.assertLess(MODULE.max_abs_diff(cached, full), 1e-12)

    def test_cached_decode_matches_full_recompute(self) -> None:
        model = MODULE.ToyDecoder()
        token_ids = [0, 1, 2]
        cached_logits, cache = model.prefill(token_ids)
        for token_id in [3, 4, 6]:
            token_ids.append(token_id)
            cached_logits.append(model.decode_one(token_id, cache))
            recomputed = model.forward_full(token_ids)[-1]
            self.assertLess(MODULE.max_abs_diff(cached_logits[-1], recomputed), 1e-12)

    def test_temperature_zero_is_greedy(self) -> None:
        logits = [0.2, 1.7, -0.3, 1.2]
        sampled = MODULE.sample_logits(logits, temperature=0.0, seed=99)
        self.assertEqual(sampled, 1)

    def test_top_k_one_is_deterministic(self) -> None:
        logits = [0.2, 1.7, -0.3, 1.2]
        sampled = MODULE.sample_logits(
            logits, temperature=1.0, top_k=1, top_p=1.0, seed=99
        )
        self.assertEqual(sampled, 1)


if __name__ == "__main__":
    unittest.main()
