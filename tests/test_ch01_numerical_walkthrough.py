from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))

from ch01_numerical_walkthrough import attention_walkthrough, temperature_rows


class NumericalWalkthroughTest(unittest.TestCase):
    def test_temperature_distributions_sum_to_one(self) -> None:
        rows = temperature_rows()
        for _, probabilities in rows:
            self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(rows[0][1][0], rows[1][1][0])
        self.assertGreater(rows[1][1][0], rows[2][1][0])

    def test_attention_walkthrough(self) -> None:
        scores, probabilities, output = attention_walkthrough()
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(output[0], 1.2309211468639987)
        self.assertAlmostEqual(output[1], 0.9047401777683999)


if __name__ == "__main__":
    unittest.main()
