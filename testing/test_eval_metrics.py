import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environment.achievements import ACHIEVEMENTS
from eval.metrics import (
    achievement_success_rates,
    crafter_score,
    crafter_score_from_success_rates,
)


class EvalMetricsTests(unittest.TestCase):
    def test_success_rates_count_episode_level_unlocks(self):
        episodes = [
            {"collect_wood": 1, "collect_drink": 0},
            {"collect_wood": 1, "collect_drink": 1},
            {"collect_wood": 0, "collect_drink": 0},
        ]

        rates = achievement_success_rates(episodes)

        self.assertAlmostEqual(rates["collect_wood"], 2 / 3)
        self.assertAlmostEqual(rates["collect_drink"], 1 / 3)
        self.assertEqual(rates["collect_diamond"], 0.0)

    def test_score_matches_reference_percent_formula(self):
        rates = {key: 0.0 for key in ACHIEVEMENTS}
        rates["collect_wood"] = 1.0
        rates["collect_drink"] = 0.5

        expected_percent = math.exp(
            sum(math.log1p(100.0 * rate) for rate in rates.values())
            / len(rates)
        ) - 1.0

        self.assertAlmostEqual(
            crafter_score_from_success_rates(rates),
            expected_percent / 100.0,
        )

    def test_empty_input_scores_zero(self):
        self.assertEqual(crafter_score([]), 0.0)

    def test_rejects_rates_outside_unit_interval(self):
        rates = {key: 0.0 for key in ACHIEVEMENTS}
        rates["collect_wood"] = 1.1

        with self.assertRaises(ValueError):
            crafter_score_from_success_rates(rates)


if __name__ == "__main__":
    unittest.main()
