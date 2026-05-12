import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crafter

from environment.achievements import ACHIEVEMENTS, TECH_TREE_ORDER
from llm.curriculum import HardcodedCurriculum


def _info(achievements: dict[str, int]) -> dict:
    return {"achievements": achievements, "inventory": {}}


class TechTreeIntegrityTests(unittest.TestCase):
    def test_tech_tree_covers_all_achievements(self):
        self.assertEqual(set(TECH_TREE_ORDER), set(ACHIEVEMENTS.keys()))
        self.assertEqual(len(TECH_TREE_ORDER), 22)

    def test_tech_tree_is_topologically_sorted(self):
        seen: set[str] = set()
        for key in TECH_TREE_ORDER:
            for prereq in ACHIEVEMENTS[key].prerequisites:
                self.assertIn(
                    prereq,
                    seen,
                    f"{key} appears before prerequisite {prereq}",
                )
            seen.add(key)

    def test_no_unknown_prerequisite_references(self):
        all_keys = set(ACHIEVEMENTS.keys())
        for key, achievement in ACHIEVEMENTS.items():
            for prereq in achievement.prerequisites:
                self.assertIn(prereq, all_keys, f"{key} references {prereq}")

    def test_catalog_keys_match_crafter_env(self):
        env = crafter.Env()
        try:
            env.reset()
            _, _, _, info = env.step(0)
            self.assertEqual(set(ACHIEVEMENTS.keys()), set(info["achievements"].keys()))
        finally:
            env.close()


class HardcodedCurriculumTests(unittest.TestCase):
    def test_first_task_is_collect_wood(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({}))

        self.assertIsNotNone(task)
        self.assertEqual(task.achievement_key, "collect_wood")
        self.assertEqual(task.name, "collect-wood")

    def test_second_task_is_collect_drink_after_wood(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({"collect_wood": 1}))

        self.assertEqual(task.achievement_key, "collect_drink")

    def test_skips_completed_tasks(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(
            _info({
                "collect_wood": 1,
                "collect_drink": 1,
                "eat_cow": 1,
                "collect_sapling": 1,
            })
        )

        self.assertEqual(task.achievement_key, "place_table")

    def test_respects_prerequisites_even_with_later_progress(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({"collect_stone": 1}))

        self.assertEqual(task.achievement_key, "collect_wood")

    def test_returns_none_when_all_done(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({key: 1 for key in TECH_TREE_ORDER}))

        self.assertIsNone(task)

    def test_is_task_complete_reads_info_flag(self):
        curriculum = HardcodedCurriculum()
        task = curriculum.propose_task(_info({}))

        self.assertFalse(curriculum.is_task_complete(task, _info({})))
        self.assertTrue(
            curriculum.is_task_complete(task, _info({"collect_wood": 1}))
        )

    def test_record_failed_does_not_change_proposal(self):
        curriculum = HardcodedCurriculum()
        task = curriculum.propose_task(_info({}))

        curriculum.record_task_failed(task, state_snapshot={"reason": "timeout"})
        next_task = curriculum.propose_task(_info({}))

        self.assertEqual(task.achievement_key, next_task.achievement_key)
        self.assertEqual(len(curriculum.failures), 1)


if __name__ == "__main__":
    unittest.main()
