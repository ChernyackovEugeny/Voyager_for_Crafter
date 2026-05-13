import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import crafter

from environment.ids import NAME_TO_ID
from environment.achievements import ACHIEVEMENTS, TECH_TREE_ORDER
from llm.curriculum import HardcodedCurriculum, SurvivalSettings


def _info(achievements: dict[str, int], inventory=None) -> dict:
    return {"achievements": achievements, "inventory": inventory or {}}


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
    def test_first_task_secures_water(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({}))

        self.assertIsNotNone(task)
        self.assertEqual(task.achievement_key, "collect_drink")
        self.assertEqual(task.name, "secure-water")

    def test_food_is_prioritized_after_water(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({"collect_drink": 1}))

        self.assertEqual(task.achievement_key, "eat_cow")
        self.assertEqual(task.name, "secure-food")

    def test_collect_wood_is_prioritized_before_shelter(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(
            _info({"collect_drink": 1, "eat_cow": 1})
        )

        self.assertEqual(task.achievement_key, "collect_wood")
        self.assertEqual(task.name, "collect-wood")

    def test_shelter_is_prioritized_after_water_food_and_wood(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(
            _info({"collect_drink": 1, "eat_cow": 1, "collect_wood": 1})
        )

        self.assertIsNone(task.achievement_key)
        self.assertEqual(task.name, "build-shelter")
        self.assertEqual(task.completion_conditions[0].key, "achievements.place_table")

    def test_skip_excludes_task_for_current_proposal(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({}), skip={"collect_drink"})

        self.assertEqual(task.achievement_key, "eat_cow")

    def test_skips_completed_tasks(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(
            _info({
                "collect_wood": 1,
                "collect_drink": 1,
                "eat_cow": 1,
                "collect_sapling": 1,
                "place_table": 1,
            })
        )

        self.assertEqual(task.achievement_key, "make_wood_pickaxe")

    def test_respects_prerequisites_even_with_later_progress(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(
            _info({
                "collect_stone": 1,
                "collect_drink": 1,
                "eat_cow": 1,
                "place_table": 1,
            })
        )

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
            curriculum.is_task_complete(task, _info({"collect_drink": 1}))
        )

    def test_record_failed_does_not_change_proposal(self):
        curriculum = HardcodedCurriculum()
        task = curriculum.propose_task(_info({}))

        curriculum.record_task_failed(task, state_snapshot={"reason": "timeout"})
        next_task = curriculum.propose_task(_info({}))

        self.assertEqual(task.achievement_key, next_task.achievement_key)
        self.assertEqual(len(curriculum.failures), 1)

    def test_survive_task_is_proposed_before_achievements_when_in_danger(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(_info({}, {"health": 6, "food": 9, "drink": 9}))

        self.assertEqual(task.name, "survive")
        self.assertIsNone(task.achievement_key)
        self.assertEqual(task.skip_key, "survive")

    def test_survive_task_is_proposed_when_hostile_visible(self):
        curriculum = HardcodedCurriculum()
        semantic = np.zeros((64, 64), dtype=int)
        semantic[11, 10] = NAME_TO_ID["zombie"]
        info = _info(
            {},
            {"health": 9, "food": 9, "drink": 9},
        )
        info.update({"semantic": semantic, "player_pos": (10, 10), "view_size": (9, 9)})

        task = curriculum.propose_task(info)

        self.assertEqual(task.name, "survive")

    def test_hostile_visible_respects_survive_skip(self):
        curriculum = HardcodedCurriculum()
        semantic = np.zeros((64, 64), dtype=int)
        semantic[11, 10] = NAME_TO_ID["skeleton"]
        info = _info({}, {"health": 9, "food": 9, "drink": 9})
        info.update({"semantic": semantic, "player_pos": (10, 10), "view_size": (9, 9)})

        task = curriculum.propose_task(info, skip={"survive"})

        self.assertNotEqual(task.name, "survive")
        self.assertEqual(task.achievement_key, "collect_drink")

    def test_survive_not_complete_while_hostile_visible(self):
        curriculum = HardcodedCurriculum()
        semantic = np.zeros((64, 64), dtype=int)
        semantic[11, 10] = NAME_TO_ID["zombie"]
        info = _info({}, {"health": 9, "food": 9, "drink": 9})
        info.update({"semantic": semantic, "player_pos": (10, 10), "view_size": (9, 9)})
        task = curriculum.propose_task(info)

        self.assertFalse(curriculum.is_task_complete(task, info))

    def test_survive_task_respects_skip(self):
        curriculum = HardcodedCurriculum()

        task = curriculum.propose_task(
            _info({}, {"health": 6, "food": 9, "drink": 9}),
            skip={"survive"},
        )

        self.assertEqual(task.achievement_key, "collect_drink")

    def test_survive_complete_requires_exit_thresholds(self):
        curriculum = HardcodedCurriculum(
            survival=SurvivalSettings(
                enter_health=6,
                enter_food=3,
                enter_drink=3,
                exit_health=8,
                exit_food=6,
                exit_drink=6,
            )
        )
        task = curriculum.propose_task(
            _info({}, {"health": 6, "food": 9, "drink": 9})
        )

        self.assertFalse(
            curriculum.is_task_complete(
                task,
                _info({}, {"health": 7, "food": 6, "drink": 6}),
            )
        )
        self.assertTrue(
            curriculum.is_task_complete(
                task,
                _info({}, {"health": 8, "food": 6, "drink": 6}),
            )
        )

    def test_survival_hysteresis_keeps_survive_until_recovered(self):
        curriculum = HardcodedCurriculum(
            survival=SurvivalSettings(
                enter_health=6,
                enter_food=3,
                enter_drink=3,
                exit_health=8,
                exit_food=6,
                exit_drink=6,
            )
        )

        danger_task = curriculum.propose_task(
            _info({}, {"health": 5, "food": 9, "drink": 9})
        )
        middle_task = curriculum.propose_task(
            _info({}, {"health": 7, "food": 9, "drink": 9})
        )
        recovered_task = curriculum.propose_task(
            _info({}, {"health": 8, "food": 6, "drink": 6})
        )

        self.assertEqual(danger_task.name, "survive")
        self.assertEqual(middle_task.achievement_key, "collect_drink")
        self.assertEqual(recovered_task.achievement_key, "collect_drink")


if __name__ == "__main__":
    unittest.main()
