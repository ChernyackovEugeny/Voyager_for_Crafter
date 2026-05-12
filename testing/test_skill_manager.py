import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import EmbeddingConfig
from skills.skill_manager import SkillManager
from storage.in_memory_skill_repository import InMemorySkillRepository


class FakeEmbedder:
    dim = 4

    _vectors = {
        "collect wood": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "gather wood": np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32),
        "mine stone": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "fight zombie": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }

    def encode(self, text: str) -> np.ndarray:
        base = text.split(". Uses:", 1)[0]
        if base in self._vectors:
            return self._vectors[base].copy()
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class SkillManagerTests(unittest.TestCase):
    def setUp(self):
        cfg = EmbeddingConfig(
            model_name="fake",
            top_k=5,
            similarity_reuse_threshold=0.85,
            similarity_dedup_threshold=0.90,
        )
        self.repo = InMemorySkillRepository()
        self.manager = SkillManager(
            repository=self.repo,
            embedder=FakeEmbedder(),
            config=cfg,
        )

    def test_save_fresh_skill_succeeds(self):
        result = self.manager.save(
            name="collect_wood",
            code="def f(state):\n    state = yield do_action()\n",
            task="collect wood",
        )

        self.assertTrue(result.saved)
        self.assertEqual(result.outcome, "ok")
        self.assertEqual(result.skill.name, "collect_wood")
        self.assertEqual(
            result.skill.description,
            "collect wood. Uses: do_action",
        )

    def test_save_duplicate_name_rejected(self):
        self.manager.save(name="one", code="def f(state): yield 0", task="collect wood")

        result = self.manager.save(
            name="one",
            code="def f(state): yield 0",
            task="mine stone",
        )

        self.assertFalse(result.saved)
        self.assertEqual(result.outcome, "name_taken")

    def test_save_duplicate_embedding_rejected(self):
        self.manager.save(
            name="collect_wood",
            code="def f(state): yield 0",
            task="collect wood",
        )

        result = self.manager.save(
            name="gather_wood",
            code="def f(state): yield 0",
            task="gather wood",
        )

        self.assertFalse(result.saved)
        self.assertEqual(result.outcome, "duplicate")
        self.assertEqual(result.similar_to, "collect_wood")
        self.assertGreater(result.similarity, 0.90)

    def test_retrieve_returns_candidates_sorted_by_similarity(self):
        self.manager.save(name="wood", code="def f(state): yield 0", task="collect wood")
        self.manager.save(name="stone", code="def f(state): yield 0", task="mine stone")
        self.manager.save(name="zombie", code="def f(state): yield 0", task="fight zombie")

        candidates = self.manager.retrieve("collect wood", k=3)

        self.assertEqual([c.skill.name for c in candidates], ["wood", "stone", "zombie"])
        self.assertGreater(candidates[0].similarity, candidates[1].similarity)

    def test_record_success_and_failure_update_counters(self):
        self.manager.save(name="wood", code="def f(state): yield 0", task="collect wood")

        self.manager.record_success("wood")
        self.manager.record_success("wood")
        self.manager.record_failure("wood")

        skill = self.repo.get("wood")
        self.assertEqual(skill.success_count, 2)
        self.assertEqual(skill.fail_count, 1)

    def test_exists_reports_repository_presence(self):
        self.assertFalse(self.manager.exists("wood"))
        self.manager.save(name="wood", code="def f(state): yield 0", task="collect wood")
        self.assertTrue(self.manager.exists("wood"))

    def test_repository_list_count_and_delete(self):
        self.manager.save(name="wood", code="def f(state): yield 0", task="collect wood")
        self.manager.save(name="stone", code="def f(state): yield 0", task="mine stone")

        self.assertEqual(self.repo.count(), 2)
        self.assertEqual(
            [skill.name for skill in self.repo.list_skills()],
            ["stone", "wood"],
        )
        self.assertTrue(self.repo.delete("wood"))
        self.assertFalse(self.repo.delete("missing"))
        self.assertEqual(self.repo.count(), 1)

    def test_missing_metric_update_logs_and_does_not_raise(self):
        with self.assertLogs("skills.skill_manager", level="WARNING") as logs:
            self.manager.record_failure("missing")

        self.assertIn("missing on metrics update", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
