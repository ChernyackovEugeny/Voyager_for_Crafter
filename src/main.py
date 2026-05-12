"""Application entry point."""
from __future__ import annotations

import logging

from agent.agent import Agent
from agent.executor import Executor
from agent.memory import SpatialMemory
from analytics.log_utils import log_session_finalize
from analytics.run_logger import RunLogger
from config import get_settings
from environment.wrapper import CrafterEnv
from llm.codegen import CodeGen
from llm.curriculum import HardcodedCurriculum
from observability.logging_config import configure_logging
from skills.embedder import TextEmbedder
from skills.skill_manager import SkillManager
from storage.skill_repository import ChromaSkillRepository


def main() -> None:
    settings = get_settings()
    configure_logging(settings.logging.level)

    log = logging.getLogger("main")
    log.info("voyager-crafter starting up")

    env = CrafterEnv(**settings.environment.crafter_kwargs)
    executor = Executor(
        max_steps_per_skill=settings.executor.max_steps_per_skill,
        health_threshold=settings.executor.health_interrupt_threshold,
    )
    skill_manager = SkillManager(
        repository=ChromaSkillRepository(settings.chroma),
        embedder=TextEmbedder(settings.embedding.model_name),
        config=settings.embedding,
    )

    with RunLogger(config_snapshot=settings.snapshot()) as run_log:
        log.info("running one episode (session=%s)", run_log.session_id)
        agent = Agent(
            env=env,
            curriculum=HardcodedCurriculum(),
            skill_manager=skill_manager,
            codegen=CodeGen(),
            executor=executor,
            memory=SpatialMemory(),
            reuse_threshold=settings.embedding.similarity_reuse_threshold,
            max_iterations_per_episode=(
                settings.executor.max_iterations_per_episode
            ),
            run_logger=run_log,
        )
        summary = agent.run()
        log_session_finalize(
            run_log,
            final_achievements=summary["final_state"]["info"].get(
                "achievements", {}
            ),
        )

    env.close()
    log.info("voyager-crafter exited cleanly")


if __name__ == "__main__":
    main()
