"""Application entry point."""
from __future__ import annotations

import argparse
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Voyager-Crafter episode.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the Crafter window after each environment step.",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=512,
        help="Crafter render size in pixels.",
    )
    parser.add_argument(
        "--render-step-delay",
        type=float,
        default=0.05,
        help="Seconds to wait between rendered environment steps.",
    )
    parser.add_argument(
        "--skill-library",
        default=None,
        help="Chroma collection name for the skill library.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = get_settings()
    if args.skill_library:
        settings.chroma.skills_collection = args.skill_library
    configure_logging(settings.logging.level)

    log = logging.getLogger("main")
    log.info("voyager-crafter starting up")
    log.info("skill library: %s", settings.chroma.skills_collection)

    env = CrafterEnv(**settings.environment.crafter_kwargs)
    executor = Executor(
        max_steps_per_skill=settings.executor.max_steps_per_skill,
        health_threshold=settings.executor.health_interrupt_threshold,
        render=args.render,
        render_size=args.render_size,
        render_delay_s=args.render_step_delay,
    )
    skill_manager = SkillManager(
        repository=ChromaSkillRepository(settings.chroma),
        embedder=TextEmbedder(settings.embedding.model_name),
        config=settings.embedding,
    )

    with RunLogger(config_snapshot=settings.snapshot()) as run_log:
        log.info("running one episode (session=%s)", run_log.session_id)
        if args.render:
            log.info(
                "render enabled: size=%d step_delay=%.3f",
                args.render_size,
                args.render_step_delay,
            )
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
