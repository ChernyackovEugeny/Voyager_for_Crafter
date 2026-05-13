"""Application entry point."""
from __future__ import annotations

import argparse
import logging

from agent.agent import Agent
from agent.executor import Executor
from agent.memory import SpatialMemory
from agent.strategies.inference import InferenceStrategy
from agent.strategies.training import TrainingStrategy
from analytics.log_utils import log_session_finalize
from analytics.run_logger import RunLogger
from config import get_settings
from environment.wrapper import CrafterEnv
from llm.codegen import CodeGen
from llm.curriculum import HardcodedCurriculum
from llm.fix_bug import FixBug
from llm.reflection import Reflection
from observability.logging_config import configure_logging
from skills.embedder import TextEmbedder
from skills.runner import SkillRuntime, load_skill
from skills.skill_manager import SkillManager
from storage.skill_repository import ChromaSkillRepository


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Voyager-Crafter episode.")
    parser.add_argument(
        "--mode",
        choices=("train", "inference"),
        default="train",
        help="Agent mode: train can call LLM and save skills; inference only reuses.",
    )
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
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to run in this session.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Environment seed for reproducible single-session runs.",
    )
    parser.add_argument(
        "--eval-id",
        default=None,
        help="Evaluation run identifier stored in analytics metadata.",
    )
    parser.add_argument(
        "--eval-run-idx",
        type=int,
        default=None,
        help="Training-run index stored in analytics metadata.",
    )
    parser.add_argument(
        "--early-stop-window",
        type=int,
        default=None,
        dest="early_stop_patience",
        help="Deprecated alias for --early-stop-patience.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Stop after N consecutive episodes without new achievements.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = get_settings()
    if args.skill_library:
        settings.chroma.skills_collection = args.skill_library
    episodes = args.episodes or settings.analytics.episodes
    early_stop_patience = (
        args.early_stop_patience
        if args.early_stop_patience is not None
        else settings.analytics.early_stop_patience
    )
    configure_logging(settings.logging.level)

    log = logging.getLogger("main")
    log.info("voyager-crafter starting up")
    log.info("mode: %s", args.mode)
    log.info("skill library: %s", settings.chroma.skills_collection)

    env = CrafterEnv(seed=args.seed, **settings.environment.crafter_kwargs)
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

    run_metadata = {
        "mode": args.mode,
        "env_seed": args.seed,
        "skill_library": settings.chroma.skills_collection,
        "eval_id": args.eval_id,
        "eval_run_idx": args.eval_run_idx,
    }

    with RunLogger(
        config_snapshot=settings.snapshot(),
        run_metadata=run_metadata,
    ) as run_log:
        log.info(
            "running episodes: count=%d early_stop_patience=%d session=%s",
            episodes,
            early_stop_patience,
            run_log.session_id,
        )
        if args.render:
            log.info(
                "render enabled: size=%d step_delay=%.3f",
                args.render_size,
                args.render_step_delay,
            )
        memory = SpatialMemory()
        if args.mode == "train":
            strategy = TrainingStrategy(
                codegen=CodeGen(),
                bug_fixer=FixBug(),
                reuse_threshold=settings.embedding.similarity_reuse_threshold,
                reflection=Reflection() if settings.llm.reflection_enabled else None,
                reflection_enabled=settings.llm.reflection_enabled,
                max_reflections_per_skill=(
                    settings.llm.max_reflections_per_skill
                ),
                max_fix_attempts=settings.llm.max_fix_attempts,
                skill_validator=lambda code: load_skill(
                    code,
                    runtime=SkillRuntime(memory=memory),
                ),
            )
        else:
            strategy = InferenceStrategy(
                reuse_threshold=settings.embedding.similarity_reuse_threshold,
            )
        agent = Agent(
            env=env,
            curriculum=HardcodedCurriculum(),
            skill_manager=skill_manager,
            strategy=strategy,
            executor=executor,
            memory=memory,
            max_iterations_per_episode=(
                settings.executor.max_iterations_per_episode
            ),
            run_logger=run_log,
        )
        summary = None
        best_achievement_count = -1
        episodes_without_progress = 0
        for episode_num in range(1, episodes + 1):
            log.info("episode %d/%d started", episode_num, episodes)
            summary = agent.run(episode_num=episode_num)
            achievements = summary["final_state"]["info"].get("achievements", {})
            achievement_count = sum(1 for value in achievements.values() if value)
            if achievement_count > best_achievement_count:
                best_achievement_count = achievement_count
                episodes_without_progress = 0
            else:
                episodes_without_progress += 1
            log.info(
                "episode %d/%d finished: achievements=%d steps=%d reward=%.3f",
                episode_num,
                episodes,
                achievement_count,
                summary["steps"],
                summary["total_reward"],
            )
            if (
                episodes_without_progress >= early_stop_patience
                and episode_num < episodes
            ):
                log.info(
                    "early stop: no new achievements for %d episodes",
                    early_stop_patience,
                )
                break

        if summary is None:
            raise RuntimeError("no episodes were run")
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
