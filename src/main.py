"""Entry point for smoke-testing the infrastructure layer.

At this stage main() does:
  1. configure logging
  2. load settings
  3. open a RunLogger context (writes a session row to Postgres)
  4. run the existing noop agent for one episode
  5. finalize the session

If Postgres is unreachable RunLogger drops to disabled mode and the agent
still runs — no analytics rows are written, only a warning is logged.
ChromaDB is not exercised yet; it will be wired in when SkillManager lands.
"""
from __future__ import annotations

import logging

from agent.agent import Agent
from analytics.log_utils import log_session_finalize
from analytics.run_logger import RunLogger
from config import get_settings
from environment.wrapper import CrafterEnv
from observability.logging_config import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings.logging.level)

    log = logging.getLogger("main")
    log.info("voyager-crafter starting up")

    env = CrafterEnv(**settings.environment.crafter_kwargs)
    agent = Agent(env)

    with RunLogger(config_snapshot=settings.snapshot()) as run_log:
        log.info("running one episode (session=%s)", run_log.session_id)
        agent.run()
        # No achievement tracking yet — agent.run is still a noop. Pass empty.
        log_session_finalize(run_log, final_achievements={})

    env.close()
    log.info("voyager-crafter exited cleanly")


if __name__ == "__main__":
    main()
