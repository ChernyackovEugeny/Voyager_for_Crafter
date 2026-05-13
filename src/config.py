"""Typed configuration loaded from environment variables.

Each domain has its own pydantic-settings group with an env_prefix so that
.env keys are namespaced (LLM_*, POSTGRES_*, CHROMA_*, EMB_*, EXEC_*,
SURVIVAL_*, ENV_*).
Single Settings aggregate is exposed via get_settings(), memoised.

Bad value or missing required field → pydantic raises ValidationError at
import time, which is the right place to fail fast — the agent should not
boot at all if config is broken.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Shared base: read .env once, ignore unknown keys (so adding new vars to
# .env never breaks unrelated groups).
class _BaseGroup(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ---------------------------------------------------------------------------
# LLM (DeepSeek via OpenAI-compatible API)
# ---------------------------------------------------------------------------

class LLMConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="LLM_", extra="ignore",
    )

    deepseek_api_key: str = Field(min_length=1)
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    codegen_model: str = "deepseek-chat"
    reflection_model: str = "deepseek-reasoner"
    curriculum_model: str = "deepseek-chat"
    codegen_temperature: float = 0.0
    reflection_temperature: float = 0.1
    curriculum_temperature: float = 0.2
    reflection_timeout_s: float = 180.0
    reflection_enabled: bool = True
    curriculum_llm_enabled: bool = False
    curriculum_timeout_s: float = 30.0
    curriculum_max_failures_in_context: int = 5
    curriculum_max_retries: int = 1
    max_reflections_per_skill: int = 3
    max_fix_attempts: int = 3
    request_timeout_s: float = 60.0


# ---------------------------------------------------------------------------
# Postgres — analytics DB
# ---------------------------------------------------------------------------

class PostgresConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="POSTGRES_", extra="ignore",
    )

    host: str = "localhost"
    port: int = 5433
    db: str = "crafter_analytics"
    user: str = "crafter"
    password: str = "changeme"

    @property
    def connection_params(self) -> dict:
        """Plain dict for psycopg2.connect(**...)."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db,
            "user": self.user,
            "password": self.password,
        }


# ---------------------------------------------------------------------------
# ChromaDB — vector store for skills
# ---------------------------------------------------------------------------

class ChromaConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="CHROMA_", extra="ignore",
    )

    host: str = "localhost"
    port: int = 8000
    skills_collection: str = "skills"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class EmbeddingConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="EMB_", extra="ignore",
    )

    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    # Invariant: reuse_threshold <= dedup_threshold.
    # Otherwise there's a dead zone where a retrieved skill is too dissimilar to
    # reuse but similar enough that the freshly generated replacement is
    # rejected as a duplicate — codegen runs every task and nothing is learned.
    similarity_reuse_threshold: float = 0.65
    similarity_dedup_threshold: float = 0.70


# ---------------------------------------------------------------------------
# Executor — runtime guards while running a skill
# ---------------------------------------------------------------------------

class ExecutorConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="EXEC_", extra="ignore",
    )

    max_steps_per_skill: int = 200
    max_iterations_per_episode: int = 50
    health_interrupt_threshold: int = 4
    min_steps_before_health_interrupt: int = 5
    stagnation_window: int = 40
    min_steps_before_stagnation_interrupt: int = 20
    max_consecutive_task_failures: int = 3
    min_reflection_steps: int = 3
    min_steps_before_danger_interrupt: int = 5


# ---------------------------------------------------------------------------
# Survival task routing
# ---------------------------------------------------------------------------

class SurvivalConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="SURVIVAL_", extra="ignore",
    )

    enter_health: int = 7
    enter_food: int = 6
    enter_drink: int = 6
    exit_health: int = 8
    exit_food: int = 6
    exit_drink: int = 6
    max_consecutive_failures: int = 3


# ---------------------------------------------------------------------------
# Environment (Crafter)
# ---------------------------------------------------------------------------

class EnvironmentConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="ENV_", extra="ignore",
    )

    view_width: int = 9
    view_height: int = 7
    world_size: int = 64

    @property
    def crafter_kwargs(self) -> dict:
        """Keyword args for crafter.Env via environment.wrapper.CrafterEnv."""
        return {
            "area": (self.world_size, self.world_size),
            "view": (self.view_width, self.view_height),
        }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LoggingConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="LOG_", extra="ignore",
    )

    level: str = "INFO"


# ---------------------------------------------------------------------------
# Analytics / training loop
# ---------------------------------------------------------------------------

class AnalyticsConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="ANALYTICS_", extra="ignore",
    )

    episodes: int = 1
    early_stop_patience: int = 10


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class EvalConfig(_BaseGroup):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="EVAL_", extra="ignore",
    )

    n_training_runs: int = 10
    n_inference_seeds: int = 50
    train_episodes: int = 100
    inference_episodes: int = 1
    last_k_training_episodes: int = 20
    base_seed: int = 1000
    inference_seed_base: int = 100_000
    report_dir: str = "eval_reports"


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

class Settings:
    """Aggregate of all configuration groups. Use get_settings()."""

    def __init__(self) -> None:
        self.llm = LLMConfig()
        self.postgres = PostgresConfig()
        self.chroma = ChromaConfig()
        self.embedding = EmbeddingConfig()
        self.executor = ExecutorConfig()
        self.survival = SurvivalConfig()
        self.environment = EnvironmentConfig()
        self.logging = LoggingConfig()
        self.analytics = AnalyticsConfig()
        self.eval = EvalConfig()

    def snapshot(self) -> dict:
        """Serializable config dump for analytics (config_snapshot column)."""
        return {
            "llm": self.llm.model_dump(exclude={"deepseek_api_key"}),
            "postgres": self.postgres.model_dump(exclude={"password"}),
            "chroma": self.chroma.model_dump(),
            "embedding": self.embedding.model_dump(),
            "executor": self.executor.model_dump(),
            "survival": self.survival.model_dump(),
            "environment": self.environment.model_dump(),
            "logging": self.logging.model_dump(),
            "analytics": self.analytics.model_dump(),
            "eval": self.eval.model_dump(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
