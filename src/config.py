"""Typed configuration loaded from environment variables.

Each domain has its own pydantic-settings group with an env_prefix so that
.env keys are namespaced (LLM_*, POSTGRES_*, CHROMA_*, EMB_*, EXEC_*, ENV_*).
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
    similarity_reuse_threshold: float = 0.85
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
    stagnation_window: int = 15


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
        self.environment = EnvironmentConfig()
        self.logging = LoggingConfig()
        self.analytics = AnalyticsConfig()

    def snapshot(self) -> dict:
        """Serializable config dump for analytics (config_snapshot column)."""
        return {
            "llm": self.llm.model_dump(exclude={"deepseek_api_key"}),
            "postgres": self.postgres.model_dump(exclude={"password"}),
            "chroma": self.chroma.model_dump(),
            "embedding": self.embedding.model_dump(),
            "executor": self.executor.model_dump(),
            "environment": self.environment.model_dump(),
            "logging": self.logging.model_dump(),
            "analytics": self.analytics.model_dump(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
