"""Structured analytics writer for Crafter training/inference runs.

Mirrors the Banking project's RunLogger pattern: a context manager that owns
a single Postgres connection, creates its schema on first use (IF NOT EXISTS),
and degrades to a no-op if the DB is unreachable so the agent keeps running.

Three tables:
  - sessions     one row per run() invocation
  - episodes     one row per episode within a session
  - llm_calls    one row per LLM API call (codegen/fix_bug/curriculum/reflection)

Usage:
    with RunLogger(config_snapshot={...}) as run_log:
        run_log.record_episode(...)
        run_log.record_llm_call(...)
        run_log.finalize(final_achievements={...})

If `run_log._disabled` is True (Postgres unreachable), every method is a
silent no-op. Callers should always use log_utils.* wrappers, never call
RunLogger methods directly — wrappers also handle `run_log is None`.
"""
from __future__ import annotations

import json
import logging
import platform
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg2
import psycopg2.extensions

from config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDL — owned by this module. Whoever runs first creates the schema.
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    duration_sec        DOUBLE PRECISION,
    status              TEXT NOT NULL DEFAULT 'running',
    run_metadata        JSONB,
    config_snapshot     JSONB,
    final_achievements  JSONB,
    python_version      TEXT,
    hostname            TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_started
    ON sessions (started_at DESC);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    episode_num     INTEGER NOT NULL,
    achievements    JSONB NOT NULL,
    steps           INTEGER NOT NULL,
    total_reward    DOUBLE PRECISION NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL,
    ended_at        TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_session
    ON episodes (session_id, episode_num);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id             TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    episode_id          TEXT REFERENCES episodes(episode_id) ON DELETE SET NULL,
    episode_num         INTEGER,
    call_type           TEXT NOT NULL,
    model               TEXT,
    prompt_template_id  TEXT,
    prompt_hash         TEXT,
    tokens_in           INTEGER NOT NULL DEFAULT 0,
    tokens_out          INTEGER NOT NULL DEFAULT 0,
    prompt_cache_hit_tokens  INTEGER NOT NULL DEFAULT 0,
    prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens         INTEGER,
    cost_usd            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    latency_ms          INTEGER NOT NULL DEFAULT 0,
    error               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_session
    ON llm_calls (session_id, call_type);
CREATE INDEX IF NOT EXISTS idx_llm_calls_hash
    ON llm_calls (prompt_hash);
"""

_MIGRATIONS = """
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS model TEXT;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS episode_num INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS prompt_cache_hit_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS reasoning_tokens INTEGER;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'running';
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS run_metadata JSONB;
"""


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# RunLogger
# ---------------------------------------------------------------------------

class RunLogger:
    """Postgres-backed analytics writer for one agent run.

    Disabled-mode: if connect() fails in __enter__, sets _disabled=True and
    every recording method becomes a no-op. The agent keeps running.
    """

    def __init__(
        self,
        config_snapshot: dict[str, Any] | None = None,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id: str = _uuid()
        self._config_snapshot = config_snapshot or {}
        self._run_metadata = run_metadata or {}
        self._conn: psycopg2.extensions.connection | None = None
        self._disabled: bool = False
        self._session_started_at: datetime | None = None
        self._final_achievements: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RunLogger":
        params = get_settings().postgres.connection_params
        try:
            self._conn = psycopg2.connect(**params)
            self._conn.autocommit = True
            self._init_schema()
            self._start_session()
            logger.info(
                "RunLogger: session %s started (host=%s db=%s)",
                self.session_id, params["host"], params["dbname"],
            )
        except Exception as exc:
            logger.warning(
                "RunLogger: Postgres analytics disabled (%s)", exc,
            )
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
            self._disabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._disabled or self._conn is None:
            return False
        try:
            status = "completed" if exc_type is None else "interrupted"
            self._end_session(status=status)
        except Exception as exc:
            logger.warning("RunLogger: failed to finalize session: %s", exc)
        finally:
            try:
                self._conn.close()
            finally:
                self._conn = None
        return False  # never suppress caller exceptions

    # ------------------------------------------------------------------
    # Public API — all methods are no-ops when _disabled
    # ------------------------------------------------------------------

    def record_episode(
        self,
        *,
        episode_num: int,
        achievements: dict,
        steps: int,
        total_reward: float,
        started_at: datetime,
        ended_at: datetime,
    ) -> str | None:
        """Insert one episode row. Returns episode_id, or None if disabled."""
        if self._disabled:
            return None
        episode_id = _uuid()
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO episodes
                        (episode_id, session_id, episode_num, achievements,
                         steps, total_reward, started_at, ended_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        episode_id, self.session_id, episode_num,
                        json.dumps(achievements), steps, total_reward,
                        started_at, ended_at,
                    ),
                )
            return episode_id
        except Exception as exc:
            logger.debug("RunLogger.record_episode failed: %s", exc)
            return None

    def record_llm_call(
        self,
        *,
        call_type: str,
        episode_id: str | None = None,
        episode_num: int | None = None,
        prompt_template_id: str | None = None,
        prompt_hash: str | None = None,
        model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        prompt_cache_hit_tokens: int = 0,
        prompt_cache_miss_tokens: int = 0,
        reasoning_tokens: int | None = None,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        error: str | None = None,
    ) -> None:
        if self._disabled:
            return
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_calls
                        (call_id, session_id, episode_id, episode_num, call_type,
                         model, prompt_template_id, prompt_hash,
                         tokens_in, tokens_out,
                         prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                         reasoning_tokens, cost_usd, latency_ms, error)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        _uuid(), self.session_id, episode_id, episode_num, call_type,
                        model, prompt_template_id, prompt_hash,
                        tokens_in, tokens_out,
                        prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                        reasoning_tokens, cost_usd, latency_ms, error,
                    ),
                )
        except Exception as exc:
            logger.debug("RunLogger.record_llm_call failed: %s", exc)

    def finalize(self, final_achievements: dict) -> None:
        """Stash final achievements; written to DB by __exit__."""
        self._final_achievements = final_achievements

    # ------------------------------------------------------------------
    # Private — schema + session lifecycle
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_DDL)
            cur.execute(_MIGRATIONS)

    def _start_session(self) -> None:
        self._session_started_at = _now()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions
                    (session_id, started_at, status, run_metadata, config_snapshot,
                     python_version, hostname)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    self.session_id,
                    self._session_started_at,
                    "running",
                    json.dumps(self._run_metadata),
                    json.dumps(self._config_snapshot),
                    sys.version.split()[0],
                    platform.node(),
                ),
            )

    def _end_session(self, *, status: str) -> None:
        finished_at = _now()
        duration = (
            (finished_at - self._session_started_at).total_seconds()
            if self._session_started_at else None
        )
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET finished_at = %s,
                    duration_sec = %s,
                    status = %s,
                    final_achievements = %s
                WHERE session_id = %s
                """,
                (
                    finished_at, duration, status,
                    json.dumps(self._final_achievements) if self._final_achievements else None,
                    self.session_id,
                ),
            )
