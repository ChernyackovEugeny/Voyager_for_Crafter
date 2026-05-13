"""Sequential evaluation runner for Voyager-Crafter.

What this module does
=====================

This is the final benchmark harness for the project. It repeatedly launches the
normal application entry point, ``src/main.py``, instead of duplicating agent
wiring here. That means evaluation uses exactly the same Agent, Strategy,
Executor, SkillManager, Chroma repository, and analytics path as normal runs.

For each training run, the runner:

1. Creates a deterministic Chroma collection name:
   ``<eval_id>_run_<run_idx>``.
2. Resets that collection before training unless ``--resume`` is skipping an
   already completed training session.
3. Runs ``src/main.py --mode train`` with a fixed environment seed.
4. Runs ``src/main.py --mode inference`` on fresh environment seeds using the
   frozen collection learned by that training run.
5. Builds a JSON and Markdown report from Postgres analytics rows.

Postgres ``episodes`` rows are the source of truth for scores. ``sessions`` rows
only group episodes and carry metadata such as ``eval_id``, ``eval_run_idx``,
``mode``, ``env_seed``, and ``skill_library``.

CLI parameters
==============

``--eval-id TEXT``
    Stable identifier for the whole evaluation. It is stored in
    ``sessions.run_metadata`` and used as the report filename. If omitted, a UTC
    timestamp ID such as ``eval_20260513_142301`` is generated.

``--n-training-runs N``
    Number of independent agents to train from scratch. Default comes from
    ``EVAL_N_TRAINING_RUNS`` or ``EvalConfig.n_training_runs``. The intended
    default is 10, matching common Crafter reporting practice.

``--n-inference-seeds N``
    Number of fresh inference sessions per trained library. Inference reuses the
    frozen skill library and should not call the LLM through TrainingStrategy.

``--train-episodes N``
    Number of training episodes per training run. Early stopping inside
    ``src/main.py`` can still stop earlier if configured.

``--inference-episodes N``
    Number of episodes per inference seed session. Usually 1, because each
    inference session already gets a fresh seed.

``--last-k-training-episodes N``
    Window used for the training-time score in the final report. This is an
    internal steady-state metric, not the official 1M-step Crafter score.

``--base-seed N``
    Training seed base. Training run ``i`` uses ``base_seed + i``.

``--inference-seed-base N``
    Inference seed base. For training run ``i`` and inference offset ``j``, the
    seed is ``inference_seed_base + i * n_inference_seeds + j``.

``--report-dir PATH``
    Directory for ``<eval_id>.json`` and ``<eval_id>.md`` reports.

``--resume``
    Skip already completed sessions for the same ``eval_id``. Completed
    training sessions are reused together with their Chroma collections.
    Completed inference sessions are skipped per seed. Interrupted sessions are
    deliberately not reused for scoring.

``--dry-run``
    Override run counts to 2 training runs and 2 inference seeds. This is for
    plumbing checks before spending real LLM budget. Episode counts are not
    changed by ``--dry-run``; pass small ``--train-episodes`` explicitly.

Interrupt and resume semantics
==============================

Each launched ``src/main.py`` process creates one Postgres ``sessions`` row with
``status='running'``. On normal exit, ``RunLogger.__exit__`` marks it
``completed``. On exception or Ctrl+C while the context manager unwinds, it is
marked ``interrupted``. If the Python process is killed hard enough that
``__exit__`` cannot run, the row may remain ``running``.

Episode rows are inserted with autocommit after each completed episode. So if an
evaluation is interrupted:

* completed episodes already written to Postgres remain available for debugging;
* the currently running unfinished episode is not written;
* interrupted/running sessions are excluded from final reports;
* ``--resume`` starts a replacement session for anything not completed.

This means interrupted sessions are not "bad data" for analysis, but they are
not benchmark data. They are intentionally ignored by report generation.

Usage examples
==============

Cheap plumbing check:

    python -B src/eval/evaluate.py --dry-run --train-episodes 2 --inference-episodes 1

Resume that same evaluation if it was interrupted:

    python -B src/eval/evaluate.py --eval-id eval_20260513_142301 --dry-run \\
        --train-episodes 2 --inference-episodes 1 --resume

Full sequential evaluation:

    python -B src/eval/evaluate.py --eval-id final_eval_001 --n-training-runs 10 \\
        --n-inference-seeds 50 --train-episodes 100 --inference-episodes 1

Rebuild reports and fill missing completed sessions:

    python -B src/eval/evaluate.py --eval-id final_eval_001 --n-training-runs 10 \\
        --n-inference-seeds 50 --train-episodes 100 --inference-episodes 1 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings, get_settings
from eval.metrics import achievement_success_rates, crafter_score
from observability.logging_config import configure_logging
from storage.skill_repository import ChromaSkillRepository


@dataclass(frozen=True)
class EvalArgs:
    eval_id: str
    n_training_runs: int
    n_inference_seeds: int
    train_episodes: int
    inference_episodes: int
    last_k_training_episodes: int
    base_seed: int
    inference_seed_base: int
    report_dir: Path
    resume: bool
    dry_run: bool


def _parse_args(argv: list[str] | None = None) -> EvalArgs:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run Voyager-Crafter evaluation.")
    parser.add_argument("--eval-id", default=None)
    parser.add_argument("--n-training-runs", type=int, default=settings.eval.n_training_runs)
    parser.add_argument("--n-inference-seeds", type=int, default=settings.eval.n_inference_seeds)
    parser.add_argument("--train-episodes", type=int, default=settings.eval.train_episodes)
    parser.add_argument("--inference-episodes", type=int, default=settings.eval.inference_episodes)
    parser.add_argument(
        "--last-k-training-episodes",
        type=int,
        default=settings.eval.last_k_training_episodes,
    )
    parser.add_argument("--base-seed", type=int, default=settings.eval.base_seed)
    parser.add_argument(
        "--inference-seed-base",
        type=int,
        default=settings.eval.inference_seed_base,
    )
    parser.add_argument("--report-dir", default=settings.eval.report_dir)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use 2 training runs and 2 inference seeds for plumbing checks.",
    )
    ns = parser.parse_args(argv)
    eval_id = ns.eval_id or datetime.now(timezone.utc).strftime("eval_%Y%m%d_%H%M%S")
    n_training_runs = 2 if ns.dry_run else ns.n_training_runs
    n_inference_seeds = 2 if ns.dry_run else ns.n_inference_seeds
    return EvalArgs(
        eval_id=eval_id,
        n_training_runs=n_training_runs,
        n_inference_seeds=n_inference_seeds,
        train_episodes=ns.train_episodes,
        inference_episodes=ns.inference_episodes,
        last_k_training_episodes=ns.last_k_training_episodes,
        base_seed=ns.base_seed,
        inference_seed_base=ns.inference_seed_base,
        report_dir=Path(ns.report_dir),
        resume=ns.resume,
        dry_run=ns.dry_run,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    settings = get_settings()
    configure_logging(settings.logging.level)

    for run_idx in range(args.n_training_runs):
        collection = _collection_name(args.eval_id, run_idx)
        if args.resume and _has_completed_session(settings, args.eval_id, run_idx, "train"):
            continue
        _reset_skill_collection(settings, collection)
        _run_agent_session(
            args=args,
            run_idx=run_idx,
            mode="train",
            collection=collection,
            seed=args.base_seed + run_idx,
            episodes=args.train_episodes,
        )

        for seed_offset in range(args.n_inference_seeds):
            inf_seed = args.inference_seed_base + run_idx * args.n_inference_seeds + seed_offset
            if args.resume and _has_completed_inference_seed(
                settings,
                args.eval_id,
                run_idx,
                inf_seed,
            ):
                continue
            _run_agent_session(
                args=args,
                run_idx=run_idx,
                mode="inference",
                collection=collection,
                seed=inf_seed,
                episodes=args.inference_episodes,
            )

    report = _build_report(settings, args)
    _write_report(args.report_dir, args.eval_id, report)


def _run_agent_session(
    *,
    args: EvalArgs,
    run_idx: int,
    mode: str,
    collection: str,
    seed: int,
    episodes: int,
) -> None:
    cmd = [
        sys.executable,
        "src/main.py",
        "--mode",
        mode,
        "--skill-library",
        collection,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--eval-id",
        args.eval_id,
        "--eval-run-idx",
        str(run_idx),
    ]
    subprocess.run(cmd, cwd=_repo_root(), check=True)


def _reset_skill_collection(settings: Settings, collection: str) -> None:
    settings.chroma.skills_collection = collection
    ChromaSkillRepository(settings.chroma).reset_collection()


def _collection_name(eval_id: str, run_idx: int) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in eval_id)
    return f"{safe}_run_{run_idx}"


def _connect(settings: Settings):
    return psycopg2.connect(**settings.postgres.connection_params)


def _has_completed_session(
    settings: Settings,
    eval_id: str,
    run_idx: int,
    mode: str,
) -> bool:
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM sessions
                WHERE status = 'completed'
                  AND run_metadata @> %s::jsonb
                LIMIT 1
                """,
                (json.dumps({
                    "eval_id": eval_id,
                    "eval_run_idx": run_idx,
                    "mode": mode,
                }),),
            )
            return cur.fetchone() is not None


def _has_completed_inference_seed(
    settings: Settings,
    eval_id: str,
    run_idx: int,
    seed: int,
) -> bool:
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM sessions
                WHERE status = 'completed'
                  AND run_metadata @> %s::jsonb
                LIMIT 1
                """,
                (json.dumps({
                    "eval_id": eval_id,
                    "eval_run_idx": run_idx,
                    "mode": "inference",
                    "env_seed": seed,
                }),),
            )
            return cur.fetchone() is not None


def _build_report(settings: Settings, args: EvalArgs) -> dict[str, Any]:
    rows = _fetch_episode_rows(settings, args.eval_id)
    costs = _fetch_cost_by_run(settings, args.eval_id)
    per_run = []
    for run_idx in range(args.n_training_runs):
        training = [
            row["achievements"]
            for row in rows
            if row["run_idx"] == run_idx and row["mode"] == "train"
        ]
        inference = [
            row["achievements"]
            for row in rows
            if row["run_idx"] == run_idx and row["mode"] == "inference"
        ]
        training_window = training[-args.last_k_training_episodes:]
        skills_learned = _skill_count(settings, _collection_name(args.eval_id, run_idx))
        per_run.append({
            "run_idx": run_idx,
            "skills_learned": skills_learned,
            "training_episodes": len(training),
            "inference_episodes": len(inference),
            "cost_usd": costs.get(run_idx, 0.0),
            "training_score": crafter_score(training_window),
            "inference_score": crafter_score(inference),
            "training_success_rates": achievement_success_rates(training_window),
            "inference_success_rates": achievement_success_rates(inference),
        })

    return {
        "eval_id": args.eval_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_training_runs": args.n_training_runs,
            "n_inference_seeds": args.n_inference_seeds,
            "train_episodes": args.train_episodes,
            "inference_episodes": args.inference_episodes,
            "last_k_training_episodes": args.last_k_training_episodes,
            "base_seed": args.base_seed,
            "inference_seed_base": args.inference_seed_base,
            "dry_run": args.dry_run,
        },
        "per_run": per_run,
        "aggregate": {
            "training_score": _mean_std([row["training_score"] for row in per_run]),
            "inference_score": _mean_std([row["inference_score"] for row in per_run]),
            "cost_usd": sum(row["cost_usd"] for row in per_run),
        },
    }


def _fetch_episode_rows(settings: Settings, eval_id: str) -> list[dict[str, Any]]:
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.run_metadata,
                    e.episode_num,
                    e.achievements
                FROM episodes e
                JOIN sessions s ON s.session_id = e.session_id
                WHERE s.run_metadata @> %s::jsonb
                  AND s.status = 'completed'
                ORDER BY s.started_at, e.episode_num
                """,
                (json.dumps({"eval_id": eval_id}),),
            )
            return [
                {
                    "run_idx": metadata.get("eval_run_idx"),
                    "mode": metadata.get("mode"),
                    "episode_num": episode_num,
                    "achievements": achievements,
                }
                for metadata, episode_num, achievements in cur.fetchall()
            ]


def _fetch_cost_by_run(settings: Settings, eval_id: str) -> dict[int, float]:
    with _connect(settings) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    s.run_metadata->>'eval_run_idx' AS run_idx,
                    COALESCE(SUM(l.cost_usd), 0.0) AS cost_usd
                FROM llm_calls l
                JOIN sessions s ON s.session_id = l.session_id
                WHERE s.run_metadata @> %s::jsonb
                  AND s.status = 'completed'
                GROUP BY s.run_metadata->>'eval_run_idx'
                """,
                (json.dumps({"eval_id": eval_id}),),
            )
            return {
                int(run_idx): float(cost_usd)
                for run_idx, cost_usd in cur.fetchall()
                if run_idx is not None
            }


def _skill_count(settings: Settings, collection: str) -> int:
    old_collection = settings.chroma.skills_collection
    try:
        settings.chroma.skills_collection = collection
        return ChromaSkillRepository(settings.chroma).count()
    except Exception:
        return 0
    finally:
        settings.chroma.skills_collection = old_collection


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}


def _write_report(report_dir: Path, eval_id: str, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{eval_id}.json"
    md_path = report_dir / f"{eval_id}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")


def _markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        f"# Eval Report - {report['eval_id']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Mean | Std |",
        "|---|---:|---:|",
        (
            "| Training score | "
            f"{aggregate['training_score']['mean']:.4f} | "
            f"{aggregate['training_score']['std']:.4f} |"
        ),
        (
            "| Inference score | "
            f"{aggregate['inference_score']['mean']:.4f} | "
            f"{aggregate['inference_score']['std']:.4f} |"
        ),
        "",
        f"Total LLM cost: ${aggregate['cost_usd']:.4f}",
        "",
        "## Per Run",
        "",
        "| Run | Skills | Train eps | Infer eps | Cost | Training score | Inference score |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_run"]:
        lines.append(
            f"| {row['run_idx']} | {row['skills_learned']} | "
            f"{row['training_episodes']} | {row['inference_episodes']} | "
            f"${row['cost_usd']:.4f} | "
            f"{row['training_score']:.4f} | {row['inference_score']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
