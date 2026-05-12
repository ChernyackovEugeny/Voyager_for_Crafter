"""CLI tools for inspecting and managing Chroma skill libraries.

The agent stores generated skills in ChromaDB collections. One collection is
one skill library. By default the library name comes from
`CHROMA_SKILLS_COLLECTION` / `settings.chroma.skills_collection`, but every
command can override it with `--library`.

Common commands:

    # List all Chroma collections.
    python src\\tools\\skills_admin.py libraries

    # List skills in the default library.
    python src\\tools\\skills_admin.py list

    # List skills in a specific library.
    python src\\tools\\skills_admin.py --library skills_good list

    # Show metadata and source code for one skill.
    python src\\tools\\skills_admin.py --library skills show collect_wood

    # Delete one bad skill from a library.
    python src\\tools\\skills_admin.py --library skills delete collect_wood

    # Delete an entire library collection.
    python src\\tools\\skills_admin.py delete-library skills_dirty

    # Export a whole library, including embeddings, to JSON.
    python src\\tools\\skills_admin.py --library skills_good export --out skills_good.json

    # Import an exported JSON into another library. Existing skill names are
    # skipped unless --overwrite is passed.
    python src\\tools\\skills_admin.py --library skills_new import skills_good.json
    python src\\tools\\skills_admin.py --library skills_new import skills_good.json --overwrite

    # Copy skills directly between two Chroma collections. Existing target
    # names are skipped unless --overwrite is passed.
    python src\\tools\\skills_admin.py copy skills_good skills_experiment
    python src\\tools\\skills_admin.py copy skills_good skills_experiment --overwrite

    # Recompute embeddings for an existing library using the task text before
    # ". Uses:" in each description. Use this after changing embedding policy.
    python src\\tools\\skills_admin.py --library skills_good reembed

Typical workflows:

    # Start a clean experiment without touching the old default library.
    python src\\main.py --render --skill-library skills_experiment_001

    # Inspect what the agent learned.
    python src\\tools\\skills_admin.py --library skills_experiment_001 list
    python src\\tools\\skills_admin.py --library skills_experiment_001 show collect_wood

    # Save a snapshot before manual cleanup.
    python src\\tools\\skills_admin.py --library skills_experiment_001 export --out skills_experiment_001.json

    # Remove a bad skill and keep training in the same library.
    python src\\tools\\skills_admin.py --library skills_experiment_001 delete collect_wood
    python src\\main.py --render --skill-library skills_experiment_001

    # Promote a good experiment to a stable library.
    python src\\tools\\skills_admin.py copy skills_experiment_001 skills_good --overwrite

Notes:
    - Chroma data persists across runs in the Docker volume `chroma_data`.
    - `delete-library` is destructive for that collection only.
    - `export` is the safest way to snapshot a known-good library before more
      training or manual cleanup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ChromaConfig, get_settings
from skills.embedder import TextEmbedder
from storage.skill_repository import ChromaSkillRepository


def _repo(library: str | None) -> ChromaSkillRepository:
    settings = get_settings()
    cfg = ChromaConfig(
        host=settings.chroma.host,
        port=settings.chroma.port,
        skills_collection=library or settings.chroma.skills_collection,
    )
    return ChromaSkillRepository(cfg)


def _print_skill_summary(skill) -> None:
    print(
        f"{skill.name} | success={skill.success_count} "
        f"fail={skill.fail_count} score={skill.episodic_score:.3f}"
    )
    print(f"  {skill.description}")


def _cmd_libraries(args) -> int:
    repo = _repo(args.library)
    for name in repo.list_collections():
        print(name)
    return 0


def _cmd_list(args) -> int:
    repo = _repo(args.library)
    skills = repo.list_skills()
    print(f"library={repo._cfg.skills_collection} count={len(skills)}")
    for skill in skills:
        _print_skill_summary(skill)
    return 0


def _cmd_show(args) -> int:
    repo = _repo(args.library)
    skill = repo.get(args.name)
    if skill is None:
        print(f"skill not found: {args.name}", file=sys.stderr)
        return 1
    _print_skill_summary(skill)
    print()
    print(skill.code)
    return 0


def _cmd_delete(args) -> int:
    repo = _repo(args.library)
    deleted = repo.delete(args.name)
    if not deleted:
        print(f"skill not found: {args.name}", file=sys.stderr)
        return 1
    print(f"deleted {args.name} from {repo._cfg.skills_collection}")
    return 0


def _cmd_delete_library(args) -> int:
    repo = _repo(args.library)
    repo.delete_collection(args.library)
    print(f"deleted library {args.library}")
    return 0


def _cmd_export(args) -> int:
    repo = _repo(args.library)
    data = {
        "library": repo._cfg.skills_collection,
        "entries": repo.export_entries(),
    }
    out = Path(args.out)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(
        f"exported {len(data['entries'])} skills from "
        f"{repo._cfg.skills_collection} to {out}"
    )
    return 0


def _cmd_import(args) -> int:
    repo = _repo(args.library)
    src = Path(args.file)
    data = json.loads(src.read_text(encoding="utf-8"))
    inserted = repo.import_entries(
        data.get("entries", []),
        overwrite=args.overwrite,
    )
    print(f"imported {inserted} skills into {repo._cfg.skills_collection}")
    return 0


def _cmd_copy(args) -> int:
    source = _repo(args.source)
    target = _repo(args.target)
    inserted = target.import_entries(
        source.export_entries(),
        overwrite=args.overwrite,
    )
    print(f"copied {inserted} skills from {args.source} to {args.target}")
    return 0


def _embedding_text(skill) -> str:
    return skill.description.split(". Uses:", 1)[0].strip()


def _cmd_reembed(args) -> int:
    repo = _repo(args.library)
    settings = get_settings()
    model_name = args.model or settings.embedding.model_name
    embedder = TextEmbedder(model_name)
    skills = repo.list_skills()
    for skill in skills:
        repo.update_embedding(skill.name, embedder.encode(_embedding_text(skill)))
    print(
        f"reembedded {len(skills)} skills in {repo._cfg.skills_collection} "
        f"with {model_name}"
    )
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage Crafter skill libraries.")
    parser.add_argument(
        "--library",
        default=None,
        help="Skill library collection name. Defaults to CHROMA_SKILLS_COLLECTION.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("libraries", help="List Chroma collections.")
    p.set_defaults(func=_cmd_libraries)

    p = sub.add_parser("list", help="List skills in a library.")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("show", help="Show one skill and its code.")
    p.add_argument("name")
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("delete", help="Delete one skill from a library.")
    p.add_argument("name")
    p.set_defaults(func=_cmd_delete)

    p = sub.add_parser("delete-library", help="Delete a whole Chroma collection.")
    p.add_argument("library")
    p.set_defaults(func=_cmd_delete_library)

    p = sub.add_parser("export", help="Export a library to JSON.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("import", help="Import a library JSON export.")
    p.add_argument("file")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_import)

    p = sub.add_parser("copy", help="Copy skills between Chroma collections.")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=_cmd_copy)

    p = sub.add_parser("reembed", help="Recompute embeddings for one library.")
    p.add_argument("--model", default=None)
    p.set_defaults(func=_cmd_reembed)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
