#!/usr/bin/env python3
"""Build and validate an additive, text-only Hugging Face Space candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


EXPECTED_RAW_SHA256 = (
    "d9eb6771f79fb6ac03381c268712710c832342c161006913b2d81bd7a7b0afec"
)
SECRET_PATTERNS = {
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "github_token": re.compile(r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path) -> dict[str, Path]:
    return {
        str(path.relative_to(root)): path
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def write_manifest(path: Path, entries: dict[str, Path]) -> None:
    path.write_text(
        "".join(f"{sha256(source)}  {relative}\n" for relative, source in entries.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.candidate.exists() or args.release_dir.exists():
        raise SystemExit("candidate and release-dir must not already exist")
    args.candidate.parent.mkdir(parents=True, exist_ok=True)
    args.release_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.protected, args.candidate)
    shutil.copytree(args.overlay, args.candidate, dirs_exist_ok=True)
    args.release_dir.mkdir(parents=True)

    protected_files = files(args.protected)
    candidate_files = files(args.candidate)
    missing_old = sorted(set(protected_files) - set(candidate_files))
    changed_old = sorted(
        relative
        for relative in protected_files
        if relative in candidate_files
        and sha256(protected_files[relative]) != sha256(candidate_files[relative])
        and relative not in {"logbook.json", "pages/index.md"}
    )
    approved_changed_old = sorted(
        relative
        for relative in ("logbook.json", "pages/index.md")
        if relative in protected_files
        and sha256(protected_files[relative]) != sha256(candidate_files[relative])
    )
    protected_snapshot_root = (
        args.candidate
        / "protected"
        / "judged-61f1a9cbf8bdb3b3b398f9f552e514c27ae59860"
    )
    changed_index_snapshots = {
        relative: (
            protected_snapshot_root / relative
        ).is_file()
        and sha256(protected_snapshot_root / relative)
        == sha256(protected_files[relative])
        for relative in approved_changed_old
    }

    subset = {
        "protected_revision": "61f1a9cbf8bdb3b3b398f9f552e514c27ae59860",
        "protected_file_count": len(protected_files),
        "candidate_file_count_before_release_metadata": len(candidate_files),
        "missing_old_paths": missing_old,
        "unexpectedly_changed_old_paths": changed_old,
        "approved_additive_index_changes": approved_changed_old,
        "changed_index_byte_identical_snapshots": changed_index_snapshots,
        "old_non_index_file_set_is_byte_identical_subset": (
            not missing_old and not changed_old
        ),
        "entire_old_tree_is_reconstructable": (
            not missing_old
            and not changed_old
            and all(changed_index_snapshots.values())
        ),
    }
    release_in_candidate = args.candidate / "release"
    release_in_candidate.mkdir()
    (release_in_candidate / "SUBSET_CHECK.json").write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n"
    )

    candidate_files = files(args.candidate)
    changed_or_new = {
        relative: path
        for relative, path in candidate_files.items()
        if relative not in protected_files
        or sha256(path) != sha256(protected_files[relative])
    }
    non_text = []
    secret_hits = []
    json_errors = []
    for relative, path in changed_or_new.items():
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            non_text.append(relative)
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secret_hits.append({"path": relative, "pattern": name})
        if path.suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                json_errors.append({"path": relative, "error": str(exc)})

    logbook = json.loads((args.candidate / "logbook.json").read_text())
    referenced_pages = []

    def visit(node: dict) -> None:
        referenced_pages.append(node["file"])
        for child in node.get("children", []):
            visit(child)

    visit(logbook["root"])
    missing_pages = sorted(
        page for page in referenced_pages if not (args.candidate / page).is_file()
    )
    raw_sha = sha256(args.candidate / "evidence" / "full_grid_raw.json")
    checks = {
        "protected_non_index_files_preserved": (
            not missing_old and not changed_old
        ),
        "changed_index_snapshots_preserved": all(
            changed_index_snapshots.values()
        ),
        "only_text_files_in_upload_allowlist": not non_text,
        "all_candidate_json_valid": not json_errors,
        "all_logbook_pages_exist": not missing_pages,
        "space_id_exact": logbook["space_id"] == "DineshAI/qbe18sZPWS",
        "full_grid_raw_sha256": raw_sha == EXPECTED_RAW_SHA256,
        "no_secret_patterns": not secret_hits,
    }
    validation = {
        "passed": all(checks.values()),
        "checks": checks,
        "non_text_changed_or_new": non_text,
        "json_errors": json_errors,
        "missing_logbook_pages": missing_pages,
        "secret_pattern_hits": secret_hits,
        "upload_file_count": len(changed_or_new),
        "full_grid_raw_sha256": raw_sha,
    }
    (args.release_dir / "SUBSET_CHECK.json").write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n"
    )
    (args.release_dir / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    (args.release_dir / "UPLOAD_ALLOWLIST.txt").write_text(
        "".join(f"{relative}\n" for relative in changed_or_new)
    )
    write_manifest(args.release_dir / "UPLOAD_MANIFEST.sha256", changed_or_new)
    write_manifest(
        args.release_dir / "CANDIDATE_TREE_MANIFEST.sha256",
        candidate_files,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
