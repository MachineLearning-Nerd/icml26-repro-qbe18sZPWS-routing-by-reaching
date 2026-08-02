#!/usr/bin/env python3
"""Publish an exact additive text allowlist to the existing qbe18sZPWS Space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


REPO_ID = "DineshAI/qbe18sZPWS"
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def read_allowlist(path: Path) -> list[str]:
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if entries != sorted(set(entries)):
        raise SystemExit("allowlist must be sorted and duplicate-free")
    for entry in entries:
        parsed = PurePosixPath(entry)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise SystemExit(f"unsafe allowlist path: {entry}")
    return entries


def gate(candidate: Path, baseline: Path, allowlist: list[str]) -> None:
    candidate_files = files(candidate)
    baseline_files = files(baseline)
    missing = sorted(set(baseline_files) - set(candidate_files))
    if missing:
        raise SystemExit(f"candidate removes baseline paths: {missing}")

    changed = sorted(
        relative
        for relative, path in candidate_files.items()
        if relative not in baseline_files
        or sha256(path) != sha256(baseline_files[relative])
    )
    if changed != allowlist:
        raise SystemExit("allowlist is not the exact candidate delta")

    for relative in allowlist:
        path = candidate_files[relative]
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise SystemExit(f"non-text suffix in allowlist: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SystemExit(f"non-UTF-8 allowlist path: {relative}") from exc
        if path.suffix == ".json":
            json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    allowlist = read_allowlist(args.allowlist)
    gate(args.candidate_dir, args.baseline_dir, allowlist)
    print(f"release gates passed: {len(allowlist)} exact UTF-8 text paths")
    for relative in allowlist:
        print(f"{sha256(args.candidate_dir / relative)}  {relative}")
    if args.dry_run:
        return

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import CommitOperationAdd, HfApi, get_token

    token = get_token()
    if not token:
        raise SystemExit("no cached Hugging Face token")
    api = HfApi()
    live_head = api.repo_info(REPO_ID, repo_type="space", token=token).sha
    if live_head != args.expected_head:
        raise SystemExit(
            f"Space HEAD changed: expected {args.expected_head}, found {live_head}"
        )
    operations = [
        CommitOperationAdd(
            path_in_repo=relative,
            path_or_fileobj=str(args.candidate_dir / relative),
        )
        for relative in allowlist
    ]
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        operations=operations,
        commit_message=args.message,
        parent_commit=live_head,
        token=token,
    )
    print(f"published revision: {commit.oid}")
    print(f"retained judged baseline: {live_head}")
    print("status: awaiting judge")


if __name__ == "__main__":
    main()
