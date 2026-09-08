#!/usr/bin/env python3
"""Validate the vLLM reader book structure without external dependencies."""

from __future__ import annotations

import json
import ast
from functools import lru_cache
import re
import subprocess
import sys
from pathlib import Path

try:
    from .math_support import check_math
except ImportError:
    from math_support import check_math


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = (
    "01-transformer-inference",
    "02-vllm-architecture",
    "03-engine-core",
    "04-scheduler",
    "05-kv-cache",
    "06-paged-attention",
    "07-model-execution",
    "08-compile-cuda-graph",
    "09-distributed",
    "10-experiments",
)
REQUIRED_FRONTMATTER = {
    "title",
    "status",
    "source_repository",
    "source_path",
    "source_commit",
    "source_branch",
    "source_dirty",
    "verified_at",
    "audience",
    "pedagogy_reviewed_at",
    "scope",
    "prerequisites",
}
VALID_STATUSES = {"outline", "draft", "verified", "stale"}
SOURCE_MAP_PATH = ROOT / "meta" / "chapter-source-map.json"
IGNORED_MARKDOWN_DIRS = {"node_modules", "build", "dist", ".git", "__pycache__"}
EXPLICIT_REFERENCE_RE = re.compile(
    r"^\[(?:源码|测试|设计)\]\s+`([^`]+)`", re.MULTILINE
)
SOURCE_PATH_RE = re.compile(
    r"`((?:vllm|tests|docs|benchmarks|examples)/[^`\s:]+?\."
    r"(?:py|md|cu|cuh|cpp|h|yaml|yml|json))(?:[:]{1,2}[^`]*)?`"
)


@lru_cache(maxsize=256)
def python_definitions(source: str) -> set[tuple[str, str]]:
    definitions = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef):
            definitions.add(("class", node.name))
        elif isinstance(node, ast.AsyncFunctionDef):
            definitions.add(("async def", node.name))
            definitions.add(("def", node.name))
        elif isinstance(node, ast.FunctionDef):
            definitions.add(("def", node.name))
    return definitions


def source_symbol_exists(source: str, symbol: str, *, python: bool) -> bool:
    definition = re.fullmatch(r"(class|def|async def) ([A-Za-z_]\w*)", symbol)
    if python and definition:
        return definition.groups() in python_definitions(source)
    return symbol in source


def parse_frontmatter(path: Path, text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path}: missing opening frontmatter delimiter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: missing closing frontmatter delimiter") from exc

    result: dict[str, str] = {}
    for line in lines[1:end]:
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"')
    return result


def check_markdown(path: Path, text: str, errors: list[str]) -> None:
    if sum(line.startswith("```") for line in text.splitlines()) % 2:
        errors.append(f"{path}: unbalanced fenced code blocks")
    if any(line.rstrip() != line for line in text.splitlines()):
        errors.append(f"{path}: trailing whitespace")


def check_pedagogy(path: Path, text: str, errors: list[str]) -> None:
    if "## 如何阅读本章" not in text:
        errors.append(f"{path}: missing chapter-specific reading guidance")

    lines = text.splitlines()
    mermaid_count = sum(line.strip() == "```mermaid" for line in lines)
    guide_count = sum(line.startswith("> **读图方法：**") for line in lines)
    if guide_count != mermaid_count:
        errors.append(
            f"{path}: expected {mermaid_count} diagram guides, found {guide_count}"
        )

    in_fence = False
    in_mermaid = False
    for index, line in enumerate(lines):
        if line.startswith("```"):
            if line.strip() == "```mermaid" and not in_fence:
                in_mermaid = True
            elif in_mermaid and line.strip() == "```":
                cursor = index + 1
                while cursor < len(lines) and not lines[cursor].strip():
                    cursor += 1
                if cursor >= len(lines) or not lines[cursor].startswith(
                    "> **读图方法：**"
                ):
                    errors.append(
                        f"{path}:{index + 1}: Mermaid diagram lacks an immediate guide"
                    )
                in_mermaid = False
            in_fence = not in_fence
            continue

        if in_fence or not re.match(r"^#{2,3}\s+", line):
            continue
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            errors.append(f"{path}:{index + 1}: empty section")
            continue
        first = lines[cursor].strip()
        jumps_to_structure = (
            first.startswith(("```", "|", "- ", "* ", "+ ", "[源码]", "[测试]", "[设计]"))
            or bool(re.match(r"^\d+\.\s+", first))
            or bool(re.match(r"^#{2,4}\s+", first))
        )
        if jumps_to_structure:
            errors.append(
                f"{path}:{index + 1}: section starts with structure before explanation"
            )


def check_local_links(path: Path, text: str, errors: list[str]) -> None:
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean_target = target.split("#", 1)[0]
        resolved = (path.parent / clean_target).resolve()
        if not resolved.exists():
            errors.append(f"{path}: broken local link {target!r}")


def check_explicit_references(
    chapter_texts: dict[str, str],
    source_root: Path,
    errors: list[str],
    reader_root: Path = ROOT,
) -> int:
    """Validate paths named by explicit source/test/design citations."""
    reference_count = 0
    for chapter, text in chapter_texts.items():
        for relative_path in EXPLICIT_REFERENCE_RE.findall(text):
            reference_count += 1
            path = Path(relative_path)
            if path.is_absolute() or ".." in path.parts:
                errors.append(
                    f"book/{chapter}/README.md: unsafe explicit reference "
                    f"{relative_path!r}"
                )
                continue

            # Most citations point into the pinned vLLM checkout. A small number
            # intentionally cite reader-local regression tests.
            if not (source_root / path).is_file() and not (reader_root / path).is_file():
                errors.append(
                    f"book/{chapter}/README.md: missing explicit reference "
                    f"{relative_path}"
                )
    return reference_count


def check_source_path_references(
    chapter_texts: dict[str, str],
    source_root: Path,
    errors: list[str],
    reader_root: Path = ROOT,
) -> int:
    """Validate every source-like path embedded in an inline code span."""
    reference_count = 0
    for chapter, text in chapter_texts.items():
        for relative_path in SOURCE_PATH_RE.findall(text):
            reference_count += 1
            path = Path(relative_path)
            if not (source_root / path).is_file() and not (reader_root / path).is_file():
                errors.append(
                    f"book/{chapter}/README.md: missing source-like path "
                    f"{relative_path}"
                )
    return reference_count


def check_source_map(
    chapter_metadata: dict[str, dict[str, str]],
    chapter_texts: dict[str, str],
    errors: list[str],
) -> tuple[int, int]:
    if not SOURCE_MAP_PATH.is_file():
        errors.append("missing source map: meta/chapter-source-map.json")
        return 0, 0

    try:
        source_map = json.loads(SOURCE_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid source map: {exc}")
        return 0, 0

    mapped_chapters = source_map.get("chapters")
    if not isinstance(mapped_chapters, dict):
        errors.append("source map: 'chapters' must be an object")
        return 0, 0

    expected = set(CHAPTERS)
    actual = set(mapped_chapters)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        errors.append(f"source map chapter mismatch: missing={missing}; extra={extra}")

    pinned_commit = source_map.get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", pinned_commit):
        errors.append("source map: source_commit must be a 40-char SHA")

    source_path = source_map.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        errors.append("source map: source_path must be a non-empty string")
        return 0, 0
    source_root = (ROOT / source_path).resolve()
    if not source_root.is_dir():
        errors.append(f"source repository not found: {source_root}")
        return 0, 0

    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        actual_commit = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cannot read source repository commit: {exc}")
        actual_commit = ""

    if actual_commit and actual_commit != pinned_commit:
        errors.append(
            "source repository has moved: "
            f"map={pinned_commit}, actual={actual_commit}"
        )

    anchor_count = 0
    for chapter in CHAPTERS:
        metadata = chapter_metadata.get(chapter, {})
        if metadata.get("source_commit") != pinned_commit:
            errors.append(
                f"book/{chapter}/README.md: source_commit differs from source map"
            )

        references = mapped_chapters.get(chapter, [])
        if not isinstance(references, list) or not references:
            errors.append(f"source map: {chapter} must contain source references")
            continue

        for reference in references:
            anchor_count += 1
            if not isinstance(reference, dict):
                errors.append(f"source map: {chapter} has a non-object reference")
                continue
            relative_path = reference.get("path")
            symbols = reference.get("symbols")
            if not isinstance(relative_path, str) or not relative_path:
                errors.append(f"source map: {chapter} has an invalid path")
                continue
            if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
                errors.append(f"source map: unsafe source path {relative_path!r}")
                continue
            source_file = source_root / relative_path
            if not source_file.is_file():
                errors.append(f"source map: missing source file {relative_path}")
                continue
            if not isinstance(symbols, list) or not symbols:
                errors.append(f"source map: {relative_path} must list symbols")
                continue
            source_text = source_file.read_text(encoding="utf-8", errors="replace")
            for symbol in symbols:
                if not isinstance(symbol, str) or not symbol:
                    errors.append(f"source map: {relative_path} has an invalid symbol")
                elif not source_symbol_exists(source_text, symbol, python=source_file.suffix == ".py"):
                    errors.append(
                        f"source map: {relative_path} no longer contains {symbol!r}"
                    )

    explicit_reference_count = check_explicit_references(
        chapter_texts, source_root, errors
    )
    source_path_reference_count = check_source_path_references(
        chapter_texts, source_root, errors
    )
    return anchor_count, explicit_reference_count + source_path_reference_count


def main() -> int:
    errors: list[str] = []
    chapter_paths = [ROOT / "book" / chapter / "README.md" for chapter in CHAPTERS]
    chapter_metadata: dict[str, dict[str, str]] = {}
    chapter_texts: dict[str, str] = {}

    for chapter, path in zip(CHAPTERS, chapter_paths):
        if not path.is_file():
            errors.append(f"missing chapter file: {path.relative_to(ROOT)}")
            continue

        text = path.read_text(encoding="utf-8")
        chapter_texts[chapter] = text
        check_markdown(path.relative_to(ROOT), text, errors)
        check_pedagogy(path.relative_to(ROOT), text, errors)
        errors.extend(f"{path.relative_to(ROOT)}: {error}" for error in check_math(text))
        try:
            metadata = parse_frontmatter(path.relative_to(ROOT), text)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        chapter_metadata[chapter] = metadata

        missing = REQUIRED_FRONTMATTER - metadata.keys()
        if missing:
            errors.append(
                f"{path.relative_to(ROOT)}: missing frontmatter keys "
                f"{', '.join(sorted(missing))}"
            )
        if metadata.get("status") not in VALID_STATUSES:
            errors.append(
                f"{path.relative_to(ROOT)}: invalid status "
                f"{metadata.get('status')!r}"
            )
        commit = metadata.get("source_commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            errors.append(
                f"{path.relative_to(ROOT)}: source_commit must be a 40-char SHA"
            )

    for path in ROOT.rglob("*.md"):
        if any(part in IGNORED_MARKDOWN_DIRS for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        check_markdown(path.relative_to(ROOT), text, errors)
        check_local_links(path, text, errors)
        if path == ROOT / "glossary" / "terms.md":
            errors.extend(f"glossary/terms.md: {error}" for error in check_math(text))

    anchor_count, explicit_reference_count = check_source_map(
        chapter_metadata, chapter_texts, errors
    )

    if errors:
        print("Book validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Book validation passed: "
        f"{len(chapter_paths)} chapters, {anchor_count} source anchors, and "
        f"{explicit_reference_count} explicit references checked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
