#!/usr/bin/env python3
"""Build an EPUB3 edition with Mermaid diagrams rendered as SVG."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .epub_content import source_paths, annotate_document, rewrite_links, stamp_mermaid, add_legacy_cover_metadata
    from .math_support import (check_math, formula_counts, math_nodes,
                               normalized_tex, read_document, render_document_math)
except ImportError:
    from epub_content import source_paths, annotate_document, rewrite_links, stamp_mermaid, add_legacy_cover_metadata
    from math_support import (check_math, formula_counts, math_nodes,
                              normalized_tex, read_document, render_document_math)


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
MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)
IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]+)\)")


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("unclosed YAML frontmatter")
    return text[end + 5 :]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def render_mermaid(
    text: str,
    chapter_name: str,
    media_dir: Path,
    mmdc: Path,
) -> str:
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        stem = f"diagram-{counter:02d}"
        source = media_dir / f"{stem}.mmd"
        output = media_dir / f"{stem}.svg"
        source.write_text(match.group(1).strip() + "\n", encoding="utf-8")
        run(
            [
                str(mmdc),
                "--input",
                str(source),
                "--output",
                str(output),
                "--backgroundColor",
                "white",
                "--configFile",
                str(ROOT / "epub" / "mermaid-config.json"),
                "--puppeteerConfigFile",
                str(ROOT / "epub" / "puppeteer-config.json"),
            ]
        )
        stamp_mermaid(output, match.group(1))
        source.unlink()
        relative = f"media/{chapter_name}/{output.name}"
        return f"![{chapter_name} 图 {counter}]({relative}){{.diagram}}"

    return MERMAID_RE.sub(replace, text)


def normalize_display_math(text: str) -> str:
    """Require native Markdown math; never hide invalid source behind conversion."""

    errors = check_math(text)
    if errors:
        raise ValueError("Invalid Markdown math: " + "; ".join(errors))
    return text


def copy_images(text: str, source_path: Path, chapter_name: str, media_dir: Path) -> str:
    def replace(match: re.Match[str]) -> str:
        alt, target = match.groups()
        if target.startswith(("http://", "https://", "data:")):
            return match.group(0)
        clean_target = target.split("#", 1)[0]
        source_image = (source_path.parent / clean_target).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(f"missing image in {source_path}: {target}")
        destination = media_dir / source_image.name
        if source_image != destination:
            shutil.copy2(source_image, destination)
        return f"![{alt}](media/{chapter_name}/{destination.name})"

    return IMAGE_RE.sub(replace, text)


def make_intro() -> str:
    """The reading guide is Markdown source, just like every chapter."""
    return (ROOT / "book" / "reading-guide.md").read_text(encoding="utf-8")


def build(output: Path, keep_stage: bool) -> None:
    pandoc = Path(shutil.which("pandoc") or "")
    mmdc = ROOT / "node_modules" / ".bin" / "mmdc"
    if not pandoc.is_file():
        raise RuntimeError("pandoc is required")
    if not mmdc.is_file():
        raise RuntimeError("Mermaid CLI is missing; run 'pnpm install' first")

    run([sys.executable, str(ROOT / "scripts" / "enrich_pedagogy.py"), "--check"])
    run([sys.executable, str(ROOT / "scripts" / "check_book.py")])

    stage = ROOT / "build" / "epub-src"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    staged_markdown = []
    prepared = []
    anchors = {}
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    for index, source in enumerate(source_paths(ROOT)):
        chapter_name = source.parent.name if source.name == "README.md" else source.stem
        text = normalize_display_math(strip_frontmatter(source.read_text(encoding="utf-8")))
        media_dir = stage / "media" / chapter_name
        media_dir.mkdir(parents=True)
        text = copy_images(text, source, chapter_name, media_dir)
        text = render_mermaid(text, chapter_name, media_dir, mmdc)
        destination = stage / f"{index:02d}-{chapter_name}.md"
        destination.write_text(text.rstrip() + "\n", encoding="utf-8")
        staged_markdown.append(destination)
        part = read_document([destination])
        anchors[source.resolve()] = annotate_document(part, source, ROOT, revision)
        prepared.append((source, part))

    for source, part in prepared:
        rewrite_links(part, source, ROOT, revision, anchors)
    document = {"pandoc-api-version": prepared[0][1]["pandoc-api-version"], "meta": {},
                "blocks": [block for _, part in prepared for block in part["blocks"]]}
    expected_math = formula_counts([path.read_text() for path in staged_markdown])
    parsed_math = Counter(
        (node["c"][0]["t"] == "DisplayMath", normalized_tex(node["c"][1]))
        for node in math_nodes(document)
    )
    if parsed_math != expected_math:
        raise ValueError("Pandoc math nodes do not match the Markdown formulas")
    formulas = render_document_math(document, stage)
    print(f"Rendered {len(formulas)} math occurrences as self-contained SVG.")
    document_path = stage / "book.json"
    document_path.write_text(json.dumps(document, ensure_ascii=False))

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(pandoc),
        "--from=json",
        "--to=epub3",
        "--toc",
        "--toc-depth=2",
        "--split-level=1",
        "--template",
        str(ROOT / "epub" / "template.xhtml"),
        "--metadata-file",
        str(ROOT / "epub" / "metadata.yaml"),
        "--css",
        str(ROOT / "epub" / "style.css"),
        "--epub-cover-image",
        str(ROOT / "epub" / "cover-imagegen.png"),
        "--resource-path",
        str(stage),
        "--output",
        str(output),
        str(document_path),
    ]
    run(command)
    add_legacy_cover_metadata(output)
    run([sys.executable, str(ROOT / "scripts" / "check_epub.py"), str(output)])

    if not keep_stage:
        shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "vllm-source-guide.epub",
    )
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()
    build(args.output.resolve(), args.keep_stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
