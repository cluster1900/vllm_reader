#!/usr/bin/env python3
"""Build an EPUB3 edition with Mermaid diagrams rendered as SVG."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


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
MATH_RE = re.compile(r"```math\s*\n(.*?)\n```", re.DOTALL)
IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]+)\)")
LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\n]+)\)")


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
        source.unlink()
        relative = f"media/{chapter_name}/{output.name}"
        return f"![{chapter_name} 图 {counter}]({relative}){{.diagram}}"

    return MERMAID_RE.sub(replace, text)


def normalize_display_math(text: str) -> str:
    """Turn documentation-friendly math fences into Pandoc display math."""

    return MATH_RE.sub(lambda match: f"$$\n{match.group(1).strip()}\n$$", text)


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


def neutralize_repo_links(text: str) -> str:
    """Avoid dangling links to files that are not packaged in the EPUB."""

    def replace(match: re.Match[str]) -> str:
        label, target = match.groups()
        if target.startswith(("http://", "https://", "mailto:", "#")):
            return match.group(0)
        return f"{label}（随书仓库：`{target}`）"

    return LINK_RE.sub(replace, text)


def make_intro() -> str:
    return """# 阅读指南

本书面向能够阅读 Python 类、函数和测试，但不要求预先掌握大模型推理、CUDA 或分布式
系统的读者。全书以 vLLM V1 为主线，从 Transformer 的一次 token 预测开始，逐步进入
EngineCore、Scheduler、KV Cache、PagedAttention、模型执行、CUDA Graph 和多 GPU。

## 源码基线

- Repository: `vllm-project/vllm`
- Commit: `5893426b88f7b3cd21101d194eb1c6f0a6f0e27b`
- Branch: `main`
- Verified at: `2026-09-07`

章节中的“源码事实”均绑定上述 revision。标为 `draft` 的章节表示正文已经形成，但真实
NVIDIA GPU 动态实验或独立人工复核尚未全部完成。

## 阅读方法

先按顺序阅读第 01 至 05 章建立请求、调度和内存主线；再阅读第 06 至 09 章理解 GPU
执行和分布式；最后用第 10 章的方法设计实验。代码路径使用仓库相对路径，符号名比
行号更适合在后续版本中重新定位。
"""


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

    intro = stage / "00-reading-guide.md"
    intro.write_text(make_intro(), encoding="utf-8")
    staged_markdown.append(intro)

    for index, chapter_name in enumerate(CHAPTERS, start=1):
        source = ROOT / "book" / chapter_name / "README.md"
        text = strip_frontmatter(source.read_text(encoding="utf-8"))
        text = normalize_display_math(text)
        media_dir = stage / "media" / chapter_name
        media_dir.mkdir(parents=True)
        text = copy_images(text, source, chapter_name, media_dir)
        text = render_mermaid(text, chapter_name, media_dir, mmdc)
        text = neutralize_repo_links(text)
        destination = stage / f"{index:02d}-{chapter_name}.md"
        destination.write_text(text.rstrip() + "\n", encoding="utf-8")
        staged_markdown.append(destination)

    glossary_source = ROOT / "glossary" / "terms.md"
    glossary = stage / "99-glossary.md"
    glossary.write_text(
        neutralize_repo_links(glossary_source.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    staged_markdown.append(glossary)

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(pandoc),
        "--from=gfm+raw_html+tex_math_dollars+attributes",
        "--to=epub3",
        "--mathml",
        "--toc",
        "--toc-depth=2",
        "--split-level=1",
        "--metadata-file",
        str(ROOT / "epub" / "metadata.yaml"),
        "--css",
        str(ROOT / "epub" / "style.css"),
        "--epub-cover-image",
        str(ROOT / "epub" / "cover.svg"),
        "--resource-path",
        str(stage),
        "--output",
        str(output),
        *(str(path) for path in staged_markdown),
    ]
    run(command)
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
