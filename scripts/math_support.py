"""Shared Markdown math checks and Pandoc-to-SVG EPUB conversion."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FORMAT = "gfm+raw_html+tex_math_dollars+attributes"


@dataclass(frozen=True)
class Formula:
    tex: str
    display: bool
    line: int


def normalized_tex(tex: str) -> str:
    return " ".join(tex.split())


def markdown_formulas(text: str) -> list[Formula]:
    """Read our $inline$ / $$display$$ convention, excluding Markdown code.

    Pandoc remains the actual Markdown parser used by the build. The scanner is
    deliberately independent so a formula silently lost by Pandoc fails QA.
    """
    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if c == "\n" else " " for c in match.group())

    text = re.sub(
        r"(?ms)^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*(?:\n|$)",
        blank, text,
    )
    text = re.sub(r"(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)", blank, text)

    def escaped(index: int) -> bool:
        count = 0
        index -= 1
        while index >= 0 and text[index] == "\\":
            count += 1
            index -= 1
        return count % 2 == 1

    result = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] != "$" or escaped(cursor):
            cursor += 1
            continue
        start = cursor
        display = text.startswith("$$", cursor)
        delimiter = "$$" if display else "$"
        cursor += len(delimiter)
        body_start = cursor
        while cursor < len(text):
            if text.startswith(delimiter, cursor) and not escaped(cursor):
                break
            cursor += 1
        line = text.count("\n", 0, start) + 1
        if cursor == len(text):
            raise ValueError(f"line {line}: unclosed {delimiter} math delimiter")
        body = text[body_start:cursor]
        if not body.strip():
            raise ValueError(f"line {line}: empty formula")
        if not display and (body != body.strip() or "\n" in body):
            raise ValueError(f"line {line}: inline math must fit one line without edge spaces")
        if display:
            before = text[text.rfind("\n", 0, start) + 1:start]
            after = text[cursor + 2:text.find("\n", cursor + 2) if "\n" in text[cursor + 2:] else len(text)]
            if before.strip() or after.strip() or not body.startswith("\n") or not body.endswith("\n"):
                raise ValueError(f"line {line}: put each $$ on its own line")
        result.append(Formula(body.strip(), display, line))
        cursor += len(delimiter)
    return result


def check_math(text: str) -> list[str]:
    errors = []
    if re.search(r"(?m)^\s*```math\s*$", text):
        errors.append("use $$ display math instead of a math code fence")
    try:
        markdown_formulas(text)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def math_nodes(value: Any) -> Iterator[dict]:
    if isinstance(value, dict):
        if value.get("t") == "Math":
            yield value
        else:
            for child in value.values():
                yield from math_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from math_nodes(child)


def read_document(paths: list[Path]) -> dict:
    result = subprocess.run(
        ["pandoc", f"--from={MARKDOWN_FORMAT}", "--to=json", *map(str, paths)],
        text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


def formula_key(tex: str, display: bool) -> str:
    return hashlib.sha256(json.dumps([display, tex]).encode()).hexdigest()[:24]


def render_document_math(document: dict, stage: Path) -> list[dict]:
    """Replace actual Pandoc Math nodes; never match code or text by regex."""
    nodes = list(math_nodes(document))
    unique: dict[str, dict] = {}
    occurrences = []
    for node in nodes:
        mode, tex = node["c"]
        display = mode["t"] == "DisplayMath"
        key = formula_key(tex, display)
        entry = {"id": key, "tex": tex, "display": display}
        unique[key] = entry
        occurrences.append(entry)

    media = stage / "media" / "math"
    media.mkdir(parents=True, exist_ok=True)
    request = stage / "math-request.json"
    request.write_text(json.dumps(list(unique.values()), ensure_ascii=False))
    try:
        rendered = subprocess.run(
            ["node", str(ROOT / "scripts" / "render_math.cjs"), str(request), str(media)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr, file=sys.stderr)
        raise
    dimensions = {item["id"]: item for item in json.loads(rendered.stdout)}
    for node, item in zip(nodes, occurrences):
        dims = dimensions[item["id"]]
        cls = "math-display" if item["display"] else "math-inline"
        style = f"width:{dims['width']};"
        if not item["display"]:
            style += f"height:{dims['height']};vertical-align:{dims['vertical_align']};"
        node.clear()
        node.update({"t": "Image", "c": [
            ["", [cls], [["style", style], ["data-tex", item["tex"]]]],
            [{"t": "Str", "c": item["tex"]}],
            [f"media/math/{item['id']}.svg", ""],
        ]})
    (stage / "math-manifest.json").write_text(
        json.dumps(occurrences, ensure_ascii=False, indent=2)
    )
    return occurrences


def formula_counts(texts: list[str]) -> Counter:
    return Counter(
        (formula.display, normalized_tex(formula.tex))
        for text in texts for formula in markdown_formulas(text)
    )
