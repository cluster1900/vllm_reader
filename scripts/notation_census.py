#!/usr/bin/env python3
"""Book-wide symbol census: which letters the formulas and code spans use, where.

Reads every `book/*/README.md` with the same formula scanner `check_book.py` uses
(`math_support.markdown_formulas`) and prints, per symbol, how many formulas in each
chapter use it. `--code` adds short code-font spans that look like symbols
(`T`, `Hq`, `[B,S,Hq,Dh]`). Output is Markdown, meant to be pasted into
`meta/notation-census-*.md` when a round of notation work starts or ends.

    python3 scripts/notation_census.py            # LaTeX symbols
    python3 scripts/notation_census.py --code     # plus code-font symbols
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re

try:
    from .math_support import ROOT, markdown_formulas
except ImportError:
    from math_support import ROOT, markdown_formulas

# A symbol is a letter or Greek command with its optional sub/superscript.
SYMBOL = re.compile(
    r"(\\(?:alpha|beta|gamma|delta|Delta|theta|lambda|mu|pi|rho|sigma|Sigma|tau|ell|omega|"
    r"kappa|epsilon|phi|psi|eta|chi)|[A-Za-z])"
    r"((?:_\{(?:[^{}]|\{[^{}]*\})*\}|_(?:[A-Za-z0-9]|\\[A-Za-z]+))?(?:\^\{[^{}]*\}|\^[A-Za-z0-9])?)"
)
NAMED = re.compile(r"\\(?:operatorname\*?|mathrm|text|mathtt|mathsf)\{[^{}]*\}")
ENVIRONMENT = re.compile(r"\\(?:begin|end)\{[a-z]+\}")
COMMAND = re.compile(
    r"\\(?:left|right|bigl|bigr|big|Big|quad|qquad|frac|sqrt|sum|prod|max|min|log|exp|cdot|"
    r"times|approx|leq?|geq?|neq?|wedge|vee|land|lor|bigwedge|iff|bmod|lfloor|rfloor|lceil|"
    r"rceil|in|to|mathbf|mathcal|hat|odot|ldots|cdots|vdots|top|pm|sim|infty|cos|sin|arg|ln|"
    r"lVert|rVert|lvert|rvert|neg|exists|ge|le)(?![A-Za-z])|\\[;,!\\]"
)
# Short code-font spans that read like a symbol rather than an identifier or path.
CODE_SYMBOL = re.compile(
    r"[A-Za-z]|[A-Z][a-z]|[A-Za-z]_[A-Za-z0-9,{}]+|\[(?=[^\]]*[A-Za-z])[A-Za-z_,*= 0-9]+\]|[A-Za-z]+:\[[A-Za-z_,*0-9 ]+\]"
)


SCRIPT = re.compile(r"([_^])\{((?:[^{}]|\{[^{}]*\})*)\}")
WRAPPER = re.compile(r"\\(?:operatorname\*?|mathrm|text|mathtt|mathsf)\{([^{}]*)\}")


def flatten_scripts(tex: str) -> str:
    """`B_{\\mathrm{token}}` -> `B_{token}` so a descriptive subscript stays part of its symbol."""
    def one(match: re.Match[str]) -> str:
        body = WRAPPER.sub(lambda m: m.group(1), match.group(2))
        return f"{match.group(1)}{{{' '.join(body.replace(chr(92) + ' ', ' ').split())}}}"
    return SCRIPT.sub(one, tex)


def formula_symbols(tex: str) -> Counter:
    """Count the symbols used by one formula; named operators are kept whole."""
    tex = ENVIRONMENT.sub(" ", flatten_scripts(tex))
    named = NAMED.findall(tex)
    # a named operator takes its own sub/superscripts with it: \mathrm{score}^{base}_i
    rest = COMMAND.sub(" ", NAMED.sub(" \x00", tex))
    rest = re.sub(r"\x00(?:[_^](?:\{[^{}]*\}|[A-Za-z0-9]))*", " ", rest)
    counts = Counter(m.group(1) + m.group(2) for m in SYMBOL.finditer(rest))
    counts.update(" ".join(n.split()) for n in named)
    return counts


def code_symbols(text: str) -> list[tuple[int, str]]:
    """Short symbol-like code spans outside fenced blocks, with their line numbers."""
    found, fenced = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if re.match(r"^\s*```", line):
            fenced = not fenced
            continue
        if fenced:
            continue
        for span in re.findall(r"`([^`\n]+)`", line):
            if len(span) <= 24 and CODE_SYMBOL.fullmatch(span):
                found.append((number, span))
    return found


def census(root: Path = ROOT, include_code: bool = False):
    math: dict[str, Counter] = defaultdict(Counter)
    code: dict[str, Counter] = defaultdict(Counter)
    totals = Counter()
    for path in sorted((root / "book").glob("*/README.md")):
        chapter = path.parent.name[:2]
        text = path.read_text(encoding="utf-8")
        for formula in markdown_formulas(text):
            totals[chapter] += 1
            for symbol in formula_symbols(formula.tex):
                math[symbol][chapter] += 1
        if include_code:
            for _, span in code_symbols(text):
                code[span][chapter] += 1
    return math, code, totals


def render(table: dict[str, Counter]) -> list[str]:
    rows = ["| 符号 | 次数 | 分章 |", "|---|---:|---|"]
    for symbol, by_chapter in sorted(table.items(), key=lambda kv: (-sum(kv[1].values()), kv[0])):
        cell = " ".join(f"{c}:{n}" for c, n in sorted(by_chapter.items()))
        escaped = symbol.replace("|", "\\|")
        rows.append(f"| `{escaped}` | {sum(by_chapter.values())} | {cell} |")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--code", action="store_true", help="also list code-font symbols")
    args = parser.parse_args()
    math, code, totals = census(ROOT, args.code)
    print(f"公式 {sum(totals.values())} 个（" + "、".join(f"{c}:{n}" for c, n in sorted(totals.items())) + "）；"
          f"LaTeX 符号 {len(math)} 种。\n")
    print("\n".join(render(math)))
    if args.code:
        print(f"\n代码样式符号 {len(code)} 种。\n")
        print("\n".join(render(code)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
