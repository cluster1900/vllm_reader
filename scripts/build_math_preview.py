#!/usr/bin/env python3
"""Build offline side-by-side previews of every Markdown/EPUB formula."""
from __future__ import annotations

from copy import deepcopy
import html
import json
from pathlib import Path
import re
import subprocess

try:
    from .math_support import ROOT, markdown_formulas, read_document, math_nodes, render_document_math
except ImportError:
    from math_support import ROOT, markdown_formulas, read_document, math_nodes, render_document_math


def build() -> Path:
    paths = sorted((ROOT / 'book').glob('*/README.md')) + [ROOT / 'glossary/terms.md']
    formulas = [(path.relative_to(ROOT), f) for path in paths for f in markdown_formulas(path.read_text())]
    document = read_document(paths)
    nodes = list(math_nodes(document))
    assert len(nodes) == len(formulas)
    math_only = {'pandoc-api-version': document['pandoc-api-version'], 'meta': {},
                 'blocks': [{'t': 'Para', 'c': [deepcopy(n)]} for n in nodes]}
    native = subprocess.run(['pandoc', '--from=json', '--to=html5', '--mathml'],
                            input=json.dumps(math_only), text=True, capture_output=True, check=True)
    mathml = re.findall(r'<math\b.*?</math>', native.stdout, re.S)
    if len(mathml) != len(nodes):
        raise ValueError('Pandoc HTML preview lost a formula: ' + native.stderr)
    stage = ROOT / 'build/math-preview'
    stage.mkdir(parents=True, exist_ok=True)
    for old_page in stage.glob('page-*.html'):
        old_page.unlink()
    occurrences = render_document_math(document, stage)
    # Dimensions are read from the generated SVG, rather than inferred from text.
    from xml.etree import ElementTree as ET
    cards = []
    for i, ((source, formula), occurrence, native_math) in enumerate(zip(formulas, occurrences, mathml), 1):
        assert formula.tex.strip() == occurrence['tex'].strip(), (source, formula.line)
        src = f"media/math/{occurrence['id']}.svg"
        svg = ET.parse(stage / src).getroot()
        width, height = svg.attrib['width'], svg.attrib['height']
        vertical = svg.attrib.get('style', '')
        cls = 'math-display' if formula.display else 'math-inline'
        style = f'width:{width};' + ('' if formula.display else f'height:{height};{vertical}')
        image = f'<img class="{cls}" src="{src}" style="{style}" alt="{html.escape(formula.tex, quote=True)}">'
        if not formula.display:
            image = '行内文字 '+image+' 后续文字。'
            native_math = '行内文字 '+native_math+' 后续文字。'
        cards.append(f'<article><h2>{i:03} · {html.escape(str(source))}:{formula.line}</h2>'
                     f'<pre>{html.escape(formula.tex)}</pre><div class="pair">'
                     f'<section><h3>Markdown 解析 → MathML</h3><div class="formula">{native_math}</div></section>'
                     f'<section><h3>EPUB → SVG 字形</h3><div class="formula">{image}</div></section>'
                     '</div></article>')
    css = '''body{font:16px/1.6 "PingFang SC",sans-serif;margin:24px;background:#f5f7fa;color:#17202a}
main{max-width:1160px;margin:auto}article{background:white;padding:16px;margin:16px 0;border:1px solid #d9e1e8;border-radius:8px;break-inside:avoid}
h1{font-size:26px}h2{font-size:13px;color:#52616d;margin:0}h3{font-size:12px;color:#52616d}
pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:8px}
.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:24px}.formula{font-size:20px;min-height:55px;overflow:auto;padding:6px 0}
img.math-inline{display:inline;margin:0 .08em;max-width:none}img.math-display{display:block;max-width:100%;height:auto;margin:0 auto}
@media(max-width:600px){.pair{grid-template-columns:minmax(0,1fr)}body{margin:12px}}
'''
    def page(content: str) -> str:
        return f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>公式显示复核</title><style>{css}</style><main><h1>公式显示复核</h1><p>同一份 TeX 的 Markdown 解析与 EPUB 字形对照；本页离线可读。</p>{content}</main></html>'
    (stage / 'index.html').write_text(page(''.join(cards)))
    for start in range(0, len(cards), 8):
        (stage / f'page-{start // 8 + 1:02}.html').write_text(page(''.join(cards[start:start + 8])))
    print(f'Formula preview: {len(cards)} expressions, {(len(cards)+7)//8} pages; {stage / "index.html"}')
    return stage


if __name__ == '__main__':
    build()
