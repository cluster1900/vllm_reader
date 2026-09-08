#!/usr/bin/env python3
"""Perform dependency-free structural checks on an EPUB3 archive."""

from __future__ import annotations

import posixpath
import base64
import binascii
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import Counter

try:
    from .math_support import formula_counts, normalized_tex
except ImportError:
    from math_support import formula_counts, normalized_tex


CONTAINER_NS = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
OPF_NS = {"opf": "http://www.idpf.org/2007/opf"}
XHTML_NS = {"x": "http://www.w3.org/1999/xhtml"}
MATHML_NS = {"m": "http://www.w3.org/1998/Math/MathML"}
ROOT = Path(__file__).resolve().parents[1]


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_epub.py BOOK.epub", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    errors: list[str] = []
    if not path.is_file():
        print(f"EPUB not found: {path}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        if not infos or infos[0].filename != "mimetype":
            fail(errors, "mimetype must be the first ZIP entry")
        elif infos[0].compress_type != zipfile.ZIP_STORED:
            fail(errors, "mimetype must be stored without compression")
        elif archive.read("mimetype") != b"application/epub+zip":
            fail(errors, "invalid mimetype content")

        container_name = "META-INF/container.xml"
        if container_name not in names:
            fail(errors, "missing META-INF/container.xml")
            rootfile = None
        else:
            container = ET.fromstring(archive.read(container_name))
            element = container.find(".//c:rootfile", CONTAINER_NS)
            rootfile = None if element is None else element.attrib.get("full-path")
            if not rootfile or rootfile not in names:
                fail(errors, f"missing OPF rootfile: {rootfile!r}")

        xhtml_items: list[str] = []
        image_items: list[str] = []
        if rootfile and rootfile in names:
            opf = ET.fromstring(archive.read(rootfile))
            opf_dir = posixpath.dirname(rootfile)
            manifest: dict[str, str] = {}
            for item in opf.findall(".//opf:manifest/opf:item", OPF_NS):
                item_id = item.attrib.get("id", "")
                href = item.attrib.get("href", "")
                media_type = item.attrib.get("media-type", "")
                full_name = posixpath.normpath(posixpath.join(opf_dir, href))
                manifest[item_id] = full_name
                if full_name not in names:
                    fail(errors, f"manifest item is missing: {full_name}")
                if media_type == "application/xhtml+xml":
                    xhtml_items.append(full_name)
                if media_type.startswith("image/"):
                    image_items.append(full_name)

            for itemref in opf.findall(".//opf:spine/opf:itemref", OPF_NS):
                item_id = itemref.attrib.get("idref", "")
                if item_id not in manifest:
                    fail(errors, f"spine idref missing from manifest: {item_id}")

        chapter_heads = 0
        mermaid_code_blocks = 0
        math_expressions = 0
        chapter_reading_guides = 0
        section_leads = 0
        diagram_guides = 0
        diagram_images = 0
        rendered_formulas: Counter = Counter()
        for item_name in xhtml_items:
            if item_name not in names:
                continue
            document_bytes = archive.read(item_name)
            if b"![" in document_bytes:
                fail(errors, f"{item_name}: contains unrendered Markdown image syntax")
            if b"{.diagram}" in document_bytes:
                fail(errors, f"{item_name}: contains an unparsed diagram attribute")
            document = ET.fromstring(document_bytes)
            chapter_heads += len(document.findall(".//x:h1", XHTML_NS))
            math_expressions += len(document.findall(".//m:math", MATHML_NS))
            for heading in document.findall(".//x:h2", XHTML_NS):
                if "".join(heading.itertext()).strip() == "如何阅读本章":
                    chapter_reading_guides += 1
            for strong in document.findall(".//x:strong", XHTML_NS):
                label = "".join(strong.itertext()).strip()
                if label == "本节先看：":
                    section_leads += 1
                elif label == "读图方法：":
                    diagram_guides += 1
            for code in document.findall(".//x:code", XHTML_NS):
                classes = code.attrib.get("class", "").split()
                if "mermaid" in classes:
                    mermaid_code_blocks += 1

            base = posixpath.dirname(item_name)
            for image in document.findall(".//x:img", XHTML_NS):
                classes = image.attrib.get("class", "").split()
                if "diagram" in classes:
                    diagram_images += 1
                is_math = "math-inline" in classes or "math-display" in classes
                if is_math:
                    tex = image.attrib.get("data-tex", "")
                    if not tex or normalized_tex(image.attrib.get("alt", "")) != normalized_tex(tex):
                        fail(errors, f"{item_name}: formula has missing/mismatched TeX alt text")
                    rendered_formulas[("math-display" in classes, normalized_tex(tex))] += 1
                source = image.attrib.get("src", "")
                if source.startswith(("http://", "https://", "data:")):
                    if is_math:
                        fail(errors, f"{item_name}: formula must use a packaged SVG file")
                    continue
                target = posixpath.normpath(posixpath.join(base, source))
                if target not in names:
                    fail(errors, f"XHTML image target is missing: {target}")
                elif is_math:
                    try:
                        svg = ET.fromstring(archive.read(target))
                    except ET.ParseError as exc:
                        fail(errors, f"{target}: invalid SVG XML: {exc}")
                        continue
                    if not svg.tag.endswith("}svg") or not svg.attrib.get("viewBox"):
                        fail(errors, f"{target}: invalid formula SVG")
                    try:
                        svg_tex = base64.b64decode(svg.attrib.get("data-tex-b64", ""), validate=True).decode("utf-8")
                    except (binascii.Error, UnicodeDecodeError):
                        svg_tex = ""
                    if normalized_tex(svg_tex) != normalized_tex(tex):
                        fail(errors, f"{target}: SVG source differs from formula TeX")
                    expected_mode = "block" if "math-display" in classes else "inline"
                    if svg.attrib.get("data-display") != expected_mode:
                        fail(errors, f"{target}: SVG inline/display mode mismatch")
                    for element in svg.iter():
                        if element.tag.endswith("}text") or element.attrib.get("data-mml-node") == "merror":
                            fail(errors, f"{target}: formula has an error or font-dependent glyph")
                        for attr, value in element.attrib.items():
                            if attr.endswith("href") and not value.startswith("#"):
                                fail(errors, f"{target}: formula depends on an external resource")

            for link in document.findall(".//x:a", XHTML_NS):
                href = link.attrib.get("href", "")
                if not href or href.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = href.split("#", 1)[0]
                resolved = posixpath.normpath(posixpath.join(base, target))
                if resolved not in names:
                    fail(errors, f"XHTML link target is missing: {resolved}")

        if chapter_heads < 11:
            fail(errors, f"expected at least 11 h1 headings, found {chapter_heads}")
        if not image_items:
            fail(errors, "EPUB contains no images")
        if not rendered_formulas:
            fail(errors, "EPUB contains no SVG formulas")
        if mermaid_code_blocks:
            fail(errors, f"found {mermaid_code_blocks} unrendered Mermaid blocks")

        chapter_sources = sorted((ROOT / "book").glob("*/README.md"))
        expected_chapter_guides = len(chapter_sources)
        expected_section_leads = 0
        expected_diagram_guides = 0
        for source in chapter_sources:
            source_text = source.read_text(encoding="utf-8")
            expected_section_leads += source_text.count("> **本节先看：**")
            expected_diagram_guides += source_text.count("> **读图方法：**")
        math_sources = chapter_sources + [ROOT / "glossary" / "terms.md"]
        expected_formulas = formula_counts([p.read_text(encoding="utf-8") for p in math_sources])
        if rendered_formulas != expected_formulas:
            fail(errors, "formula mismatch between Markdown and EPUB: "
                 f"{sum((expected_formulas - rendered_formulas).values())} missing/changed, "
                 f"{sum((rendered_formulas - expected_formulas).values())} unexpected")
        if chapter_reading_guides != expected_chapter_guides:
            fail(
                errors,
                "chapter reading guide mismatch: "
                f"expected {expected_chapter_guides}, found {chapter_reading_guides}",
            )
        if section_leads != expected_section_leads:
            fail(
                errors,
                "section lead mismatch: "
                f"expected {expected_section_leads}, found {section_leads}",
            )
        if diagram_guides != expected_diagram_guides:
            fail(
                errors,
                "diagram guide mismatch: "
                f"expected {expected_diagram_guides}, found {diagram_guides}",
            )
        if diagram_images != diagram_guides:
            fail(
                errors,
                "diagram/image mismatch: "
                f"{diagram_images} diagram images, {diagram_guides} guides",
            )

    if errors:
        print("EPUB validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        f"EPUB validation passed: {len(xhtml_items)} XHTML files, "
        f"{len(image_items)} images, {sum(rendered_formulas.values())} SVG formulas, "
        f"{math_expressions} MathML expressions, "
        f"{chapter_heads} h1 headings, {section_leads} section leads, "
        f"{diagram_guides} diagram guides."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
