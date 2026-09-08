"""Preserve source links and compare EPUB reading content with Markdown."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import posixpath
import re
import subprocess
import tempfile
import zipfile
from urllib.parse import quote, unquote, urlsplit
from xml.etree import ElementTree as ET

try:
    from .math_support import MARKDOWN_FORMAT, normalized_tex
except ImportError:
    from math_support import MARKDOWN_FORMAT, normalized_tex

READER_REPOSITORY = "https://github.com/cluster1900/vllm_reader"


def source_paths(root: Path) -> list[Path]:
    return [root / "book/reading-guide.md", *sorted((root / "book").glob("*/README.md")),
            root / "glossary/terms.md"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_metadata(text: str) -> str:
    if text.startswith("---\n"):
        return text.split("\n---\n", 1)[1]
    return text


def prefix(path: Path) -> str:
    # EPUB IDs must not start with a digit, otherwise Pandoc prepends "id_".
    return "chapter-" + path.parent.name if path.name == "README.md" else path.stem


def ast_nodes(value, kind):
    if isinstance(value, dict):
        if value.get("t") == kind:
            yield value
        else:
            for child in value.values():
                yield from ast_nodes(child, kind)
    elif isinstance(value, list):
        for child in value:
            yield from ast_nodes(child, kind)


def annotate_document(document: dict, source: Path, root: Path, revision: str) -> dict[str, str]:
    root, source = root.resolve(), source.resolve()
    anchors = {}
    first = True
    for header in ast_nodes(document, "Header"):
        level, attrs, _ = header["c"]
        old = attrs[0]
        attrs[0] = f"{prefix(source)}--{old}"
        anchors[old] = attrs[0]
        if level == 1 and first:
            attrs[2].extend([
                ["data-source-path", source.relative_to(root).as_posix()],
                ["data-source-sha256", digest(source.read_bytes())],
                ["data-reader-commit", revision],
            ])
            first = False
    if first:
        raise ValueError(f"{source}: no main heading")
    return anchors


def resolve_local(href: str, source: Path) -> tuple[Path, str]:
    parts = urlsplit(href)
    target = (source.parent / unquote(parts.path)).resolve() if parts.path else source.resolve()
    if target.is_dir():
        target /= "README.md"
    return target, unquote(parts.fragment)


def rewrite_links(document: dict, source: Path, root: Path, revision: str,
                  anchors: dict[Path, dict[str, str]]) -> None:
    root, source = root.resolve(), source.resolve()
    anchors = {path.resolve(): mapping for path, mapping in anchors.items()}
    for node in ast_nodes(document, "Link"):
        target = node["c"][2]
        href = target[0]
        if urlsplit(href).scheme or href.startswith("//"):
            continue
        path, fragment = resolve_local(href, source)
        if path in anchors:
            mapping = anchors[path]
            key = fragment or next(iter(mapping))
            if key not in mapping:
                raise ValueError(f"{source}: missing Markdown anchor {href}")
            target[0] = "#" + mapping[key]
        else:
            relative = path.relative_to(root).as_posix()
            if not path.exists():
                raise FileNotFoundError(path)
            target[0] = f"{READER_REPOSITORY}/blob/{revision}/{quote(relative)}"
            if fragment:
                target[0] += "#" + quote(fragment)


def stamp_mermaid(path: Path, source: str) -> None:
    # Preserve Mermaid's SVG bytes and namespaces; only annotate the root tag.
    svg = path.read_text()
    code = source.strip() + "\n"
    payload = base64.b64encode(code.encode()).decode()
    svg = svg.replace("<svg ", f'<svg data-mermaid-b64="{payload}" ', 1)
    path.write_text(svg)


def add_legacy_cover_metadata(path: Path) -> None:
    """Keep EPUB3 cover-image and older reader thumbnail metadata in agreement."""
    opf_ns = "http://www.idpf.org/2007/opf"
    ns = {"o": opf_ns, "c": "urn:oasis:names:tc:opendocument:xmlns:container"}
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        opf_path = container.find(".//c:rootfile", ns).attrib["full-path"]
        opf = ET.fromstring(archive.read(opf_path))
        covers = [e for e in opf.findall(".//o:manifest/o:item", ns)
                  if "cover-image" in e.attrib.get("properties", "").split()]
        if len(covers) != 1:
            raise ValueError("EPUB must identify exactly one cover image")
        metadata = opf.find("o:metadata", ns)
        entry = metadata.find('o:meta[@name="cover"]', ns)
        if entry is None:
            entry = ET.SubElement(metadata, f"{{{opf_ns}}}meta", {"name": "cover"})
        entry.set("content", covers[0].attrib["id"])
        ET.register_namespace("", opf_ns)
        ET.register_namespace("dc", "http://purl.org/dc/elements/1.1/")
        replacement = ET.tostring(opf, encoding="utf-8", xml_declaration=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".epub", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with zipfile.ZipFile(temporary, "w") as updated:
                for info in archive.infolist():
                    updated.writestr(info, replacement if info.filename == opf_path else archive.read(info.filename))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def source_html(source: Path) -> ET.Element:
    text = strip_metadata(source.read_text())
    result = subprocess.run(
        ["pandoc", f"--from={MARKDOWN_FORMAT}", "--to=html5", "--mathml", "--wrap=none"],
        input=text, text=True, capture_output=True, check=True,
    )
    return ET.fromstring("<body>" + result.stdout + "</body>")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def canonical_body(body: ET.Element, image_reader, link_target) -> list[tuple]:
    """Ordered semantic tokens: preserve blocks, tables, code, math and images.

    Ignore only presentation wrappers/IDs and prose line-wrapping whitespace.
    Code content is exact apart from line endings and the writer's final newline.
    """
    tokens = []
    containers = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li",
                  "blockquote", "table", "thead", "tbody", "tfoot", "tr", "th", "td",
                  "strong", "em", "del", "summary", "details", "code", "hr", "br", "figcaption"}

    def text(value):
        for word in re.findall(r"\S+", value or ""):
            tokens.append(("text", word))

    def visit(element):
        tag = local_name(element.tag)
        classes = element.attrib.get("class", "").split()
        if tag == "pre":
            code = next((e for e in element if local_name(e.tag) == "code"), element)
            content = "".join(code.itertext()).replace("\r\n", "\n").rstrip("\n")
            if "mermaid" in classes + code.attrib.get("class", "").split():
                tokens.append(("mermaid", digest((content.strip() + "\n").encode())))
            else:
                tokens.append(("code-block", content))
            return
        if tag in {"p", "figure"} and not (element.text or "").strip():
            children = list(element)
            if children and local_name(children[0].tag) == "img" and "diagram" in children[0].attrib.get("class", "").split():
                # The generated diagram label replaces a Mermaid code block.
                visit(children[0])
                return
        if tag == "math":
            annotation = next(e for e in element.iter() if local_name(e.tag) == "annotation")
            tokens.append(("math", element.attrib.get("display") == "block", normalized_tex("".join(annotation.itertext()))))
            return
        if tag == "img":
            data = image_reader(element.attrib["src"])
            if "math-inline" in classes or "math-display" in classes:
                tokens.append(("math", "math-display" in classes, normalized_tex(element.attrib["data-tex"])))
            elif "diagram" in classes:
                svg = ET.fromstring(data)
                code = base64.b64decode(svg.attrib["data-mermaid-b64"], validate=True)
                tokens.append(("mermaid", digest(code)))
            else:
                tokens.append(("image", element.attrib.get("alt", ""), digest(data)))
            return
        if tag == "a":
            tokens.append(("link-start", link_target(element.attrib.get("href", ""))))
        elif tag == "input":
            tokens.append(("input", element.attrib.get("type"), "checked" in element.attrib))
        elif tag in containers:
            extra = ()
            if tag in {"th", "td"}:
                extra = (element.attrib.get("rowspan", "1"), element.attrib.get("colspan", "1"))
            elif tag == "ol":
                extra = (element.attrib.get("start", "1"),)
            tokens.append(("start", tag, *extra))
        text(element.text)
        for child in element:
            visit(child)
            text(child.tail)
        if tag == "a":
            tokens.append(("link-end",))
        elif tag in containers:
            tokens.append(("end", tag))

    visit(body)
    return tokens


def difference(expected: list[tuple], actual: list[tuple]) -> str | None:
    if expected == actual:
        return None
    index = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
    return f"difference at token {index}: Markdown={expected[index:index+3]!r}; EPUB={actual[index:index+3]!r}"


def check_content(archive, spine: list[str], root: Path, errors: list[str]) -> dict:
    root = root.resolve()
    sources = source_paths(root)
    expected_paths = [p.relative_to(root).as_posix() for p in sources]
    html = {p.resolve(): source_html(p) for p in sources}
    first_ids = {}
    anchor_sources = {}
    expected_toc = []
    for source, tree in html.items():
        for header in tree.iter():
            if local_name(header.tag) in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                old = header.attrib.get("id", "")
                first_ids.setdefault(source, old)
                anchor_sources[f"{prefix(source)}--{old}"] = (source.relative_to(root).as_posix(), old)
                if local_name(header.tag) in {"h1", "h2"}:
                    expected_toc.append((" ".join("".join(header.itertext()).split()),
                                         ("chapter", source.relative_to(root).as_posix(), old)))

    def expected_link(href, source):
        if urlsplit(href).scheme or href.startswith("//"):
            return ("url", href)
        target, fragment = resolve_local(href, source)
        relative = target.relative_to(root).as_posix()
        if target in html:
            return ("chapter", relative, fragment or first_ids[target])
        return ("repo", relative, fragment)

    def actual_link(href):
        if href.startswith(READER_REPOSITORY + "/blob/"):
            rest = href[len(READER_REPOSITORY + "/blob/"):]
            revision, _, target = rest.partition("/")
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                errors.append(f"repository link must pin a commit: {href}")
            parts = urlsplit(target)
            return ("repo", unquote(parts.path), unquote(parts.fragment))
        if urlsplit(href).scheme or href.startswith("//"):
            return ("url", href)
        fragment = unquote(urlsplit(href).fragment)
        if fragment in anchor_sources:
            return ("chapter", *anchor_sources[fragment])
        return ("unmapped", href)

    found = []
    total_tokens = 0
    for name in spine:
        tree = ET.fromstring(archive.read(name))
        marked = [e for e in tree.iter() if local_name(e.tag) == "h1" and e.attrib.get("data-source-path")]
        if not marked:
            if posixpath.basename(name) not in {"cover.xhtml", "title_page.xhtml", "nav.xhtml"}:
                errors.append(f"{name}: unexpected reading-order page without a Markdown source")
            continue  # cover, title and navigation are checked separately
        if len(marked) != 1:
            errors.append(f"{name}: expected exactly one Markdown source marker")
            continue
        marker = marked[0]
        relative = marker.attrib["data-source-path"]
        found.append(relative)
        if relative not in expected_paths:
            errors.append(f"{name}: unexpected Markdown source {relative}")
            continue
        source = (root / relative).resolve()
        if marker.attrib.get("data-source-sha256") != digest(source.read_bytes()):
            errors.append(f"{relative}: EPUB was built from different Markdown bytes")
        body = next(e for e in tree.iter() if local_name(e.tag) == "body")
        expected = canonical_body(html[source], lambda src: (source.parent / unquote(src)).read_bytes(),
                                  lambda href: expected_link(href, source))
        actual = canonical_body(body, lambda src: archive.read(posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(src)))), actual_link)
        mismatch = difference(expected, actual)
        if mismatch:
            errors.append(f"{relative}: {mismatch}")
        total_tokens += len(expected)
    if found != expected_paths:
        errors.append(f"EPUB source order/presence differs: expected {expected_paths}; found {found}")
    actual_toc = []
    for name in archive.namelist():
        if name.endswith(".xhtml"):
            tree = ET.fromstring(archive.read(name))
            for nav in tree.iter():
                if local_name(nav.tag) == "nav" and nav.attrib.get("{http://www.idpf.org/2007/ops}type") == "toc":
                    actual_toc.extend((" ".join("".join(a.itertext()).split()), actual_link(a.attrib.get("href", "")))
                                      for a in nav.iter() if local_name(a.tag) == "a")
    if expected_toc != actual_toc:
        errors.append("EPUB table of contents differs from Markdown headings: " + str(difference(expected_toc, actual_toc)))
    return {"documents": len(found), "tokens": total_tokens, "toc_entries": len(actual_toc)}
