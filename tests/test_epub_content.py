import base64
from pathlib import Path
import tempfile
import unittest
import zipfile
from xml.etree import ElementTree as ET

from scripts.epub_content import (canonical_body, difference, stamp_mermaid,
                                 add_legacy_cover_metadata, annotate_document)


def tokens(markup, image_reader=lambda _: b'image'):
    return canonical_body(ET.fromstring('<body>'+markup+'</body>'), image_reader, lambda href: href)


class ContentComparisonTests(unittest.TestCase):
    def test_source_marker_and_xml_safe_header_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / 'book/01-test/README.md'
            source.parent.mkdir(parents=True)
            source.write_text('# 标题\n')
            node = {'t': 'Header', 'c': [1, ['标题', [], []], [{'t': 'Str', 'c': '标题'}]]}
            mapping = annotate_document(node, source, root, 'a' * 40)
            self.assertEqual(mapping['标题'], 'chapter-01-test--标题')
            self.assertEqual(dict(node['c'][1][2])['data-source-path'], 'book/01-test/README.md')

    def test_legacy_cover_metadata_preserves_epub_container_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'book.epub'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
                z.writestr('META-INF/container.xml', '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="EPUB/content.opf"/></rootfiles></container>')
                z.writestr('EPUB/content.opf', '<package xmlns="http://www.idpf.org/2007/opf"><metadata/><manifest><item id="picture" href="cover.png" properties="cover-image" media-type="image/png"/></manifest></package>')
                z.writestr('EPUB/cover.png', b'unchanged image')
            add_legacy_cover_metadata(path)
            with zipfile.ZipFile(path) as z:
                self.assertEqual(z.infolist()[0].filename, 'mimetype')
                self.assertEqual(z.infolist()[0].compress_type, zipfile.ZIP_STORED)
                self.assertEqual(z.read('EPUB/cover.png'), b'unchanged image')
                opf = ET.fromstring(z.read('EPUB/content.opf'))
                self.assertEqual(opf.find('.//{http://www.idpf.org/2007/opf}meta').attrib,
                                 {'name': 'cover', 'content': 'picture'})

    def test_layout_wrappers_and_prose_line_wrap_do_not_change_content(self):
        a = tokens('<h1>书名</h1><p>before\n<strong>bold</strong> after</p>')
        b = tokens('<section><h1 id="generated">书名</h1><p>before <strong>bold</strong> after</p></section>')
        self.assertIsNone(difference(a, b))

    def test_same_heading_and_paragraph_count_cannot_hide_changed_words(self):
        self.assertIsNotNone(difference(tokens('<h1>书</h1><p>正确结论</p>'),
                                        tokens('<h1>书</h1><p>错误结论</p>')))

    def test_missing_paragraph_is_detected(self):
        self.assertIsNotNone(difference(tokens('<p>A</p><p>B</p>'), tokens('<p>A</p>')))

    def test_table_value_change_is_detected(self):
        template = '<table><tbody><tr><td>{}</td></tr></tbody></table>'
        self.assertIsNotNone(difference(tokens(template.format('1024')), tokens(template.format('2048'))))

    def test_code_indentation_is_not_normalized_away(self):
        self.assertIsNotNone(difference(tokens('<pre><code>if x:\n    run()</code></pre>'),
                                        tokens('<pre><code>if x:\nrun()</code></pre>')))

    def test_formula_order_matters_even_when_counts_match(self):
        a = '<p><img class="math-inline" src="x" data-tex="x"/><img class="math-inline" src="y" data-tex="y"/></p>'
        b = '<p><img class="math-inline" src="y" data-tex="y"/><img class="math-inline" src="x" data-tex="x"/></p>'
        self.assertIsNotNone(difference(tokens(a), tokens(b)))

    def test_link_destination_and_label_are_preserved(self):
        self.assertIsNotNone(difference(tokens('<p><a href="A">说明</a></p>'),
                                        tokens('<p><a href="B">说明</a></p>')))

    def test_mermaid_source_survives_rendering_with_xml_special_characters(self):
        source = 'flowchart LR\nA["a & b"] --> B\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'diagram.svg'
            path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"></svg>')
            stamp_mermaid(path, source)
            svg = ET.fromstring(path.read_bytes())
            self.assertEqual(base64.b64decode(svg.attrib['data-mermaid-b64']).decode(), source)
            expected = tokens('<pre class="mermaid"><code>flowchart LR\nA["a &amp; b"] --&gt; B\n</code></pre>')
            actual = tokens('<p><img class="diagram" src="diagram.svg"/></p>', lambda _: path.read_bytes())
            self.assertIsNone(difference(expected, actual))

    def test_ordinary_image_bytes_are_compared(self):
        markup = '<p><img src="image.svg" alt="原图"/></p>'
        self.assertIsNotNone(difference(tokens(markup, lambda _: b'original'),
                                        tokens(markup, lambda _: b'changed')))


if __name__ == '__main__':
    unittest.main()
