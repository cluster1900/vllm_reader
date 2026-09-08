import tempfile
import unittest
from pathlib import Path

from scripts.epub_content import rewrite_links


class SourceLinkTests(unittest.TestCase):
    def test_repository_link_preserves_label_and_pins_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'book/chapter/README.md'
            source.parent.mkdir(parents=True)
            target = root / 'examples/demo.py'
            target.parent.mkdir()
            target.write_text('pass\n')
            label = [{'t': 'Code', 'c': [['', [], []], 'demo()']}]
            node = {'t': 'Link', 'c': [['', [], []], label, ['../../examples/demo.py', '']]}
            rewrite_links(node, source, root, 'a' * 40, {})
            self.assertEqual(node['c'][1], label)
            self.assertEqual(node['c'][2][0],
                             'https://github.com/cluster1900/vllm_reader/blob/' + 'a' * 40 + '/examples/demo.py')

    def test_internal_link_uses_embedded_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'book/a/README.md'
            target = root / 'book/b/README.md'
            target.parent.mkdir(parents=True)
            target.touch()
            node = {'t': 'Link', 'c': [['', [], []], [{'t': 'Str', 'c': '原文字'}], ['../b/#details', '']]}
            rewrite_links(node, source, root, 'a' * 40, {target: {'title': 'b--title', 'details': 'b--details'}})
            self.assertEqual(node['c'][2][0], '#b--details')

    def test_code_and_images_are_not_rewritten_as_links(self):
        doc = [{'t': 'Code', 'c': [['', [], []], '[x](../demo.py)']},
               {'t': 'Image', 'c': [['', [], []], [{'t': 'Str', 'c': '图'}], ['media/a.svg', '']]}]
        before = repr(doc)
        rewrite_links(doc, Path('/book/a.md'), Path('/book'), 'a' * 40, {})
        self.assertEqual(repr(doc), before)


if __name__ == '__main__':
    unittest.main()
