import json
import base64
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from xml.etree import ElementTree as ET

from scripts.math_support import (
    ROOT, check_math, formula_counts, markdown_formulas, math_nodes,
    read_document, render_document_math,
)
from scripts.build_epub import normalize_display_math


class MarkdownMathTests(unittest.TestCase):
    def test_inline_and_display_are_distinct(self):
        formulas = markdown_formulas("单位 $B_i$。\n\n$$\n\\frac{a}{b}\n$$\n")
        self.assertEqual([(f.tex, f.display) for f in formulas],
                         [("B_i", False), (r"\frac{a}{b}", True)])

    def test_code_and_escaped_dollars_are_not_math(self):
        source = "`$CODE$` 与 ``$other$``，价格 \\$5。\n```python\nx='$fake$'\n```\n$real$"
        self.assertEqual([f.tex for f in markdown_formulas(source)], ["real"])

    def test_unclosed_math_is_reported(self):
        self.assertIn("unclosed", check_math("解释 $x + y")[0])

    def test_display_delimiters_require_separate_lines(self):
        self.assertIn("own line", check_math("$$x+y$$")[0])

    def test_inline_math_cannot_eat_multiple_paragraphs(self):
        self.assertIn("one line", check_math("$x\n\n另一段$ ")[0])

    def test_build_no_longer_silently_fixes_math_code_fences(self):
        with self.assertRaisesRegex(ValueError, "code fence"):
            normalize_display_math("```math\nx+y\n```\n")
        native = "$$\nx+y\n$$\n"
        self.assertEqual(normalize_display_math(native), native)


@unittest.skipUnless(shutil.which("pandoc") and shutil.which("node") and
                     (ROOT / "node_modules/mathjax-full").exists(),
                     "requires installed Pandoc and Node dependencies")
class MathBuildTests(unittest.TestCase):
    def test_matrix_formula_is_parsed_and_code_remains_code(self):
        source = "$$\n\\begin{bmatrix}a & b \\\\ c & d\\end{bmatrix} = \\begin{bmatrix}e & f \\\\ g & h\\end{bmatrix}\n$$\n\n`$keep$`\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "chapter.md"
            p.write_text(source)
            doc = read_document([p])
            self.assertEqual(len(list(math_nodes(doc))), 1)
            formulas = render_document_math(doc, Path(tmp))
            self.assertEqual(len(formulas), 1)
            self.assertEqual(len(list(math_nodes(doc))), 0)
            self.assertIn('"Code"', json.dumps(doc))
            image = list((Path(tmp) / "media/math").glob("*.svg"))[0].read_text()
            svg = ET.fromstring(image)
            self.assertIn("&", base64.b64decode(svg.attrib["data-tex-b64"]).decode())
            self.assertIn("<path", image)
            self.assertNotIn("<text", image)
            self.assertNotIn("<use", image)

    def test_inline_baseline_and_tex_survive_ast_conversion(self):
        doc = {"blocks": [{"t": "Para", "c": [{"t": "Math", "c": [
            {"t": "InlineMath"}, r"\frac{a}{b}"]}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            render_document_math(doc, Path(tmp))
        image = doc["blocks"][0]["c"][0]
        attrs = dict(image["c"][0][2])
        self.assertEqual(attrs["data-tex"], r"\frac{a}{b}")
        self.assertIn("vertical-align:", attrs["style"])
        self.assertIn("math-inline", image["c"][0][1])

    def test_bad_tex_fails_instead_of_rendering_plaintext(self):
        doc = {"blocks": [{"t": "Math", "c": [
            {"t": "DisplayMath"}, r"\nonexistentmacro{x}"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as raised:
                render_document_math(doc, Path(tmp))
        self.assertIn("Undefined control sequence", raised.exception.stderr)

    def test_each_formula_occurrence_is_retained_even_when_images_are_shared(self):
        doc = {"blocks": [{"t": "Math", "c": [{"t": "InlineMath"}, "x"]},
                          {"t": "Math", "c": [{"t": "InlineMath"}, "x"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            entries = render_document_math(doc, Path(tmp))
            self.assertEqual(len(entries), 2)
            self.assertEqual(len(list((Path(tmp) / "media/math").glob("*.svg"))), 1)
        self.assertEqual(formula_counts(["$x$ $x$"])[(False, "x")], 2)


if __name__ == "__main__":
    unittest.main()
