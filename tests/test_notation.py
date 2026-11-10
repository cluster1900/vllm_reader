import tempfile
import unittest
from pathlib import Path

from scripts.math_support import ROOT, check_math
from scripts.notation_census import census, code_symbols, formula_symbols, render


class FormulaSymbolsTest(unittest.TestCase):
    def test_descriptive_subscript_stays_part_of_the_symbol(self):
        symbols = formula_symbols(r"B_{\mathrm{token}}=L(H_kD_k+H_vD_v)B_{\mathrm{dtype}}")
        self.assertEqual(symbols["B_{token}"], 1)
        self.assertEqual(symbols["B_{dtype}"], 1)
        self.assertEqual(symbols["H_k"], 1)
        self.assertEqual(symbols["D_v"], 1)
        self.assertNotIn("B", symbols)

    def test_named_operators_are_whole_and_take_their_scripts(self):
        symbols = formula_symbols(r"\mathrm{score}^{\mathrm{base}}_i=\max(CI_i,W_i)")
        self.assertEqual(symbols["\\mathrm{score}"], 1)
        self.assertEqual(symbols["C"], 1)
        self.assertEqual(symbols["I_i"], 1)
        for stray in ("b", "a", "s", "e"):
            self.assertNotIn(stray, symbols)

    def test_environments_do_not_leak_letters(self):
        symbols = formula_symbols(r"\begin{aligned} a_i &= \operatorname{LSE}\\ x \end{aligned}")
        self.assertEqual(set(symbols), {"a_i", "x", "\\operatorname{LSE}"})


class CodeSymbolsTest(unittest.TestCase):
    def test_finds_symbol_like_spans_outside_fences(self):
        text = "形状 `[B,S,Hq,Dh]`，温度 `T`，路径 `vllm/config.py`。\n```text\n`X`\n```\n再看 `H_kv`。\n"
        self.assertEqual(code_symbols(text), [(1, "[B,S,Hq,Dh]"), (1, "T"), (5, "H_kv")])

    def test_pure_number_lists_are_not_symbols(self):
        self.assertEqual(code_symbols("块表 `[2, 5, 3]`\n"), [])


class CensusTest(unittest.TestCase):
    def test_counts_formulas_per_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chapter = root / "book/01-demo"
            chapter.mkdir(parents=True)
            (chapter / "README.md").write_text(
                "设 $T$ 为 token 数，温度也写 $T$。\n\n$$\np_i=\\frac{e^{z_i}}{\\sum_j e^{z_j}}\n$$\n",
                encoding="utf-8",
            )
            math, code, totals = census(root, include_code=True)
        self.assertEqual(totals["01"], 3)
        self.assertEqual(math["T"]["01"], 2)
        self.assertEqual(math["p_i"]["01"], 1)
        self.assertEqual(code, {})
        rows = render(math)
        self.assertEqual(rows[0], "| 符号 | 次数 | 分章 |")
        self.assertIn("| `T` | 2 | 01:2 |", rows)


class NotationGuideTest(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / "glossary/notation.md").read_text(encoding="utf-8")

    def test_formulas_parse(self):
        self.assertEqual(check_math(self.text), [])

    def test_has_every_required_section(self):
        for heading in ("## 1 全局符号表", "## 2 命名规则", "## 3 局部字母表", "## 4 改名对照表",
                        "## 5 写法规则", "## 6 公式讲解模板", "## 7 配图规则",
                        "## 8 B1–B8 的优先清单", "## 9 共享文件与改动权限"):
            self.assertIn(heading, self.text)

    def test_template_requires_how_it_is_computed(self):
        template = self.text.split("## 6 公式讲解模板", 1)[1].split("## 7 配图规则", 1)[0]
        self.assertIn("**怎么算出来：**", template)
        self.assertIn("必写项", template)
        self.assertIn("推导", template)
        self.assertIn("算例", template)

    def test_priority_list_names_all_27_groups(self):
        section = self.text.split("## 8 B1–B8 的优先清单", 1)[1].split("## 9 ", 1)[0]
        rows = [line for line in section.splitlines() if line.startswith("| B")]
        self.assertEqual(len(rows), 27)

    def test_diagram_rule_covers_required_and_optional(self):
        rule = self.text.split("## 7 配图规则", 1)[1].split("## 8 ", 1)[0]
        self.assertIn("必配的 10 处", rule)
        self.assertIn("review 提出的要求", rule)
        self.assertIn("每个批次 PR 的正文必须列出做了哪些可选位置及原因", rule)


if __name__ == "__main__":
    unittest.main()
