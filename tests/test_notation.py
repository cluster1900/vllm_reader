import re
import tempfile
import unittest
from pathlib import Path

from scripts.math_support import ROOT, check_math
from scripts.notation_census import WRAPPER, census, code_symbols, flatten_scripts, formula_symbols, render


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _table_rows(section: str) -> list[list[str]]:
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        rows.append(cells)
    return rows


def _normal(tex: str) -> str:
    """One canonical spelling: `\\mathrm{blk}` and `blk` alike, no spaces, no spacing commands."""
    tex = WRAPPER.sub(lambda m: m.group(1), flatten_scripts(tex))
    tex = re.sub(r"\\[,;! ]", "", tex)
    return re.sub(r"\s+", "", tex)


def _atoms(cell: str) -> list[str]:
    """Every symbol a table cell registers: each `$...$`, split at top-level commas, left of `=`."""
    atoms = []
    for tex in re.findall(r"\$([^$]+)\$", cell):
        depth, piece, pieces = 0, "", []
        for char in tex:
            depth += (char == "{") - (char == "}")
            if char == "," and depth == 0:
                pieces.append(piece)
                piece = ""
            else:
                piece += char
        pieces.append(piece)
        atoms.extend(_normal(piece.split("=", 1)[0]) for piece in pieces if piece.strip())
    return atoms


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

    def _registries(self):
        global_rows = _table_rows(_section(self.text, "## 1 全局符号表", "## 2 命名规则"))
        local_rows = _table_rows(_section(self.text, "## 3 局部字母表", "## 4 改名对照表"))
        global_atoms = [a for row in global_rows if row[0] != "符号" for a in _atoms(row[0])]
        local_atoms = [a for row in local_rows if row[0] != "章" for a in _atoms(row[1])]
        return global_atoms, local_atoms

    def test_global_symbols_are_registered_once(self):
        global_atoms, _ = self._registries()
        self.assertGreater(len(global_atoms), 100)
        duplicated = sorted({a for a in global_atoms if global_atoms.count(a) > 1})
        self.assertEqual(duplicated, [], "one letter, one meaning: registered twice in section 1")

    def test_local_letters_are_not_global_symbols(self):
        global_atoms, local_atoms = self._registries()
        self.assertGreater(len(local_atoms), 40)
        self.assertEqual(sorted(set(global_atoms) & set(local_atoms)), [], "local letter is also in section 1")

    def test_every_renamed_symbol_is_registered(self):
        """The rename table may only introduce symbols that section 1 or 3 defines (any column)."""
        def keys(text: str) -> set[str]:
            found = set()
            for tex in re.findall(r"\$([^$]+)\$", text):
                for key in formula_symbols(tex):
                    key = key.replace("\\operatorname", "\\mathrm").replace("^{ }", "")  # K^{\top} is K, transposed
                    found.add(key.replace("_{TP-1}", "_i"))  # A_{TP-1} is the last A_i
            return found

        registered = keys(_section(self.text, "## 1 全局符号表", "## 2 命名规则")) | keys(
            _section(self.text, "## 3 局部字母表", "## 4 改名对照表"))
        unregistered = {}
        for row in _table_rows(_section(self.text, "## 4 改名对照表", "## 5 写法规则")):
            if len(row) < 4 or row[0] == "旧写法":
                continue
            for key in keys(row[1]) - registered:
                unregistered[key] = row[1]
        self.assertEqual(unregistered, {}, "new notation used in section 4 but not registered in section 1 or 3")

    def test_collisions_found_in_review_stay_fixed(self):
        global_atoms, local_atoms = self._registries()
        for symbol in ("Q", "\\ell", "\\theta"):
            self.assertEqual(global_atoms.count(symbol) + local_atoms.count(symbol), 1, symbol)
        for symbol in ("b", "i"):
            self.assertNotIn(symbol, global_atoms)
        for symbol in ("N_{gen}", "N_{req}", "N_{SLO}", "N_{GPU}", "N_{users}"):
            self.assertNotIn(symbol, global_atoms + local_atoms)
        for symbol in ("N_{sch,i}", "N_{drf,i}", "N_{acc,i}", "E_{max}", "cap\\_bound", "\\overline{\\Deltat}_{sys}",
                       "\\Deltat_{clientqueue}", "n_{req}", "n_{SLO}", "n_{GPU}", "n_{users}"):
            self.assertIn(symbol, global_atoms + local_atoms)

    def test_diagram_rule_covers_required_and_optional(self):
        rule = self.text.split("## 7 配图规则", 1)[1].split("## 8 ", 1)[0]
        self.assertIn("必配的 10 处", rule)
        self.assertIn("review 提出的要求", rule)
        self.assertIn("每个批次 PR 的正文必须列出做了哪些可选位置及原因", rule)


if __name__ == "__main__":
    unittest.main()
