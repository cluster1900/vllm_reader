import unittest

from scripts.enrich_pedagogy import (
    add_diagram_guides,
    add_section_leads,
    diagram_guide,
    heading_topic,
    section_lead,
)


class PedagogyTextTests(unittest.TestCase):
    def test_reviewed_section_lead_is_preserved(self):
        source = "## 入口\n\n> **本节先看：** 默认走多进程。\n\n- 分支\n"
        self.assertEqual(add_section_leads(source), source)

    def test_reviewed_diagram_guide_is_preserved(self):
        source = (
            "## 调用\n\n```mermaid\nflowchart LR\nA --> B\n```\n\n"
            "> **读图方法：** 箭头是 EngineCore 派发，不是 Scheduler 直接调用。\n"
        )
        self.assertEqual(add_diagram_guides(source), source)

    def test_heading_topic_removes_number_and_decorative_quotes(self):
        self.assertEqual(
            heading_topic('### 8.1.1 “GPU utilization 低”不只说明算子慢'),
            "GPU utilization 低不只说明算子慢",
        )

    def test_special_section_has_specific_guidance(self):
        lead = section_lead("## 阅读目标", "list")
        self.assertIn("能力检查", lead)
        self.assertNotIn("参数名", lead)

    def test_flowchart_guide_explains_direction(self):
        guide = diagram_guide("flowchart LR\nA --> B", "## 1.1 主路径", 1)
        self.assertIn("从左向右", guide)
        self.assertIn("主路径", guide)

    def test_sequence_guide_explains_time_axis(self):
        guide = diagram_guide(
            "sequenceDiagram\nA->>B: call", "### 一次请求", 1
        )
        self.assertIn("从上到下", guide)
        self.assertIn("参与者", guide)

    def test_section_enrichment_is_idempotent(self):
        source = "## 1.1 总体流程\n\n### 第一步\n\n正文。\n"
        once = add_section_leads(source)
        self.assertEqual(add_section_leads(once), once)
        self.assertEqual(once.count("**本节先看：**"), 1)

    def test_diagram_enrichment_is_idempotent(self):
        source = "## 1.1 流程\n\n```mermaid\nflowchart TD\nA --> B\n```\n"
        once = add_diagram_guides(source)
        self.assertEqual(add_diagram_guides(once), once)
        self.assertEqual(once.count("**读图方法：**"), 1)


if __name__ == "__main__":
    unittest.main()
