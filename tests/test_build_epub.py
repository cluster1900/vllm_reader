import unittest

from scripts.build_epub import neutralize_repo_links


class NeutralizeRepoLinksTests(unittest.TestCase):
    def test_local_repository_link_becomes_text(self):
        source = "查看 [实现](../vllm/engine/core.py)。"
        self.assertEqual(
            neutralize_repo_links(source),
            "查看 实现（随书仓库：`../vllm/engine/core.py`）。",
        )

    def test_image_is_preserved(self):
        source = "![调度流程](media/scheduler.svg){.diagram}"
        self.assertEqual(neutralize_repo_links(source), source)

    def test_unmatched_bracket_cannot_consume_later_image(self):
        source = "stage 0: [0, 3)\n\n![流水线切分](media/pipeline.svg){.diagram}"
        self.assertEqual(neutralize_repo_links(source), source)


if __name__ == "__main__":
    unittest.main()
