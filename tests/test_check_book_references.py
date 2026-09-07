import tempfile
import unittest
from pathlib import Path

from scripts.check_book import check_explicit_references, check_source_path_references


class ExplicitReferenceCheckTest(unittest.TestCase):
    def test_checks_upstream_and_reader_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            reader_root = root / "reader"
            (source_root / "vllm").mkdir(parents=True)
            (reader_root / "tests").mkdir(parents=True)
            (source_root / "vllm" / "core.py").write_text("pass\n")
            (reader_root / "tests" / "reader_test.py").write_text("pass\n")

            texts = {
                "01-transformer-inference": "\n".join(
                    [
                        "[源码] `vllm/core.py` - `EngineCore`",
                        "[测试] `tests/reader_test.py` - reader regression",
                        "[设计] `docs/missing.md` - missing",
                        "[源码] `../escape.py` - unsafe",
                    ]
                )
            }
            errors: list[str] = []

            count = check_explicit_references(
                texts, source_root, errors, reader_root=reader_root
            )

            self.assertEqual(count, 4)
            self.assertEqual(len(errors), 2)
            self.assertIn("missing explicit reference docs/missing.md", errors[0])
            self.assertIn("unsafe explicit reference '../escape.py'", errors[1])

    def test_checks_untagged_source_paths_with_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "source"
            reader_root = root / "reader"
            (source_root / "vllm").mkdir(parents=True)
            (source_root / "vllm" / "core.py").write_text("pass\n")

            texts = {
                "03-engine-core": "See `vllm/core.py::EngineCore` and "
                "`tests/missing.py:test_case`."
            }
            errors: list[str] = []

            count = check_source_path_references(
                texts, source_root, errors, reader_root=reader_root
            )

            self.assertEqual(count, 2)
            self.assertEqual(
                errors,
                ["book/03-engine-core/README.md: missing source-like path tests/missing.py"],
            )


if __name__ == "__main__":
    unittest.main()
