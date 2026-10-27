import unittest
from unittest.mock import MagicMock, patch

from runtime_lifecycle import (
    activate_embedding_runtime,
    activate_qa_runtimes,
    activate_vgen_runtime,
    release_all_model_memory,
)


class RuntimeLifecycleTest(unittest.TestCase):
    def test_activate_vgen_releases_embedding_before_prepare(self):
        events = []
        runtime = MagicMock()
        runtime.prepare.side_effect = lambda: events.append("prepare-vgen")

        with patch(
            "runtime_lifecycle.release_embedding_runtime",
            side_effect=lambda: events.append("release-embedding"),
        ), patch("runtime_lifecycle.get_vgen_runtime", return_value=runtime):
            result = activate_vgen_runtime()

        self.assertIs(result, runtime)
        self.assertEqual(events, ["release-embedding", "prepare-vgen"])

    def test_activate_embedding_releases_vgen_before_prepare(self):
        events = []
        runtime = MagicMock()
        runtime.prepare.side_effect = lambda: events.append("prepare-embedding")

        with patch(
            "runtime_lifecycle.release_vgen_runtime",
            side_effect=lambda: events.append("release-vgen"),
        ), patch("runtime_lifecycle.get_embedding_runtime", return_value=runtime):
            result = activate_embedding_runtime()

        self.assertIs(result, runtime)
        self.assertEqual(events, ["release-vgen", "prepare-embedding"])

    def test_activate_qa_prepares_both_without_releasing_either(self):
        events = []
        vgen_runtime = MagicMock()
        embedding_runtime = MagicMock()
        vgen_runtime.prepare.side_effect = lambda: events.append("prepare-vgen")
        embedding_runtime.prepare.side_effect = lambda: events.append(
            "prepare-embedding"
        )

        with patch(
            "runtime_lifecycle.get_vgen_runtime",
            return_value=vgen_runtime,
        ), patch(
            "runtime_lifecycle.get_embedding_runtime",
            return_value=embedding_runtime,
        ), patch(
            "runtime_lifecycle.release_vgen_runtime"
        ) as release_vgen, patch(
            "runtime_lifecycle.release_embedding_runtime"
        ) as release_embedding:
            result = activate_qa_runtimes()

        self.assertEqual(result, (vgen_runtime, embedding_runtime))
        self.assertEqual(events, ["prepare-vgen", "prepare-embedding"])
        release_vgen.assert_not_called()
        release_embedding.assert_not_called()

    def test_release_all_releases_both_models(self):
        with patch("runtime_lifecycle.release_vgen_runtime") as release_vgen, patch(
            "runtime_lifecycle.release_embedding_runtime"
        ) as release_embedding:
            release_all_model_memory()

        release_vgen.assert_called_once()
        release_embedding.assert_called_once()


if __name__ == "__main__":
    unittest.main()
