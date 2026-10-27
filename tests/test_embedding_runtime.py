from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from embedding_runtime import E5EmbeddingRuntime


class FakeTokenizer:
    def __init__(self):
        self.texts = None
        self.calls = []

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        self.texts = list(texts)
        self.calls.append(self.texts)
        return {
            "input_ids": torch.tensor([[1, 2] for _ in texts]),
            "attention_mask": torch.tensor([[1, 1] for _ in texts]),
        }


class E5EmbeddingRuntimeTest(unittest.TestCase):
    def setUp(self):
        device_patch = patch("embedding_runtime.E5EmbeddingRuntime._select_device", return_value=torch.device("cpu"))
        device_patch.start()
        self.addCleanup(device_patch.stop)

    def test_adds_e5_prefixes_and_returns_normalized_vectors(self):
        tokenizer = FakeTokenizer()
        model = MagicMock()
        model.return_value = SimpleNamespace(
            last_hidden_state=torch.ones((2, 2, 384), dtype=torch.float32)
        )

        with patch("embedding_runtime.AutoTokenizer") as tokenizer_cls, patch(
            "embedding_runtime.AutoModel"
        ) as model_cls:
            tokenizer_cls.from_pretrained.return_value = tokenizer
            model_cls.from_pretrained.return_value = model

            runtime = E5EmbeddingRuntime(model_path="models/embedding")
            embeddings = runtime.embed_texts(["پرسش", "متن سند"], prefix="query: ")

        self.assertEqual(tokenizer.texts, ["query: پرسش", "query: متن سند"])
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 384)
        self.assertAlmostEqual(
            sum(value * value for value in embeddings[0]),
            1.0,
            places=5,
        )
        model.eval.assert_called_once()
        model_cls.from_pretrained.assert_called_once_with(
            "models/embedding",
            local_files_only=True,
        )

    def test_embeds_texts_in_bounded_batches(self):
        tokenizer = FakeTokenizer()
        model = MagicMock()
        model.side_effect = lambda **batch: SimpleNamespace(
            last_hidden_state=torch.ones(
                (batch["input_ids"].shape[0], 2, 384),
                dtype=torch.float32,
            )
        )

        with patch("embedding_runtime.AutoTokenizer") as tokenizer_cls, patch(
            "embedding_runtime.AutoModel"
        ) as model_cls:
            tokenizer_cls.from_pretrained.return_value = tokenizer
            model_cls.from_pretrained.return_value = model

            runtime = E5EmbeddingRuntime(
                model_path="models/embedding",
                batch_size=2,
            )
            embeddings = runtime.embed_passages(["one", "two", "three", "four", "five"])

        self.assertEqual(
            tokenizer.calls,
            [
                ["passage: one", "passage: two"],
                ["passage: three", "passage: four"],
                ["passage: five"],
            ],
        )
        self.assertEqual(model.call_count, 3)
        self.assertEqual(len(embeddings), 5)

    def test_rejects_invalid_batch_size(self):
        with self.assertRaisesRegex(ValueError, "batch_size"):
            E5EmbeddingRuntime(batch_size=0)


if __name__ == "__main__":
    unittest.main()
