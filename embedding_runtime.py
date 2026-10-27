import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from device_memory import release_device_memory
from config import EMBEDDING_MODEL_PATH


_EMBEDDING_RUNTIME = None
_EMBEDDING_RUNTIME_LOCK = threading.Lock()


class E5EmbeddingRuntime:
    def __init__(self, model_path=EMBEDDING_MODEL_PATH, max_length=512, batch_size=4):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.model_path = str(Path(model_path))
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.device = self._select_device()
        self._lock = threading.RLock()

    def embed_query(self, query):
        return self.embed_texts([query], prefix="query: ")[0]

    def embed_passages(self, passages):
        return self.embed_texts(passages, prefix="passage: ")

    def embed_texts(self, texts, prefix):
        if not texts:
            return []

        normalized = [f"{prefix}{self._normalize_text(text)}" for text in texts]
        with self._lock:
            self._ensure_model()
            results = []
            with torch.inference_mode():
                for start in range(0, len(normalized), self.batch_size):
                    results.extend(
                        self._embed_batch(normalized[start : start + self.batch_size])
                    )
            return results

    def prepare(self):
        with self._lock:
            self._ensure_model()

    def reset_model(self):
        with self._lock:
            model = self.model
            tokenizer = self.tokenizer
            self.model = None
            self.tokenizer = None
            del model, tokenizer
            release_device_memory(self.device)

    def _embed_batch(self, texts):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = self.model(**batch)
        embeddings = self._mean_pool(
            outputs.last_hidden_state,
            batch["attention_mask"],
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        result = embeddings.detach().cpu().tolist()
        del batch, outputs, embeddings
        return result

    def _ensure_model(self):
        if self.model is not None and self.tokenizer is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model.eval()
        if self.device.type != "cpu":
            self.model.to(self.device)

    def _mean_pool(self, token_embeddings, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _select_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _normalize_text(self, text):
        return " ".join(str(text or "").split())


def get_embedding_runtime(model_path=EMBEDDING_MODEL_PATH):
    global _EMBEDDING_RUNTIME
    normalized_model_path = str(Path(model_path))
    with _EMBEDDING_RUNTIME_LOCK:
        if (
            _EMBEDDING_RUNTIME is None
            or _EMBEDDING_RUNTIME.model_path != normalized_model_path
        ):
            _EMBEDDING_RUNTIME = E5EmbeddingRuntime(model_path=normalized_model_path)
        return _EMBEDDING_RUNTIME


def release_embedding_runtime():
    with _EMBEDDING_RUNTIME_LOCK:
        if _EMBEDDING_RUNTIME is not None:
            _EMBEDDING_RUNTIME.reset_model()


def reset_embedding_runtime():
    global _EMBEDDING_RUNTIME
    with _EMBEDDING_RUNTIME_LOCK:
        if _EMBEDDING_RUNTIME is not None:
            _EMBEDDING_RUNTIME.reset_model()
        _EMBEDDING_RUNTIME = None
