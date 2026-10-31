"""
Sentence-transformer embedder with an on-disk cache.

Implements the harness's `TextEmbedder` protocol (`embed(texts) -> (N, D)`).
Embeddings are L2-normalized, so a dot product is cosine similarity, which is
what the clustering and retrieval steps will expect.

Encoding the ~37k CGA-CMV context comments takes a few minutes on a laptop, so
vectors are cached per model under `data/processed/embeddings/` (gitignored),
keyed by the SHA-256 of each text. Reruns only encode texts not seen before.

`sentence-transformers` is imported lazily so the rest of the package works
without torch installed.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("data/processed/embeddings")


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self.cache_path = (
            Path(cache_dir) / f"{model_name.replace('/', '__')}.npz" if cache_dir else None
        )
        if self.cache_path and self.cache_path.exists():
            with np.load(self.cache_path) as data:
                self._cache = dict(zip(data["keys"].tolist(), data["vectors"]))
            log.info("Loaded %d cached embeddings from %s", len(self._cache), self.cache_path)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerEmbedder. "
                    "Run with: uv run --with sentence-transformers ..."
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        if self._cache:
            return len(next(iter(self._cache.values())))
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [text_key(t) for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in self._cache}
        if missing:
            log.info("Encoding %d new texts with %s", len(missing), self.model_name)
            vectors = self.model.encode(
                list(missing.values()),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(missing) > 1000,
            )
            self._cache.update(zip(missing.keys(), vectors.astype(np.float32)))
            self._save()
        if not keys:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack([self._cache[k] for k in keys])

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._cache.keys()))
        vectors = np.vstack(list(self._cache.values()))
        tmp = self.cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vectors=vectors)
        tmp.replace(self.cache_path)
