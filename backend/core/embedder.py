"""
embedder.py — dual-stream embeddings: semantic ⊕ spatial.

A pre-trained sentence encoder is extraordinary at "what does this text mean"
and completely blind to "where was this text".  A hand-engineered geometric
descriptor is the exact mirror image.  GLYPHMIND runs both and late-fuses them
into a single vector so that one ANN lookup retrieves on meaning *and* form.

    v(chunk) = [ Wsem · L2(SBERT(caption)) ‖ Wspa · L2(Φ(geometry)) ]

Because both halves are independently L2-normalised before weighting, the dot
product decomposes exactly:

    sim(a,b) = Wsem² · cos_sem(a,b) + Wspa² · cos_spa(a,b)

…so the contribution of each stream to any score is recoverable and is shown
live in the UI's RAG Observatory.  Retrieval here is not a black box.

Φ — the 72-dimensional spatial descriptor
-----------------------------------------
   0-3    normalised bbox (r0, c0, h, w)
   4-5    normalised centroid
   6      aspect ratio (log-scaled, squashed)
   7      area fraction of the canvas
   8      ink density
   9-19   glyph-class histogram (11 classes)
  20-35   4x4 occupancy grid (the chunk's own "thumbnail")
  36-51   4x4 structural-ink grid
  52-55   edge-contact flags (does the ink touch each border?)
  56-57   horizontal / vertical symmetry scores
  58-59   mean horizontal / vertical run length (normalised)
  60-63   directional gradient energies (↕ ↔ ╱ ╲)
  64-67   row / column profile moments
  68-71   corner, junction, arrow and shade saliency
"""
from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .canvas import BBox, Canvas, N_GLYPH_CLASSES

try:
    from ..config import (EMBED_MODEL_NAME, FORCE_OFFLINE_ENCODER,
                          SEMANTIC_DIM_FALLBACK, SEMANTIC_WEIGHT, SPATIAL_DIM,
                          SPATIAL_GRID, SPATIAL_WEIGHT)
except ImportError:                                        # pragma: no cover
    from config import (EMBED_MODEL_NAME, FORCE_OFFLINE_ENCODER,
                        SEMANTIC_DIM_FALLBACK, SEMANTIC_WEIGHT, SPATIAL_DIM,
                        SPATIAL_GRID, SPATIAL_WEIGHT)


def l2(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


# ==========================================================================
#  SEMANTIC STREAM
# ==========================================================================
class HashedEncoder:
    """Zero-dependency fallback: hashed word + char-ngram projection.

    Not as strong as a transformer, but deterministic, instant, offline, and
    good enough that the system is fully usable if the model download fails.
    """

    dim = SEMANTIC_DIM_FALLBACK
    name = "hashed-ngram-384 (offline fallback)"

    def _bucket(self, token: str) -> int:
        return int(hashlib.blake2s(token.encode(), digest_size=4).hexdigest(), 16) % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            low = (t or "").lower()
            words = re.findall(r"[a-z0-9]+", low)
            for w in words:
                out[i, self._bucket("w:" + w)] += 1.0
                for k in (3, 4):
                    for j in range(max(0, len(w) - k + 1)):
                        out[i, self._bucket("c:" + w[j:j + k])] += 0.45
            for a, b in zip(words, words[1:]):
                out[i, self._bucket("p:" + a + "_" + b)] += 0.8
        # sublinear damping then L2
        out = np.sign(out) * np.log1p(np.abs(out))
        return l2(out)


class SentenceEncoder:
    """Thin lazy wrapper around a pre-built local sentence-transformers model."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()
        self._failed = False
        self.dim = 384
        self.name = f"{model_name} (local)"

    def _load(self):
        if self._model is not None or self._failed:
            return
        with self._lock:
            if self._model is not None or self._failed:
                return
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer(self.model_name)
                self.dim = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._failed = True

    def available(self) -> bool:
        self._load()
        return self._model is not None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        assert self._model is not None
        vecs = self._model.encode(list(texts), convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  show_progress_bar=False, batch_size=64)
        return np.asarray(vecs, dtype=np.float32)


_semantic_backend = None
_semantic_lock = threading.Lock()


def semantic_backend():
    global _semantic_backend
    if _semantic_backend is not None:
        return _semantic_backend
    with _semantic_lock:
        if _semantic_backend is not None:
            return _semantic_backend
        if FORCE_OFFLINE_ENCODER:
            _semantic_backend = HashedEncoder()
        else:
            st = SentenceEncoder()
            _semantic_backend = st if st.available() else HashedEncoder()
    return _semantic_backend


def encode_semantic(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, semantic_backend().dim), dtype=np.float32)
    return semantic_backend().encode(texts)


# ==========================================================================
#  SPATIAL STREAM
# ==========================================================================
def _grid_pool(mask: np.ndarray, g: int) -> np.ndarray:
    """Average-pool a boolean mask down to a g x g grid."""
    if mask.size == 0:
        return np.zeros(g * g, dtype=np.float32)
    h, w = mask.shape
    out = np.zeros((g, g), dtype=np.float32)
    rb = np.linspace(0, h, g + 1).astype(int)
    cb = np.linspace(0, w, g + 1).astype(int)
    for i in range(g):
        for j in range(g):
            r0, r1 = rb[i], max(rb[i] + 1, rb[i + 1])
            c0, c1 = cb[j], max(cb[j] + 1, cb[j + 1])
            sub = mask[r0:min(r1, h), c0:min(c1, w)]
            out[i, j] = float(sub.mean()) if sub.size else 0.0
    return out.ravel()


def _mean_run(mask: np.ndarray, axis: int) -> float:
    if mask.size == 0:
        return 0.0
    m = mask if axis == 1 else mask.T
    runs: List[int] = []
    for row in m:
        n = 0
        for v in row:
            if v:
                n += 1
            elif n:
                runs.append(n); n = 0
        if n:
            runs.append(n)
    if not runs:
        return 0.0
    return float(np.mean(runs)) / max(1.0, m.shape[1])


def spatial_vector(canvas: Canvas, box: BBox, class_hist: Optional[Sequence[float]] = None
                   ) -> np.ndarray:
    """Φ — the hand-engineered geometric descriptor."""
    H, W = max(1, canvas.height), max(1, canvas.width)
    r0 = max(0, box.r0); c0 = max(0, box.c0)
    r1 = min(H, box.r1); c1 = min(W, box.c1)
    occ = canvas.occupancy[r0:r1, c0:c1]
    stc = canvas.structure[r0:r1, c0:c1]
    cls = canvas.classes[r0:r1, c0:c1]

    v = np.zeros(SPATIAL_DIM, dtype=np.float32)
    h, w = max(0, r1 - r0), max(0, c1 - c0)

    # 0-3 normalised bbox
    v[0] = r0 / H; v[1] = c0 / W; v[2] = h / H; v[3] = w / W
    # 4-5 centroid
    cr, cc = box.centroid
    v[4] = min(1.0, max(0.0, cr / H)); v[5] = min(1.0, max(0.0, cc / W))
    # 6 aspect ratio (log, squashed to 0..1)
    ar = (w + 1e-6) / (h + 1e-6)
    v[6] = 1.0 / (1.0 + math.exp(-math.log(ar + 1e-6)))
    # 7 area fraction
    v[7] = (h * w) / float(H * W)
    # 8 density
    v[8] = float(occ.mean()) if occ.size else 0.0
    # 9-19 glyph class histogram
    if class_hist is not None and len(class_hist) >= N_GLYPH_CLASSES:
        v[9:9 + N_GLYPH_CLASSES] = np.asarray(class_hist[:N_GLYPH_CLASSES], dtype=np.float32)
    elif cls.size:
        counts = np.bincount(cls.ravel(), minlength=N_GLYPH_CLASSES).astype(np.float32)
        v[9:9 + N_GLYPH_CLASSES] = counts / max(1.0, counts.sum())

    g = 4 if SPATIAL_GRID != 4 else SPATIAL_GRID    # layout below assumes 4x4
    base = 9 + N_GLYPH_CLASSES                      # = 20
    v[base:base + g * g] = _grid_pool(occ, g)       # 20-35
    v[base + 16:base + 32] = _grid_pool(stc, g)     # 36-51

    # 52-55 border contact
    if occ.size:
        v[52] = float(occ[0, :].mean())
        v[53] = float(occ[-1, :].mean())
        v[54] = float(occ[:, 0].mean())
        v[55] = float(occ[:, -1].mean())
    # 56-57 symmetry
    if occ.size:
        lr = float((occ == occ[:, ::-1]).mean())
        ud = float((occ == occ[::-1, :]).mean())
        v[56] = lr; v[57] = ud
    # 58-59 run lengths
    v[58] = _mean_run(occ, axis=1)
    v[59] = _mean_run(occ, axis=0)
    # 60-63 directional gradient energy
    if occ.size and occ.shape[0] > 1 and occ.shape[1] > 1:
        f = occ.astype(np.float32)
        v[60] = float(np.abs(np.diff(f, axis=0)).mean())       # vertical change
        v[61] = float(np.abs(np.diff(f, axis=1)).mean())       # horizontal change
        d1 = f[1:, 1:] - f[:-1, :-1]
        d2 = f[1:, :-1] - f[:-1, 1:]
        v[62] = float(np.abs(d1).mean())
        v[63] = float(np.abs(d2).mean())
    # 64-67 profile moments
    if occ.size:
        rp = occ.mean(axis=1); cp = occ.mean(axis=0)
        v[64] = float(rp.mean()); v[65] = float(rp.std())
        v[66] = float(cp.mean()); v[67] = float(cp.std())
    # 68-71 saliency of structurally meaningful glyph families
    if cls.size:
        n = float(cls.size)
        v[68] = float((cls == 1).sum() / n)    # corners
        v[69] = float((cls == 4).sum() / n)    # junctions
        v[70] = float((cls == 6).sum() / n)    # arrows
        v[71] = float((cls == 7).sum() / n)    # shading
    return v


# ==========================================================================
#  FUSION
# ==========================================================================
class FusedEmbedder:
    """Produces [Wsem·L2(sem) ‖ Wspa·L2(spa)] and reports stream provenance."""

    def __init__(self) -> None:
        self.sem_dim = semantic_backend().dim
        self.spa_dim = SPATIAL_DIM

    @property
    def dim(self) -> int:
        return self.sem_dim + self.spa_dim

    @property
    def backend_name(self) -> str:
        return semantic_backend().name

    def embed_chunks(self, canvas: Canvas, chunks: Sequence) -> np.ndarray:
        captions = [c.describe() for c in chunks]
        sem = encode_semantic(captions) if captions else np.zeros((0, self.sem_dim), np.float32)
        sem = l2(sem)
        spa = np.stack([spatial_vector(canvas, c.bbox, c.class_hist) for c in chunks]) \
            if chunks else np.zeros((0, self.spa_dim), np.float32)
        spa = l2(spa)
        return np.hstack([sem * SEMANTIC_WEIGHT, spa * SPATIAL_WEIGHT]).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        sem = l2(encode_semantic([text]))
        spa = np.zeros((1, self.spa_dim), dtype=np.float32)
        return np.hstack([sem * SEMANTIC_WEIGHT, spa]).astype(np.float32)[0]

    def embed_query(self, text: str, spatial_prior: Optional[np.ndarray] = None
                    ) -> np.ndarray:
        """A query may carry a *spatial prior* — e.g. 'below the load balancer'
        resolves to a concrete geometric target, which we inject into the
        spatial half so the ANN search itself becomes position-aware."""
        sem = l2(encode_semantic([text]))
        if spatial_prior is None:
            spa = np.zeros((1, self.spa_dim), dtype=np.float32)
        else:
            spa = l2(spatial_prior.reshape(1, -1).astype(np.float32))
        return np.hstack([sem * SEMANTIC_WEIGHT, spa * SPATIAL_WEIGHT]).astype(np.float32)[0]

    def split(self, v: np.ndarray):
        return v[..., :self.sem_dim], v[..., self.sem_dim:]


_fused: Optional[FusedEmbedder] = None


def get_embedder() -> FusedEmbedder:
    global _fused
    if _fused is None:
        _fused = FusedEmbedder()
    return _fused
