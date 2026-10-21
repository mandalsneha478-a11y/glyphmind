"""
vector_store.py — a compact, dependency-free, namespaced dense index.

Why not FAISS / Chroma?  Because the corpora here are thousands of vectors, not
billions, and an exact brute-force cosine scan over a contiguous float32 matrix
is both faster (no index build, no recall loss) and *fully introspectable* —
which matters, because the UI displays the real per-stream contribution of
every single retrieved item.  Nothing is approximated and nothing is hidden.

Namespaces keep the three subsystems isolated inside one store:
    diagram:<doc_id>   spatial chunks of an indexed blueprint
    components         ASCII/ANSI art components for the Architect
    lexicon            semantic word clusters for the Poetry engine
    shapes             calligram structural formulas
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._lock = threading.RLock()
        self._vecs: Dict[str, np.ndarray] = {}            # ns -> (N, dim)
        self._ids: Dict[str, List[str]] = {}
        self._payloads: Dict[str, List[dict]] = {}
        self._pos: Dict[str, Dict[str, int]] = {}

    # -- mutation ----------------------------------------------------------
    def upsert(self, namespace: str, ids: Sequence[str], vectors: np.ndarray,
               payloads: Sequence[dict]) -> None:
        if len(ids) == 0:
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: got {vectors.shape[1]}, want {self.dim}")
        with self._lock:
            if namespace not in self._vecs:
                self._vecs[namespace] = np.zeros((0, self.dim), dtype=np.float32)
                self._ids[namespace] = []
                self._payloads[namespace] = []
                self._pos[namespace] = {}
            keep_v, keep_i, keep_p = [], [], []
            pos = self._pos[namespace]
            for i, (cid, vec, pl) in enumerate(zip(ids, vectors, payloads)):
                if cid in pos:
                    self._vecs[namespace][pos[cid]] = vec
                    self._payloads[namespace][pos[cid]] = dict(pl)
                else:
                    keep_v.append(vec); keep_i.append(cid); keep_p.append(dict(pl))
            if keep_v:
                start = len(self._ids[namespace])
                self._vecs[namespace] = np.vstack(
                    [self._vecs[namespace], np.stack(keep_v)]).astype(np.float32)
                self._ids[namespace].extend(keep_i)
                self._payloads[namespace].extend(keep_p)
                for off, cid in enumerate(keep_i):
                    pos[cid] = start + off

    def drop(self, namespace: str) -> None:
        with self._lock:
            self._vecs.pop(namespace, None)
            self._ids.pop(namespace, None)
            self._payloads.pop(namespace, None)
            self._pos.pop(namespace, None)

    # -- query -------------------------------------------------------------
    def search(self, namespace: str, query: np.ndarray, k: int = 10, *,
               where: Optional[callable] = None) -> List[Tuple[str, float, dict]]:
        with self._lock:
            mat = self._vecs.get(namespace)
            if mat is None or mat.shape[0] == 0:
                return []
            ids = self._ids[namespace]
            pls = self._payloads[namespace]
            q = np.asarray(query, dtype=np.float32).reshape(-1)
            qn = np.linalg.norm(q) or 1.0
            mn = np.linalg.norm(mat, axis=1)
            mn[mn == 0] = 1.0
            sims = (mat @ q) / (mn * qn)
            order = np.argsort(-sims)
            out: List[Tuple[str, float, dict]] = []
            for idx in order:
                if where is not None and not where(pls[idx]):
                    continue
                out.append((ids[idx], float(sims[idx]), pls[idx]))
                if len(out) >= k:
                    break
            return out

    def search_split(self, namespace: str, query: np.ndarray, k: int,
                     sem_dim: int) -> List[Tuple[str, float, float, float, dict]]:
        """Like `search`, but also returns the semantic and spatial components
        of every score, which is what the Observatory panel visualises."""
        with self._lock:
            mat = self._vecs.get(namespace)
            if mat is None or mat.shape[0] == 0:
                return []
            ids = self._ids[namespace]; pls = self._payloads[namespace]
            q = np.asarray(query, dtype=np.float32).reshape(-1)
            qn = np.linalg.norm(q) or 1.0
            mn = np.linalg.norm(mat, axis=1); mn[mn == 0] = 1.0
            total = (mat @ q) / (mn * qn)
            sem = (mat[:, :sem_dim] @ q[:sem_dim]) / (mn * qn)
            spa = (mat[:, sem_dim:] @ q[sem_dim:]) / (mn * qn)
            order = np.argsort(-total)[:k]
            return [(ids[i], float(total[i]), float(sem[i]), float(spa[i]), pls[i])
                    for i in order]

    # -- access ------------------------------------------------------------
    def get(self, namespace: str, cid: str) -> Optional[dict]:
        with self._lock:
            pos = self._pos.get(namespace, {})
            if cid not in pos:
                return None
            return self._payloads[namespace][pos[cid]]

    def vector_of(self, namespace: str, cid: str) -> Optional[np.ndarray]:
        with self._lock:
            pos = self._pos.get(namespace, {})
            if cid not in pos:
                return None
            return self._vecs[namespace][pos[cid]].copy()

    def all_payloads(self, namespace: str) -> List[dict]:
        with self._lock:
            return list(self._payloads.get(namespace, []))

    def all_ids(self, namespace: str) -> List[str]:
        with self._lock:
            return list(self._ids.get(namespace, []))

    def namespaces(self) -> List[str]:
        with self._lock:
            return sorted(self._vecs.keys())

    def size(self, namespace: Optional[str] = None) -> int:
        with self._lock:
            if namespace:
                return len(self._ids.get(namespace, []))
            return sum(len(v) for v in self._ids.values())

    # -- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        with self._lock:
            meta = {"dim": self.dim, "namespaces": list(self._vecs.keys())}
            (path / "meta.json").write_text(json.dumps(meta))
            for ns in self._vecs:
                safe = ns.replace("/", "__").replace(":", "--")
                np.save(path / f"{safe}.vecs.npy", self._vecs[ns])
                (path / f"{safe}.payload.json").write_text(json.dumps(
                    {"ids": self._ids[ns], "payloads": self._payloads[ns]}))

    def load(self, path: Path) -> bool:
        path = Path(path)
        mp = path / "meta.json"
        if not mp.exists():
            return False
        try:
            meta = json.loads(mp.read_text())
            if int(meta.get("dim", -1)) != self.dim:
                return False                      # encoder changed -> reindex
            with self._lock:
                for ns in meta.get("namespaces", []):
                    safe = ns.replace("/", "__").replace(":", "--")
                    vp = path / f"{safe}.vecs.npy"
                    pp = path / f"{safe}.payload.json"
                    if not (vp.exists() and pp.exists()):
                        continue
                    vecs = np.load(vp)
                    blob = json.loads(pp.read_text())
                    self._vecs[ns] = vecs.astype(np.float32)
                    self._ids[ns] = blob["ids"]
                    self._payloads[ns] = blob["payloads"]
                    self._pos[ns] = {cid: i for i, cid in enumerate(blob["ids"])}
            return True
        except Exception:
            return False


# ==========================================================================
#  Lexical BM25 — the sparse half of the hybrid
# ==========================================================================
import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.72) -> None:
        self.k1 = k1; self.b = b
        self.docs: Dict[str, List[Tuple[List[str], Counter, dict]]] = {}
        self.df: Dict[str, Counter] = {}
        self.avglen: Dict[str, float] = {}
        self.ids: Dict[str, List[str]] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return _TOK.findall((text or "").lower())

    def index(self, namespace: str, ids: Sequence[str], texts: Sequence[str],
              payloads: Sequence[dict]) -> None:
        docs = []
        df = Counter()
        total = 0
        for t in texts:
            toks = self.tokenize(t)
            total += len(toks)
            tf = Counter(toks)
            docs.append((toks, tf, {}))
            for w in tf:
                df[w] += 1
        self.docs[namespace] = [(d[0], d[1], dict(p)) for d, p in zip(docs, payloads)]
        self.df[namespace] = df
        self.ids[namespace] = list(ids)
        self.avglen[namespace] = (total / len(docs)) if docs else 1.0

    def search(self, namespace: str, query: str, k: int = 10
               ) -> List[Tuple[str, float, dict]]:
        docs = self.docs.get(namespace)
        if not docs:
            return []
        q = self.tokenize(query)
        if not q:
            return []
        N = len(docs)
        df = self.df[namespace]
        avg = self.avglen[namespace] or 1.0
        scores = np.zeros(N, dtype=np.float32)
        for w in set(q):
            n = df.get(w, 0)
            if n == 0:
                continue
            idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
            for i, (toks, tf, _) in enumerate(docs):
                f = tf.get(w, 0)
                if not f:
                    continue
                dl = len(toks) or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / avg)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        order = np.argsort(-scores)[:k]
        ids = self.ids[namespace]
        return [(ids[i], float(scores[i]), docs[i][2]) for i in order if scores[i] > 0]
