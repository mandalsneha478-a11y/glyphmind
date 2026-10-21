"""
assets.py — the generative-side retrieval index.

The Reader subsystem retrieves *chunks of a drawing*.  The two generative
subsystems retrieve *parts to build a drawing out of*: frames, textures,
motifs, palettes, layouts, fonts, shape formulas and lexical clusters.

Same machinery, different corpus.  Each asset is embedded from its name + tags
+ kind, and retrieved by cosine similarity with an optional kind filter and an
MMR pass so a prompt about "ocean" doesn't return five near-identical wave
textures and nothing else.
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedder import encode_semantic, l2
from .vector_store import BM25


class AssetIndex:
    """Semantic + lexical retrieval over heterogeneous art assets."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.assets: List[Dict] = []
        self.by_id: Dict[str, Dict] = {}
        self.vecs: Optional[np.ndarray] = None
        self.bm25 = BM25()
        self._built = False

    def add_all(self, assets: Sequence[Dict]) -> None:
        with self._lock:
            for a in assets:
                if a["id"] in self.by_id:
                    continue
                self.by_id[a["id"]] = a
                self.assets.append(a)
            self._built = False

    def build(self) -> None:
        with self._lock:
            if self._built or not self.assets:
                return
            caps = [self._caption(a) for a in self.assets]
            self.vecs = l2(encode_semantic(caps))
            self.bm25.index("assets", [a["id"] for a in self.assets], caps,
                            [{"id": a["id"], "kind": a["kind"]} for a in self.assets])
            self._built = True

    @staticmethod
    def _caption(a: Dict) -> str:
        bits = [a.get("kind", ""), a.get("name", ""), a.get("tags", "")]
        if a.get("layer"):
            bits.append(f"{a['layer']} layer scene element")
        if a.get("w"):
            bits.append(f"{a['h']} rows by {a['w']} columns")
        if a.get("ramp"):
            bits.append("shading ramp " + a["ramp"].strip())
        if a.get("slots"):
            bits.append("layout with slots " + ", ".join(s["role"] for s in a["slots"]))
        return " · ".join(b for b in bits if b)

    # -- retrieval ---------------------------------------------------------
    def search(self, query: str, *, kind: Optional[str] = None, k: int = 5,
               mmr: float = 0.35) -> List[Tuple[Dict, float]]:
        self.build()
        if self.vecs is None or not self.assets:
            return []
        q = l2(encode_semantic([query]))[0]
        sims = self.vecs @ q

        lex = {cid: s for cid, s, _ in self.bm25.search("assets", query, 40)}
        if lex:
            mx = max(lex.values()) or 1.0
            for i, a in enumerate(self.assets):
                sims[i] = 0.78 * sims[i] + 0.22 * (lex.get(a["id"], 0.0) / mx)

        idx = [i for i, a in enumerate(self.assets) if kind is None or a["kind"] == kind]
        if not idx:
            return []
        idx.sort(key=lambda i: -sims[i])

        # Maximal-marginal-relevance so results stay diverse.
        chosen: List[int] = []
        while idx and len(chosen) < k:
            if not chosen:
                chosen.append(idx.pop(0)); continue
            best_i, best_s = None, -1e9
            for i in idx[:24]:
                div = max(float(self.vecs[i] @ self.vecs[j]) for j in chosen)
                s = (1 - mmr) * float(sims[i]) - mmr * div
                if s > best_s:
                    best_s, best_i = s, i
            if best_i is None:
                break
            chosen.append(best_i); idx.remove(best_i)
        return [(self.assets[i], float(sims[i])) for i in chosen]

    def best(self, query: str, kind: str, fallback_id: Optional[str] = None
             ) -> Tuple[Dict, float]:
        res = self.search(query, kind=kind, k=1)
        if res:
            return res[0]
        if fallback_id and fallback_id in self.by_id:
            return self.by_id[fallback_id], 0.0
        pool = [a for a in self.assets if a["kind"] == kind]
        return (pool[0] if pool else {}), 0.0

    def get(self, aid: str) -> Optional[Dict]:
        return self.by_id.get(aid)

    def count(self, kind: Optional[str] = None) -> int:
        if kind is None:
            return len(self.assets)
        return sum(1 for a in self.assets if a["kind"] == kind)

    def kinds(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.assets:
            out[a["kind"]] = out.get(a["kind"], 0) + 1
        return out


_index: Optional[AssetIndex] = None
_lock = threading.Lock()


def get_asset_index() -> AssetIndex:
    """Singleton built from every generative corpus in the project."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is not None:
            return _index
        idx = AssetIndex()
        try:
            from ..corpus.components import all_components
            from ..corpus.shapes import all_shapes
            from ..corpus.lexicon import all_lexicon
            from ..corpus.scene_kb import all_scene_elements
        except ImportError:                                # pragma: no cover
            from corpus.components import all_components
            from corpus.shapes import all_shapes
            from corpus.lexicon import all_lexicon
            from corpus.scene_kb import all_scene_elements
        idx.add_all(all_components())
        idx.add_all(all_shapes())
        idx.add_all(all_lexicon())
        idx.add_all(all_scene_elements())
        idx.build()
        _index = idx
    return _index
