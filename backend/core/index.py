"""
index.py — the ingestion pipeline and the in-memory catalogue.

INGEST( raw text )
   -> Canvas                 2-D character matrix, never flattened
   -> ParseResult            nodes / edges / annotations  (computer vision)
   -> SpatialChunk[]         multi-scale 2-D chunks  +  one chunk per node/edge
   -> FusedEmbedder          [semantic ‖ spatial] vectors
   -> VectorStore + BM25     dense & sparse indices, namespaced per document
   -> DiagramGraph           networkx symbolic layer

Every stage is timed and reported so the UI can show the real cost of indexing.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ascii_parser import ParseResult, parse_canvas
from .canvas import BBox, Canvas
from .embedder import FusedEmbedder, get_embedder
from .graph_store import DiagramGraph
from .spatial_chunker import (SpatialChunk, chunk_canvas, chunk_statistics,
                              class_histogram, position_phrase)
from .vector_store import BM25, VectorStore

try:
    from ..config import INDEX_DIR
except ImportError:                                   # pragma: no cover
    from config import INDEX_DIR


def _doc_id(name: str, text: str) -> str:
    h = hashlib.blake2s((name + "\x00" + text).encode(), digest_size=6).hexdigest()
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:28] or "doc"
    return f"{slug}-{h}"


@dataclass
class Document:
    id: str
    name: str
    raw: str
    kind: str                       # topology | floorplan | state | rack | custom
    canvas: Canvas
    parse: ParseResult
    graph: DiagramGraph
    chunks: List[SpatialChunk]
    stats: Dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    def meta(self) -> Dict:
        return {
            "id": self.id, "name": self.name, "kind": self.kind,
            "height": self.canvas.height, "width": self.canvas.width,
            "chunks": len(self.chunks),
            "nodes": self.graph.G.number_of_nodes(),
            "edges": self.graph.U.number_of_edges(),
            "stats": self.stats,
            "created": self.created,
        }


class IndexManager:
    """Owns every document, the shared vector store, and the lexical index."""

    def __init__(self) -> None:
        self.embedder: FusedEmbedder = get_embedder()
        self.store = VectorStore(self.embedder.dim)
        self.bm25 = BM25()
        self.docs: Dict[str, Document] = {}

    # -- ingest ------------------------------------------------------------
    def ingest(self, name: str, text: str, kind: str = "custom",
               progress: Optional[callable] = None) -> Document:
        t_all = time.perf_counter()
        emit = progress or (lambda *_a, **_k: None)

        t0 = time.perf_counter()
        cv = Canvas.from_text(text)
        t_canvas = time.perf_counter() - t0
        emit("canvas", {"height": cv.height, "width": cv.width,
                        "cells": cv.height * cv.width,
                        "ink": int(cv.occupancy.sum()), "ms": round(t_canvas * 1000, 2)})

        did = _doc_id(name, cv.to_text())

        t0 = time.perf_counter()
        pr = parse_canvas(cv)
        t_parse = time.perf_counter() - t0
        emit("parse", {**pr.diagnostics, "ms": round(t_parse * 1000, 2)})

        t0 = time.perf_counter()
        chunks = chunk_canvas(cv, did)
        chunks.extend(_structural_chunks(cv, did, pr))
        t_chunk = time.perf_counter() - t0
        emit("chunk", {**chunk_statistics(chunks), "ms": round(t_chunk * 1000, 2)})

        t0 = time.perf_counter()
        vecs = self.embedder.embed_chunks(cv, chunks)
        t_embed = time.perf_counter() - t0
        emit("embed", {"vectors": int(vecs.shape[0]), "dim": int(vecs.shape[1]),
                       "semantic_dim": self.embedder.sem_dim,
                       "spatial_dim": self.embedder.spa_dim,
                       "backend": self.embedder.backend_name,
                       "ms": round(t_embed * 1000, 2)})

        ns = f"diagram:{did}"
        payloads = [c.to_dict() for c in chunks]
        self.store.drop(ns)
        self.store.upsert(ns, [c.id for c in chunks], vecs, payloads)
        self.bm25.index(ns, [c.id for c in chunks],
                        [f"{c.readable} {c.describe()}" for c in chunks], payloads)

        t0 = time.perf_counter()
        graph = DiagramGraph.from_parse(did, pr)
        t_graph = time.perf_counter() - t0
        emit("graph", {**graph.summary(), "ms": round(t_graph * 1000, 2)})

        stats = {
            "timings_ms": {
                "canvas": round(t_canvas * 1000, 2),
                "parse": round(t_parse * 1000, 2),
                "chunk": round(t_chunk * 1000, 2),
                "embed": round(t_embed * 1000, 2),
                "graph": round(t_graph * 1000, 2),
                "total": round((time.perf_counter() - t_all) * 1000, 2),
            },
            "chunk_kinds": chunk_statistics(chunks),
            "parse": pr.diagnostics,
            "ink_ratio": round(float(cv.occupancy.mean()), 4),
        }

        doc = Document(id=did, name=name, raw=cv.to_text(), kind=kind, canvas=cv,
                       parse=pr, graph=graph, chunks=chunks, stats=stats)
        self.docs[did] = doc
        emit("done", doc.meta())
        return doc

    # -- access ------------------------------------------------------------
    def get(self, did: str) -> Optional[Document]:
        return self.docs.get(did)

    def list(self) -> List[Dict]:
        return [d.meta() for d in sorted(self.docs.values(), key=lambda x: x.created)]

    def chunk_by_id(self, did: str, cid: str) -> Optional[SpatialChunk]:
        doc = self.docs.get(did)
        if not doc:
            return None
        return next((c for c in doc.chunks if c.id == cid), None)

    def delete(self, did: str) -> bool:
        if did in self.docs:
            self.store.drop(f"diagram:{did}")
            self.docs.pop(did)
            return True
        return False


def _structural_chunks(cv: Canvas, did: str, pr: ParseResult) -> List[SpatialChunk]:
    """One chunk per detected node and edge, carrying its symbolic identity.

    These are the chunks that make hybrid retrieval actually hybrid: they are
    simultaneously addressable by vector similarity *and* by graph traversal,
    because their payload carries the node/edge id.
    """
    out: List[SpatialChunk] = []
    for n in pr.nodes:
        out.append(SpatialChunk(
            id=f"{did}:node:{n.id}", doc_id=did, kind="node", bbox=n.bbox,
            text=n.raw or cv.text_of(n.bbox, strip=False),
            readable=n.label, scale=0, density=cv.density(n.bbox),
            class_hist=class_histogram(cv, n.bbox),
            meta={"node_id": n.id, "label": n.label, "style": n.style,
                  "position_phrase": position_phrase(cv, n.bbox)},
        ))
    for e in pr.edges:
        if not e.bbox:
            continue
        src = next((n.label for n in pr.nodes if n.id == e.source), e.source)
        dst = next((n.label for n in pr.nodes if n.id == e.target), e.target)
        arrow = "->" if e.directed else "--"
        out.append(SpatialChunk(
            id=f"{did}:edge:{e.id}", doc_id=did, kind="edge", bbox=e.bbox,
            text=cv.text_of(e.bbox, strip=False),
            readable=f"{src} {arrow} {dst} {e.label}".strip(),
            scale=0, density=cv.density(e.bbox),
            class_hist=class_histogram(cv, e.bbox),
            meta={"edge_id": e.id, "source": e.source, "target": e.target,
                  "source_label": src, "target_label": dst,
                  "label": e.label, "directed": e.directed,
                  "position_phrase": position_phrase(cv, e.bbox)},
        ))
    for a in pr.annotations:
        out.append(SpatialChunk(
            id=f"{did}:note:{a.id}", doc_id=did, kind="region", bbox=a.bbox,
            text=a.text, readable=a.text, scale=0, density=cv.density(a.bbox),
            class_hist=class_histogram(cv, a.bbox),
            meta={"annotation": True,
                  "position_phrase": position_phrase(cv, a.bbox)},
        ))
    return out
