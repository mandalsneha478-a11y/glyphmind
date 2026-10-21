"""
retriever.py — the hybrid, intent-routed, spatially-aware retrieval engine.

Four retrievers run against every question, and their rankings are fused with
Reciprocal Rank Fusion (RRF), which is robust to the fact that a cosine score
and a BM25 score live on incomparable scales:

      RRF(d) = Σ_r  w_r / (K + rank_r(d))

  R1  DENSE      cosine over the fused [semantic ‖ spatial] vectors
  R2  LEXICAL    BM25 over chunk captions + readable text
  R3  SPATIAL    cosine over the spatial half only, against a *spatial prior*
                 synthesised from the question ("below the gateway", "top-right")
  R4  GRAPH      symbolic expansion: resolve entities -> k-hop neighbourhood ->
                 every chunk whose geometry overlaps that neighbourhood

R4 is the one that conventional RAG cannot have, because conventional RAG never
recovered a graph in the first place.

On top of fusion sits an INTENT ROUTER.  Structural questions (connectivity,
reachability, counting, centrality) are answered by *computing* over the graph,
not by generating.  The retrieved chunks then serve as citations for an answer
that was already proved correct.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .canvas import BBox, Canvas
from .embedder import l2, spatial_vector
from .graph_store import DiagramGraph
from .index import Document, IndexManager
from .spatial_chunker import SpatialChunk

try:
    from ..config import (FINAL_TOP_K, GRAPH_BOOST, GRAPH_HOPS, RRF_K,
                          TOP_K_DENSE, TOP_K_LEXICAL, TOP_K_SPATIAL)
except ImportError:                                   # pragma: no cover
    from config import (FINAL_TOP_K, GRAPH_BOOST, GRAPH_HOPS, RRF_K,
                        TOP_K_DENSE, TOP_K_LEXICAL, TOP_K_SPATIAL)


# ==========================================================================
#  Intent routing
# ==========================================================================
INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("path", re.compile(r"\b(path|route|reach|reachable|from\s+.+\s+to\s+|traverse|get from|flow from|hop)\b", re.I)),
    ("connectivity", re.compile(r"\b(connect|connected|connects|attach|link|linked|talk to|talks to|feeds|feed into|depend|neighbou?r|wired|downstream|upstream|adjacent to|transition|transitions|out of|leads? to|goes? to)\b", re.I)),
    ("direction", re.compile(r"\b(above|below|under|beneath|left of|right of|next to|beside|top[- ]?(left|right)|bottom[- ]?(left|right)|north|south|east|west|quadrant|corner|centre|center)\b", re.I)),
    ("topology", re.compile(r"\b(hub|bottleneck|single point|spof|critical|central|cycle|loop|isolated|orphan|root|entry point|sink|dead end|component|cluster)\b", re.I)),
    ("inventory", re.compile(r"\b(how many|count|list (all|every|the)|what (components|boxes|nodes|rooms|services|elements)|enumerate|inventory)\b", re.I)),
    ("containment", re.compile(r"\b(inside|within|contains|contained|enclosed|holds)\b", re.I)),
]

DIRECTION_WORDS = {
    "above": "above", "over": "above", "north": "above", "up from": "above",
    "below": "below", "under": "below", "beneath": "below", "south": "below",
    "left of": "left", "west": "left",
    "right of": "right", "east": "right",
}

QUADRANTS = ["top-left", "top left", "top-right", "top right",
             "bottom-left", "bottom left", "bottom-right", "bottom right",
             "top", "bottom", "left", "right", "centre", "center", "middle"]


def classify_intent(q: str) -> Tuple[str, List[str]]:
    hits = [name for name, pat in INTENT_PATTERNS if pat.search(q)]
    if not hits:
        return "semantic", []
    # Most specific wins. "what is inside the north wing" mentions a compass
    # word, but it is a containment question, so containment outranks direction.
    priority = ["path", "topology", "containment", "direction", "inventory",
                "connectivity"]
    for p in priority:
        if p in hits:
            return p, hits
    return hits[0], hits


# ==========================================================================
#  Evidence container
# ==========================================================================
@dataclass
class RetrievedChunk:
    chunk: SpatialChunk
    rrf: float
    dense: Optional[float] = None
    semantic: Optional[float] = None
    spatial: Optional[float] = None
    lexical: Optional[float] = None
    graph_hit: bool = False
    ranks: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d.update({
            "score": round(self.rrf, 5),
            "dense": None if self.dense is None else round(self.dense, 4),
            "semantic": None if self.semantic is None else round(self.semantic, 4),
            "spatial": None if self.spatial is None else round(self.spatial, 4),
            "lexical": None if self.lexical is None else round(self.lexical, 4),
            "graph_hit": self.graph_hit,
            "ranks": self.ranks,
        })
        # keep payloads light for the wire
        if len(d.get("text", "")) > 1600:
            d["text"] = d["text"][:1600] + "\n…"
        return d


@dataclass
class Evidence:
    question: str
    intent: str
    intent_hits: List[str]
    entities: List[Dict]
    facts: List[Dict]
    chunks: List[RetrievedChunk]
    graph_trace: Dict
    highlights: List[Dict]
    timings: Dict
    retriever_stats: Dict

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "intent": self.intent,
            "intent_hits": self.intent_hits,
            "entities": self.entities,
            "facts": self.facts,
            "chunks": [c.to_dict() for c in self.chunks],
            "graph_trace": self.graph_trace,
            "highlights": self.highlights,
            "timings": self.timings,
            "retriever_stats": self.retriever_stats,
        }


# ==========================================================================
#  Spatial prior synthesis
# ==========================================================================
def synth_spatial_prior(doc: Document, question: str,
                        anchors: Sequence[str]) -> Tuple[Optional[np.ndarray], Dict]:
    """Turn locative language into an actual region of the canvas.

    "what sits below the message broker"  ->  a BBox underneath the broker's box
    "what is in the top-right"            ->  the top-right quadrant BBox

    That BBox is then run through the same Φ descriptor used at index time, so
    the ANN search is literally searching *a place*, not a phrase.
    """
    cv = doc.canvas
    q = question.lower()
    info: Dict = {}

    # 1. relative to a resolved anchor node
    direction = None
    for word, canon in DIRECTION_WORDS.items():
        if word in q:
            direction = canon
            break
    if direction and anchors:
        b = doc.graph.bbox(anchors[0])
        if b:
            if direction == "below":
                box = BBox(b.r1, max(0, b.c0 - 6), min(cv.height, b.r1 + 14), min(cv.width, b.c1 + 6))
            elif direction == "above":
                box = BBox(max(0, b.r0 - 14), max(0, b.c0 - 6), b.r0, min(cv.width, b.c1 + 6))
            elif direction == "left":
                box = BBox(max(0, b.r0 - 3), max(0, b.c0 - 40), min(cv.height, b.r1 + 3), b.c0)
            else:
                box = BBox(max(0, b.r0 - 3), b.c1, min(cv.height, b.r1 + 3), min(cv.width, b.c1 + 40))
            info = {"mode": "relative", "direction": direction,
                    "anchor": doc.graph.label(anchors[0]), "box": box.to_dict()}
            return l2(spatial_vector(cv, box)), info

    # 2. absolute quadrant
    for quad in QUADRANTS:
        if quad in q:
            qn = quad.replace(" ", "-")
            r0 = 0 if "top" in qn else (cv.height // 2 if "bottom" in qn else cv.height // 4)
            r1 = cv.height // 2 if "top" in qn else (cv.height if "bottom" in qn else 3 * cv.height // 4)
            c0 = 0 if "left" in qn else (cv.width // 2 if "right" in qn else cv.width // 4)
            c1 = cv.width // 2 if "left" in qn else (cv.width if "right" in qn else 3 * cv.width // 4)
            box = BBox(r0, c0, max(r0 + 1, r1), max(c0 + 1, c1))
            info = {"mode": "absolute", "quadrant": qn, "box": box.to_dict()}
            return l2(spatial_vector(cv, box)), info

    # 3. anchor itself (no direction) — bias toward the anchor's own geometry
    if anchors:
        b = doc.graph.bbox(anchors[0])
        if b:
            info = {"mode": "anchor", "anchor": doc.graph.label(anchors[0]),
                    "box": b.to_dict()}
            return l2(spatial_vector(cv, b)), info
    return None, info


# ==========================================================================
#  Entity resolution
# ==========================================================================
STOP = set("""what which where who how many is are the a an of in on to from do does
and or for with that this it its can could should would there their them they
me my i you your please tell show find list give all any between via through
sits sit located location position diagram chart picture image""".split())


def resolve_entities(doc: Document, question: str) -> List[Dict]:
    g = doc.graph
    found: Dict[str, Dict] = {}

    # a. direct label containment (longest label first => greedy, accurate)
    labels = sorted(
        ((g.label(n) or "").strip(), n) for n in g.G.nodes if (g.label(n) or "").strip()
    )
    labels.sort(key=lambda x: -len(x[0]))
    ql = " " + re.sub(r"[^a-z0-9 ]+", " ", question.lower()) + " "
    for lab, nid in labels:
        if len(lab) < 2:
            continue
        pat = " " + re.sub(r"[^a-z0-9 ]+", " ", lab.lower()).strip() + " "
        if pat.strip() and pat in ql:
            found.setdefault(nid, {"id": nid, "label": lab, "confidence": 1.0,
                                   "how": "exact label match"})

    # b. fuzzy token overlap for anything still unmatched
    toks = [t for t in re.findall(r"[a-z0-9][a-z0-9\-_]+", question.lower())
            if t not in STOP and len(t) > 2]
    for t in toks:
        for nid, score in g.resolve(t, limit=3):
            if nid in found:
                continue
            if score >= 0.6:
                found[nid] = {"id": nid, "label": g.label(nid),
                              "confidence": round(score, 3),
                              "how": f"fuzzy match on '{t}'"}
    out = sorted(found.values(), key=lambda d: -d["confidence"])
    # order by appearance in the question so "from A to B" keeps A before B
    def pos(d):
        i = question.lower().find(d["label"].lower())
        return i if i >= 0 else 10_000
    out.sort(key=lambda d: (pos(d), -d["confidence"]))
    return out[:6]


# ==========================================================================
#  Fact derivation (grounding)
# ==========================================================================
def derive_facts(doc: Document, question: str, intent: str,
                 entities: List[Dict]) -> Tuple[List[Dict], Dict, List[Dict]]:
    """Compute, do not generate.  Returns (facts, graph_trace, highlights)."""
    g = doc.graph
    cv = doc.canvas
    facts: List[Dict] = []
    trace: Dict = {"intent": intent, "operations": []}
    highlights: List[Dict] = []
    ids = [e["id"] for e in entities]

    def hl(nid: str, role: str) -> None:
        b = g.bbox(nid)
        if b:
            highlights.append({"bbox": b.to_dict(), "label": g.label(nid), "role": role})

    if intent == "path" and len(ids) >= 2:
        a, b = ids[0], ids[1]
        trace["operations"].append(f"all_simple_paths({g.label(a)} → {g.label(b)})")
        paths = g.paths(a, b)
        hl(a, "source"); hl(b, "target")
        if paths:
            for p in paths[:4]:
                facts.append({"type": "path", "hops": len(p) - 1,
                              "nodes": [g.label(x) for x in p],
                              "node_ids": p})
                for mid in p[1:-1]:
                    hl(mid, "waypoint")
        else:
            facts.append({"type": "no_path", "from": g.label(a), "to": g.label(b)})
        sp = g.shortest(a, b)
        if sp:
            facts.append({"type": "shortest_path", "hops": len(sp) - 1,
                          "nodes": [g.label(x) for x in sp], "node_ids": sp})

    elif intent == "connectivity" and ids:
        for nid in ids[:2]:
            nb = g.neighbors(nid)
            trace["operations"].append(f"neighbors({g.label(nid)})")
            hl(nid, "focus")
            facts.append({"type": "connectivity", "node": g.label(nid),
                          "node_id": nid,
                          "outgoing": nb["outgoing"], "incoming": nb["incoming"],
                          "undirected": nb["undirected"],
                          "degree": len(nb["outgoing"]) + len(nb["incoming"]) + len(nb["undirected"])})
            for rec in nb["outgoing"] + nb["incoming"] + nb["undirected"]:
                hl(rec["node"], "neighbour")

    elif intent == "direction":
        direction = next((c for w, c in DIRECTION_WORDS.items() if w in question.lower()), None)
        if direction and ids:
            nid = ids[0]
            hl(nid, "anchor")
            trace["operations"].append(f"nodes_in_direction({g.label(nid)}, {direction})")
            res = g.nodes_in_direction(nid, direction)
            facts.append({"type": "direction", "anchor": g.label(nid),
                          "direction": direction,
                          "matches": [{"label": g.label(n), "distance": round(d, 2)}
                                      for n, d in res[:8]]})
            for n, _ in res[:8]:
                hl(n, "match")
        else:
            quad = next((q for q in QUADRANTS if q in question.lower()), None)
            if quad:
                trace["operations"].append(f"nodes_in_quadrant({quad})")
                res = g.nodes_in_quadrant(quad, cv.height, cv.width)
                facts.append({"type": "quadrant", "quadrant": quad,
                              "matches": [g.label(n) for n in res][:12]})
                for n in res[:12]:
                    hl(n, "match")

    elif intent == "topology":
        s = g.summary()
        trace["operations"].append("betweenness_centrality + degree + components + cycles")
        facts.append({"type": "topology", **s})
        cyc = g.cycles(4)
        if cyc:
            facts.append({"type": "cycles",
                          "cycles": [[g.label(x) for x in c] for c in cyc]})
        for h in s.get("hubs", [])[:4]:
            hl(h["id"], "hub")

    elif intent == "inventory":
        trace["operations"].append("graph.nodes + chunk census")
        nodes = [{"label": g.label(n), "style": g.G.nodes[n].get("style"),
                  "bbox": g.G.nodes[n].get("bbox")} for n in g.G.nodes]
        facts.append({"type": "inventory", "count": len(nodes),
                      "nodes": nodes[:60],
                      "edges": g.U.number_of_edges(),
                      "annotations": [a.text for a in doc.parse.annotations][:20]})
        for n in list(g.G.nodes)[:40]:
            hl(n, "item")

    elif intent == "containment" and ids:
        nid = ids[0]
        b = g.bbox(nid)
        hl(nid, "container")
        if b:
            inner: List[str] = []
            for other in g.G.nodes:
                if other == nid:
                    continue
                ob = g.bbox(other)
                if ob and ob.overlap_fraction(b) > 0.9:
                    inner.append(g.label(other)); hl(other, "contained")
            facts.append({"type": "containment", "container": g.label(nid),
                          "contains": inner,
                          "raw_text": cv.readable_text(b)})

    # Always attach a cheap global frame so even a pure-semantic answer is grounded.
    if not any(f["type"] == "topology" for f in facts):
        facts.append({"type": "frame", **g.summary(),
                      "canvas": {"height": cv.height, "width": cv.width}})
    for e in entities:
        hl(e["id"], "entity") if not any(h["label"] == e["label"] for h in highlights) else None
    trace["entities"] = entities
    return facts, trace, highlights


# ==========================================================================
#  The retriever
# ==========================================================================
class HybridRetriever:
    def __init__(self, manager: IndexManager) -> None:
        self.m = manager

    def retrieve(self, doc: Document, question: str, *,
                 top_k: int = FINAL_TOP_K,
                 progress: Optional[callable] = None) -> Evidence:
        emit = progress or (lambda *_a, **_k: None)
        timings: Dict[str, float] = {}
        ns = f"diagram:{doc.id}"
        emb = self.m.embedder

        t0 = time.perf_counter()
        intent, hits = classify_intent(question)
        entities = resolve_entities(doc, question)
        timings["route_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("route", {"intent": intent, "signals": hits,
                       "entities": entities, "ms": timings["route_ms"]})

        # ---- R3 prior ----------------------------------------------------
        t0 = time.perf_counter()
        prior, prior_info = synth_spatial_prior(doc, question, [e["id"] for e in entities])
        timings["prior_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("prior", {**prior_info, "ms": timings["prior_ms"],
                       "active": prior is not None})

        # ---- R1 dense ----------------------------------------------------
        t0 = time.perf_counter()
        qvec = emb.embed_query(question, prior)
        dense = self.m.store.search_split(ns, qvec, TOP_K_DENSE, emb.sem_dim)
        timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("dense", {"hits": len(dense), "ms": timings["dense_ms"],
                       "top": [{"id": d[0], "score": round(d[1], 4),
                                "semantic": round(d[2], 4), "spatial": round(d[3], 4),
                                "kind": d[4].get("kind")} for d in dense[:5]]})

        # ---- R2 lexical --------------------------------------------------
        t0 = time.perf_counter()
        lexical = self.m.bm25.search(ns, question, TOP_K_LEXICAL)
        timings["lexical_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("lexical", {"hits": len(lexical), "ms": timings["lexical_ms"],
                         "top": [{"id": x[0], "score": round(x[1], 3),
                                  "kind": x[2].get("kind")} for x in lexical[:5]]})

        # ---- R3 spatial-only --------------------------------------------
        t0 = time.perf_counter()
        spatial: List[Tuple[str, float, dict]] = []
        if prior is not None:
            pq = np.zeros(emb.dim, dtype=np.float32)
            pq[emb.sem_dim:] = prior
            spatial = [(i, s, p) for i, s, p in self.m.store.search(ns, pq, TOP_K_SPATIAL)]
        timings["spatial_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("spatial", {"hits": len(spatial), "ms": timings["spatial_ms"],
                         "top": [{"id": x[0], "score": round(x[1], 4),
                                  "kind": x[2].get("kind")} for x in spatial[:5]]})

        # ---- R4 graph expansion ------------------------------------------
        t0 = time.perf_counter()
        seeds = [e["id"] for e in entities]
        expanded = doc.graph.expand(seeds, GRAPH_HOPS) if seeds else set()
        graph_boxes = [doc.graph.bbox(n) for n in expanded]
        graph_boxes = [b for b in graph_boxes if b is not None]
        graph_hits: Set[str] = set()
        if graph_boxes:
            for ch in doc.chunks:
                for b in graph_boxes:
                    if ch.bbox.overlap_fraction(b) > 0.25 or b.overlap_fraction(ch.bbox) > 0.6:
                        graph_hits.add(ch.id)
                        break
        timings["graph_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("graph", {"seeds": [doc.graph.label(s) for s in seeds],
                       "expanded": len(expanded), "chunk_hits": len(graph_hits),
                       "hops": GRAPH_HOPS, "ms": timings["graph_ms"]})

        # ---- fusion ------------------------------------------------------
        t0 = time.perf_counter()
        by_id: Dict[str, RetrievedChunk] = {}
        chunk_map = {c.id: c for c in doc.chunks}

        def touch(cid: str) -> Optional[RetrievedChunk]:
            ch = chunk_map.get(cid)
            if ch is None:
                return None
            if cid not in by_id:
                by_id[cid] = RetrievedChunk(chunk=ch, rrf=0.0)
            return by_id[cid]

        weights = {"dense": 1.0, "lexical": 0.75, "spatial": 0.9, "graph": 1.0}
        if intent in ("direction", "containment"):
            weights["spatial"] = 1.35
            weights["lexical"] = 0.5
        if intent in ("path", "connectivity", "topology", "inventory"):
            weights["graph"] = 1.5

        for rank, (cid, total, sem, spa, _pl) in enumerate(dense):
            rc = touch(cid)
            if rc:
                rc.dense = total; rc.semantic = sem; rc.spatial = spa
                rc.ranks["dense"] = rank + 1
                rc.rrf += weights["dense"] / (RRF_K + rank + 1)
        for rank, (cid, score, _pl) in enumerate(lexical):
            rc = touch(cid)
            if rc:
                rc.lexical = score
                rc.ranks["lexical"] = rank + 1
                rc.rrf += weights["lexical"] / (RRF_K + rank + 1)
        for rank, (cid, score, _pl) in enumerate(spatial):
            rc = touch(cid)
            if rc:
                if rc.spatial is None:
                    rc.spatial = score
                rc.ranks["spatial"] = rank + 1
                rc.rrf += weights["spatial"] / (RRF_K + rank + 1)
        for cid in graph_hits:
            rc = touch(cid)
            if rc:
                rc.graph_hit = True
                rc.rrf += weights["graph"] * GRAPH_BOOST / RRF_K

        # kind priors: structural chunks are the most citable units
        kind_bonus = {"node": 1.25, "edge": 1.2, "region": 1.05,
                      "tile": 1.0, "row_band": 0.9, "col_band": 0.9, "document": 0.75}
        for rc in by_id.values():
            rc.rrf *= kind_bonus.get(rc.chunk.kind, 1.0)

        ranked = sorted(by_id.values(), key=lambda r: -r.rrf)

        # diversity: avoid returning eight near-identical overlapping tiles
        chosen: List[RetrievedChunk] = []
        for rc in ranked:
            if len(chosen) >= top_k:
                break
            if any(rc.chunk.bbox.iou(o.chunk.bbox) > 0.72 and rc.chunk.kind == o.chunk.kind
                   for o in chosen):
                continue
            chosen.append(rc)
        timings["fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("fuse", {"candidates": len(by_id), "selected": len(chosen),
                      "weights": weights, "ms": timings["fusion_ms"]})

        # ---- grounding ---------------------------------------------------
        t0 = time.perf_counter()
        facts, trace, highlights = derive_facts(doc, question, intent, entities)
        trace["expanded_nodes"] = [doc.graph.label(n) for n in list(expanded)[:30]]
        timings["ground_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        emit("ground", {"facts": len(facts), "operations": trace["operations"],
                        "ms": timings["ground_ms"]})

        timings["total_ms"] = round(sum(v for k, v in timings.items() if k.endswith("_ms")), 2)
        stats = {
            "dense": len(dense), "lexical": len(lexical),
            "spatial": len(spatial), "graph": len(graph_hits),
            "candidates": len(by_id), "returned": len(chosen),
            "weights": weights, "prior": prior_info,
            "encoder": emb.backend_name,
            "dims": {"semantic": emb.sem_dim, "spatial": emb.spa_dim, "fused": emb.dim},
        }
        return Evidence(question=question, intent=intent, intent_hits=hits,
                        entities=entities, facts=facts, chunks=chosen,
                        graph_trace=trace, highlights=highlights,
                        timings=timings, retriever_stats=stats)
