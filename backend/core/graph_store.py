"""
graph_store.py — the symbolic half of the hybrid.

Dense vectors are wonderful at "find me something that feels like this" and
hopeless at "is there a path from the ingress controller to the audit log that
does not traverse the public subnet".  That second question is a graph query,
and GLYPHMIND answers it by graph query — exactly, with a proof trace — rather
than by hoping a language model guesses right.

Every answer that this module produces is *derived*, not *generated*.  It is
the grounding layer: the LLM (when present) is only ever allowed to narrate
facts that came out of here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from .ascii_parser import Edge, Node, ParseResult
from .canvas import BBox


class DiagramGraph:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id
        self.G = nx.MultiDiGraph()
        self.U = nx.Graph()              # undirected view for reachability
        self.label_index: Dict[str, List[str]] = {}

    # -- build -------------------------------------------------------------
    @staticmethod
    def from_parse(doc_id: str, pr: ParseResult) -> "DiagramGraph":
        dg = DiagramGraph(doc_id)
        for n in pr.nodes:
            dg.G.add_node(n.id, label=n.label, bbox=n.bbox.to_dict(),
                          style=n.style, raw=n.raw, **{k: v for k, v in n.attrs.items()
                                                       if isinstance(v, (int, float, str, bool))})
            dg.U.add_node(n.id, label=n.label)
            key = n.label.strip().lower()
            dg.label_index.setdefault(key, []).append(n.id)
        for e in pr.edges:
            dg.G.add_edge(e.source, e.target, key=e.id, label=e.label,
                          directed=e.directed, bus=bool(e.attrs.get("bus")),
                          bidirectional=bool(e.attrs.get("bidirectional")),
                          length=int(e.attrs.get("length", 0)),
                          bbox=e.bbox.to_dict() if e.bbox else None)
            if not e.directed:
                dg.G.add_edge(e.target, e.source, key=e.id + "_r", label=e.label,
                              directed=False, bus=bool(e.attrs.get("bus")),
                              bidirectional=bool(e.attrs.get("bidirectional")),
                              length=int(e.attrs.get("length", 0)),
                              bbox=e.bbox.to_dict() if e.bbox else None)
            dg.U.add_edge(e.source, e.target, label=e.label)

        # --- co-reference of repeated bracket states -----------------------
        # A finite-state diagram draws `[ PENDING ]` twice when two transitions
        # both re-enter it.  Geometrically those are two rectangles; logically
        # they are one state.  We keep both instances (position still matters
        # for spatial questions) but stitch them together in the undirected
        # view so reachability and path queries behave correctly.
        by_label: Dict[str, List[str]] = {}
        for n in pr.nodes:
            if n.style == "bracket" and n.label.strip():
                by_label.setdefault(n.label.strip().lower(), []).append(n.id)
        for lab, group in by_label.items():
            if len(group) < 2:
                continue
            anchor = group[0]
            for other in group[1:]:
                dg.U.add_edge(anchor, other, label="", alias=True)
                dg.G.add_edge(anchor, other, key=f"alias_{other}", label="",
                              directed=False, alias=True, bus=False,
                              bidirectional=False, length=0, bbox=None)
                dg.G.add_edge(other, anchor, key=f"alias_{anchor}_{other}", label="",
                              directed=False, alias=True, bus=False,
                              bidirectional=False, length=0, bbox=None)
                dg.G.nodes[other]["alias_of"] = anchor
        return dg

    # -- lookup ------------------------------------------------------------
    def label(self, nid: str) -> str:
        return self.G.nodes.get(nid, {}).get("label", nid)

    def bbox(self, nid: str) -> Optional[BBox]:
        d = self.G.nodes.get(nid, {}).get("bbox")
        return BBox.from_dict(d) if d else None

    def resolve(self, text: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Fuzzy entity linking from free text to node ids."""
        t = (text or "").strip().lower()
        if not t:
            return []
        scored: List[Tuple[str, float]] = []
        toks = set(w for w in t.replace("-", " ").replace("_", " ").split() if len(w) > 1)
        for nid, data in self.G.nodes(data=True):
            lab = (data.get("label") or "").strip().lower()
            if not lab:
                continue
            if lab == t:
                scored.append((nid, 1.0)); continue
            ltoks = set(w for w in lab.replace("-", " ").replace("_", " ").split() if len(w) > 1)
            if not ltoks:
                continue
            if lab in t or t in lab:
                scored.append((nid, 0.86)); continue
            inter = toks & ltoks
            if inter:
                j = len(inter) / len(toks | ltoks)
                if j >= 0.2:
                    scored.append((nid, 0.5 + 0.4 * j))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # -- structural queries ------------------------------------------------
    def neighbors(self, nid: str) -> Dict[str, List[Dict]]:
        out = {"outgoing": [], "incoming": [], "undirected": []}
        for _, dst, data in self.G.out_edges(nid, data=True):
            if data.get("alias"):
                continue
            rec = {"node": dst, "label": self.label(dst), "edge_label": data.get("label", "")}
            (out["outgoing"] if data.get("directed") else out["undirected"]).append(rec)
        for src, _, data in self.G.in_edges(nid, data=True):
            if data.get("alias"):
                continue
            if data.get("directed"):
                out["incoming"].append({"node": src, "label": self.label(src),
                                        "edge_label": data.get("label", "")})
        # de-duplicate undirected mirrors
        seen = set(); uniq = []
        for r in out["undirected"]:
            if r["node"] in seen:
                continue
            seen.add(r["node"]); uniq.append(r)
        out["undirected"] = uniq
        return out

    def paths(self, a: str, b: str, cutoff: int = 7, limit: int = 6) -> List[List[str]]:
        try:
            gen = nx.all_simple_paths(self.G, a, b, cutoff=cutoff)
            found = []
            for p in gen:
                found.append(p)
                if len(found) >= limit:
                    break
            if found:
                return found
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        try:
            gen = nx.all_simple_paths(self.U, a, b, cutoff=cutoff)
            return [p for i, p in zip(range(limit), gen)]
        except Exception:
            return []

    def shortest(self, a: str, b: str) -> Optional[List[str]]:
        for g in (self.G, self.U):
            try:
                return nx.shortest_path(g, a, b)
            except Exception:
                continue
        return None

    def expand(self, seeds: Sequence[str], hops: int = 2) -> Set[str]:
        out: Set[str] = set()
        for s in seeds:
            if s not in self.U:
                continue
            out.add(s)
            frontier = {s}
            for _ in range(hops):
                nxt: Set[str] = set()
                for n in frontier:
                    nxt |= set(self.U.neighbors(n))
                nxt -= out
                out |= nxt
                frontier = nxt
        return out

    def centrality(self, top: int = 8) -> List[Tuple[str, str, float]]:
        if self.U.number_of_nodes() == 0:
            return []
        try:
            bc = nx.betweenness_centrality(self.U)
        except Exception:
            bc = {n: 0.0 for n in self.U.nodes}
        deg = dict(self.U.degree())
        maxd = max(deg.values()) if deg else 1
        score = {n: 0.6 * bc.get(n, 0) + 0.4 * (deg.get(n, 0) / max(1, maxd))
                 for n in self.U.nodes}
        ranked = sorted(score.items(), key=lambda x: -x[1])[:top]
        return [(n, self.label(n), float(s)) for n, s in ranked]

    def components(self) -> List[List[str]]:
        return [sorted(c) for c in nx.connected_components(self.U)]

    def cycles(self, limit: int = 8) -> List[List[str]]:
        try:
            return [c for i, c in zip(range(limit), nx.simple_cycles(self.G)) if len(c) > 1]
        except Exception:
            return []

    def isolated(self) -> List[str]:
        return [n for n in self.U.nodes if self.U.degree(n) == 0]

    def roots(self) -> List[str]:
        return [n for n in self.G.nodes
                if self.G.in_degree(n) == 0 and self.G.out_degree(n) > 0]

    def sinks(self) -> List[str]:
        return [n for n in self.G.nodes
                if self.G.out_degree(n) == 0 and self.G.in_degree(n) > 0]

    def summary(self) -> Dict:
        return {
            "nodes": self.G.number_of_nodes(),
            "edges": self.U.number_of_edges(),
            "components": len(self.components()),
            "isolated": len(self.isolated()),
            "roots": [self.label(n) for n in self.roots()][:10],
            "sinks": [self.label(n) for n in self.sinks()][:10],
            "hubs": [{"id": i, "label": l, "score": round(s, 4)}
                     for i, l, s in self.centrality(6)],
            "has_cycles": bool(self.cycles(1)),
        }

    # -- spatial relations (graph ⟂ geometry) ------------------------------
    def spatial_relation(self, a: str, b: str) -> str:
        ba, bb = self.bbox(a), self.bbox(b)
        if not ba or not bb:
            return "unknown"
        ar, ac = ba.centroid
        br, bc = bb.centroid
        dr, dc = br - ar, bc - ac
        parts = []
        if abs(dr) > max(1.0, ba.height * 0.5):
            parts.append("below" if dr > 0 else "above")
        if abs(dc) > max(1.0, ba.width * 0.5):
            parts.append("to the right of" if dc > 0 else "to the left of")
        if not parts:
            return "overlapping / adjacent to"
        return " and ".join(parts)

    def nodes_in_direction(self, nid: str, direction: str) -> List[Tuple[str, float]]:
        base = self.bbox(nid)
        if not base:
            return []
        br, bc = base.centroid
        out = []
        for other in self.G.nodes:
            if other == nid:
                continue
            ob = self.bbox(other)
            if not ob:
                continue
            orr, occ = ob.centroid
            dr, dc = orr - br, occ - bc
            ok = {
                "above": dr < -0.5 and abs(dc) <= max(base.width, ob.width) * 1.6,
                "below": dr > 0.5 and abs(dc) <= max(base.width, ob.width) * 1.6,
                "left": dc < -0.5 and abs(dr) <= max(base.height, ob.height) * 2.2,
                "right": dc > 0.5 and abs(dr) <= max(base.height, ob.height) * 2.2,
            }.get(direction, False)
            if ok:
                out.append((other, float((dr ** 2 + dc ** 2) ** 0.5)))
        out.sort(key=lambda x: x[1])
        return out

    def nodes_in_quadrant(self, quadrant: str, canvas_h: int, canvas_w: int
                          ) -> List[str]:
        q = quadrant.lower()
        out = []
        for nid in self.G.nodes:
            b = self.bbox(nid)
            if not b:
                continue
            r, c = b.centroid
            vr, vc = r / max(1, canvas_h), c / max(1, canvas_w)
            v = "top" if vr < 0.4 else ("bottom" if vr > 0.6 else "middle")
            h = "left" if vc < 0.4 else ("right" if vc > 0.6 else "centre")
            tag = f"{v}-{h}"
            if ("top" in q and v == "top") or ("bottom" in q and v == "bottom") \
               or ("left" in q and h == "left") or ("right" in q and h == "right") \
               or ("cent" in q and tag == "middle-centre"):
                # require both halves when both are mentioned
                mentions = sum(1 for w in ("top", "bottom", "left", "right") if w in q)
                if mentions >= 2:
                    ok = all((w not in q) or (w in tag) for w in ("top", "bottom", "left", "right"))
                    if ok:
                        out.append(nid)
                else:
                    out.append(nid)
        return out

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "nodes": [{"id": n, **{k: v for k, v in d.items()}}
                      for n, d in self.G.nodes(data=True)],
            "edges": [{"source": u, "target": v, **{k: w for k, w in d.items()}}
                      for u, v, d in self.G.edges(data=True)],
        }
