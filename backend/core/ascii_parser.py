"""
ascii_parser.py — structural vision for text.

This module performs what is, in effect, *computer vision on a character
matrix*.  It recovers the latent graph that a human sees instantly when they
look at a box-and-arrow diagram, and which no language model can recover from a
flattened token stream.

Pipeline
--------
  1. NODE DETECTION
       a. Rectangle scan.  Find top-left corner glyphs, walk the top edge to a
          top-right corner, drop both verticals, verify the bottom edge closes.
          Handles ASCII `+---+` and all Unicode box styles, single and double.
       b. Bracket scan.  `[ label ]`, `( label )`, `< label >`, `{ label }` on a
          single row — the idiom used in quick network sketches.
       c. Fallback.  If a canvas contains no rectangles at all (e.g. an indented
          tree or a floor-plan legend), isolated text clusters become nodes so
          the graph layer still has something to reason over.

  2. WIRE SEGMENTATION
       Mask every structural glyph that is NOT part of a node's border, then
       take 8-connected components.  Each component is one physical wire run.

  3. INCIDENCE + DIRECTION
       For each wire run, find every node whose border it touches.  Inspect the
       terminal glyphs at each contact for arrowheads to recover direction.
       Two contacts -> one edge.  N contacts -> a bus, expanded into pairwise
       edges flagged `bus=True`.

  4. LABELLING
       Text glyphs lying inside a node become its label.  Text glyphs hugging a
       wire run (within 1 cell, not inside any node) become the edge label.

  5. ANNOTATIONS
       Everything else — titles, legends, captions — is captured as free text
       with its own geometry so it remains retrievable.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .canvas import (ALL_BOX, ARROWHEADS, BBox, BOX_CORNERS, BOX_DIAG,
                     BOX_HORIZ, BOX_JUNCTION, BOX_VERT, SOFT_ARROWHEADS, Canvas)

# Corner glyph families -----------------------------------------------------
TL = set("+┌╔╭┏")
TR = set("+┐╗╮┓")
BL = set("+└╚╰┗")
BR = set("+┘╝╯┛")
HORIZ_EDGE = BOX_HORIZ | BOX_JUNCTION | TL | TR | BL | BR
VERT_EDGE = BOX_VERT | BOX_JUNCTION | TL | TR | BL | BR

WIRE_GLYPHS = BOX_HORIZ | BOX_VERT | BOX_JUNCTION | BOX_DIAG | ARROWHEADS | BOX_CORNERS

ARROW_INTO = {
    (0, 1): set(">►→»"),     # travelling right, so a right-pointing head means
    (0, -1): set("<◄←«"),    # the wire terminates *into* the thing on the right
    (1, 0): set("vV▼↓"),
    (-1, 0): set("^▲↑"),
}


def is_arrow_glyph(cv: Canvas, r: int, c: int) -> bool:
    """Context-sensitive arrowhead test.

    `v` is a letter in `Svc`, `Archive` and `VLAN`, and an arrowhead in

            |
            v
        +---+---+

    Disambiguation rule: a soft arrowhead counts as an arrow only when a wire
    glyph sits on the matching side and no letter touches it horizontally.
    """
    ch = cv.at(r, c)
    if ch in ARROWHEADS:
        return True
    if ch not in SOFT_ARROWHEADS:
        return False
    left, right = cv.at(r, c - 1), cv.at(r, c + 1)
    if left.isalnum() or right.isalnum():
        return False
    probe = cv.at(r - 1, c) if ch in "vV" else cv.at(r + 1, c)
    return probe in (BOX_VERT | BOX_JUNCTION | BOX_CORNERS | BOX_DIAG)


@dataclass
class Node:
    id: str
    label: str
    bbox: BBox
    style: str                       # "rect" | "bracket" | "text"
    raw: str = ""
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "bbox": self.bbox.to_dict(),
                "style": self.style, "attrs": self.attrs}


@dataclass
class Edge:
    id: str
    source: str
    target: str
    directed: bool
    label: str = ""
    bbox: Optional[BBox] = None
    cells: List[Tuple[int, int]] = field(default_factory=list)
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "target": self.target,
                "directed": self.directed, "label": self.label,
                "bbox": self.bbox.to_dict() if self.bbox else None,
                "attrs": self.attrs,
                "cells": self.cells[:400]}


@dataclass
class Annotation:
    id: str
    text: str
    bbox: BBox

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "bbox": self.bbox.to_dict()}


@dataclass
class ParseResult:
    nodes: List[Node]
    edges: List[Edge]
    annotations: List[Annotation]
    diagnostics: Dict

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "annotations": [a.to_dict() for a in self.annotations],
            "diagnostics": self.diagnostics,
        }


def _nid(prefix: str, *parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.blake2s(raw.encode(), digest_size=5).hexdigest()}"


# ==========================================================================
#  1. NODE DETECTION
# ==========================================================================
def _detect_rectangles(cv: Canvas) -> List[Node]:
    nodes: List[Node] = []
    taken: Set[Tuple[int, int, int, int]] = set()
    H, W = cv.height, cv.width

    for r in range(H):
        for c in range(W):
            ch = cv.at(r, c)
            if ch not in TL:
                continue
            # The top-left corner must actually open rightwards and downwards.
            if cv.at(r, c + 1) not in HORIZ_EDGE or cv.at(r + 1, c) not in VERT_EDGE:
                continue
            # Walk the top edge.
            for c2 in range(c + 2, W):
                top = cv.at(r, c2)
                if top in TR:
                    # Candidate right edge: must open downwards.
                    if cv.at(r + 1, c2) not in VERT_EDGE:
                        continue
                    # Walk both verticals down until a bottom corner lines up.
                    for r2 in range(r + 2, H):
                        bl, br = cv.at(r2, c), cv.at(r2, c2)
                        if bl in BL and br in BR:
                            # Verify the bottom edge is (mostly) solid.
                            span = [cv.at(r2, x) for x in range(c + 1, c2)]
                            solid = sum(1 for x in span if x in HORIZ_EDGE)
                            if not span or solid / len(span) < 0.6:
                                break
                            # Verify the verticals are (mostly) solid.
                            lv = [cv.at(x, c) for x in range(r + 1, r2)]
                            rv = [cv.at(x, c2) for x in range(r + 1, r2)]
                            lok = sum(1 for x in lv if x in VERT_EDGE) / max(1, len(lv))
                            rok = sum(1 for x in rv if x in VERT_EDGE) / max(1, len(rv))
                            if lok < 0.6 or rok < 0.6:
                                break
                            key = (r, c, r2 + 1, c2 + 1)
                            if key in taken:
                                break
                            taken.add(key)
                            box = BBox(r, c, r2 + 1, c2 + 1)
                            inner = BBox(r + 1, c + 1, r2, c2)
                            label = cv.readable_text(inner)
                            nodes.append(Node(
                                id=_nid("n", r, c, r2, c2),
                                label=label or f"box@{r},{c}",
                                bbox=box, style="rect",
                                raw=cv.text_of(box, strip=False),
                                attrs={"inner": inner.to_dict(),
                                       "double": cv.at(r, c) in "╔",
                                       "rounded": cv.at(r, c) in "╭"},
                            ))
                            break
                        if bl not in VERT_EDGE or br not in VERT_EDGE:
                            break
                    break
                if top not in HORIZ_EDGE:
                    break
    # --- nested-rectangle label correction --------------------------------
    # A container rectangle naively inherits the text of everything it encloses
    # (a rack slot would be labelled with the name of the server inside it).
    # Recompute each label from the glyphs that belong to NO child rectangle.
    for n in nodes:
        children = [m for m in nodes
                    if m is not n and m.bbox.area < n.bbox.area
                    and m.bbox.overlap_fraction(n.bbox) >= 0.95]
        if not children:
            continue
        inner = BBox(n.bbox.r0 + 1, n.bbox.c0 + 1, n.bbox.r1 - 1, n.bbox.c1 - 1)
        words: List[str] = []
        for rr in range(max(0, inner.r0), min(cv.height, inner.r1)):
            buf = []
            for cc in range(max(0, inner.c0), min(cv.width, inner.c1)):
                if any(m.bbox.contains_point(rr, cc) for m in children):
                    buf.append(" ")
                elif cv.cls_at(rr, cc) in (8, 9):
                    buf.append(cv.at(rr, cc))
                elif cv.at(rr, cc) in "._-/()&,'":
                    buf.append(cv.at(rr, cc))
                else:
                    buf.append(" ")
            words.extend("".join(buf).split())
        own = " ".join(words).strip(" ._-")
        n.label = own or f"container ({len(children)} items)"
        n.attrs["container"] = True
        n.attrs["child_count"] = len(children)

    # Drop rectangles fully contained in a larger one *and* label-free.
    filtered: List[Node] = []
    for n in nodes:
        parent = next((m for m in nodes
                       if m is not n and m.bbox.overlap_fraction(n.bbox) < 1.0
                       and n.bbox.overlap_fraction(m.bbox) >= 0.999
                       and m.bbox.area > n.bbox.area), None)
        if parent is not None and not n.label.strip():
            continue
        # A wrapper that holds exactly one child and carries no text of its own
        # (a rack slot, a group frame) adds nothing a human would name.
        if n.attrs.get("container") and n.attrs.get("child_count") == 1 \
                and n.label.startswith("container"):
            continue
        filtered.append(n)
    return filtered


BRACKET_RE = re.compile(r"(\[[^\[\]]{1,48}\]|\([^()]{1,48}\)|\{[^{}]{1,48}\}|<[^<>]{1,48}>)")


def _detect_brackets(cv: Canvas, existing: Sequence[Node]) -> List[Node]:
    nodes: List[Node] = []
    for r in range(cv.height):
        line = "".join(cv.grid[r])
        for m in BRACKET_RE.finditer(line):
            c0, c1 = m.start(), m.end()
            inner = m.group(0)[1:-1].strip()
            if not inner or not any(ch.isalnum() for ch in inner):
                continue
            box = BBox(r, c0, r + 1, c1)
            if any(n.bbox.overlap_fraction(box) > 0.5 or box.overlap_fraction(n.bbox) > 0.5
                   for n in existing):
                continue
            nodes.append(Node(
                id=_nid("b", r, c0, c1), label=inner, bbox=box,
                style="bracket", raw=m.group(0),
                attrs={"inner": BBox(r, c0 + 1, r + 1, c1 - 1).to_dict()},
            ))
    return nodes


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ._/&'-]{0,46}[A-Za-z0-9)]|[A-Za-z0-9]{2,}")


def _detect_text_clusters(cv: Canvas, existing: Sequence[Node]) -> List[Node]:
    """Fallback node source for diagrams that use no boxes at all."""
    nodes: List[Node] = []
    for r in range(cv.height):
        line = "".join(cv.grid[r])
        for m in re.finditer(r"\S(?:[^\n]{0,60}?\S)?(?=\s{2,}|$)", line):
            frag = m.group(0).strip()
            if len(frag) < 2 or not any(ch.isalnum() for ch in frag):
                continue
            box = BBox(r, m.start(), r + 1, m.start() + len(m.group(0)))
            if any(box.overlap_fraction(n.bbox) > 0.3 for n in existing):
                continue
            clean = re.sub(r"[^\w .&/'()-]+", " ", frag).strip()
            if not clean:
                continue
            nodes.append(Node(id=_nid("t", r, m.start()), label=clean,
                              bbox=box, style="text", raw=frag))
    return nodes


# ==========================================================================
#  2. WIRE SEGMENTATION
# ==========================================================================
def _wire_mask(cv: Canvas, nodes: Sequence[Node]) -> np.ndarray:
    mask = np.zeros((cv.height, cv.width), dtype=bool)
    for r in range(cv.height):
        for c in range(cv.width):
            if cv.at(r, c) in WIRE_GLYPHS or is_arrow_glyph(cv, r, c):
                mask[r, c] = True
    for n in nodes:
        b = n.bbox
        mask[max(0, b.r0):b.r1, max(0, b.c0):b.c1] = False
    return mask


H_BRIDGE = 14      # max columns an inline edge label may occupy
V_BRIDGE = 4       # max rows an inline edge label may occupy


def _bridge_targets(mask: np.ndarray, cv: Canvas, nodes: Sequence[Node],
                    r: int, c: int) -> List[Tuple[int, int]]:
    """Cells reachable by *jumping over an inline edge label*.

    `[ DRAFT ] ---- submit ----> [ PENDING ]` draws one logical connector, but
    the word `submit` physically severs the glyph run in two.  A naive
    connected-component pass therefore sees two half-wires that each touch only
    one node, and silently drops the edge — which is precisely the class of
    failure that makes ordinary pipelines useless on diagrams.

    So: from a wire cell, probe outward along each cardinal axis.  If we cross
    only non-node, non-wire cells, at least one of which is alphanumeric (i.e.
    it really is a label, not empty space), and we land on another wire cell,
    the two runs are one connector.
    """
    h, w = mask.shape
    out: List[Tuple[int, int]] = []
    for dr, dc, limit in ((0, 1, H_BRIDGE), (0, -1, H_BRIDGE),
                          (1, 0, V_BRIDGE), (-1, 0, V_BRIDGE)):
        saw_text = False
        for step in range(1, limit + 2):
            nr, nc = r + dr * step, c + dc * step
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if any(n.bbox.contains_point(nr, nc) for n in nodes):
                break
            if mask[nr, nc]:
                if step > 1 and (saw_text or step == 2):
                    out.append((nr, nc))
                break
            if cv.cls_at(nr, nc) in (8, 9):
                saw_text = True
            elif cv.at(nr, nc) != " ":
                break            # some other glyph: not a label gap
            if step > limit:
                break
    return out


def _components(mask: np.ndarray, cv: Optional[Canvas] = None,
                nodes: Sequence[Node] = ()) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    comps: List[List[Tuple[int, int]]] = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
                if cv is not None:
                    for (br, bc) in _bridge_targets(mask, cv, nodes, rr, cc):
                        if not seen[br, bc]:
                            seen[br, bc] = True
                            stack.append((br, bc))
            comps.append(cells)
    return comps


# ==========================================================================
#  3. INCIDENCE + DIRECTION
# ==========================================================================
def _node_at(nodes: Sequence[Node], r: int, c: int) -> Optional[Node]:
    for n in nodes:
        if n.bbox.contains_point(r, c):
            return n
    return None


def _contacts(cells: Sequence[Tuple[int, int]], nodes: Sequence[Node], cv: Canvas
              ) -> Dict[str, Dict]:
    """Map node-id -> contact information for one wire component."""
    out: Dict[str, Dict] = {}
    # Distance-1 in all 8 directions, plus distance-2 on the cardinals so that
    # `[ NODE ] --->` (one space of padding) still registers as a contact.
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1),
            (0, 2), (0, -2), (2, 0), (-2, 0)]
    for (r, c) in cells:
        for dr, dc in dirs:
            if abs(dr) == 2 or abs(dc) == 2:
                mid = (r + dr // 2, c + dc // 2)
                if cv.at(*mid) != " ":
                    continue              # only bridge a single blank cell
            n = _node_at(nodes, r + dr, c + dc)
            if n is None:
                continue
            dr, dc = (dr // 2, dc // 2) if (abs(dr) == 2 or abs(dc) == 2) else (dr, dc)
            info = out.setdefault(n.id, {"points": [], "arrow_in": False,
                                         "arrow_out": False})
            info["points"].append((r, c))
            # Arrowhead sitting on the contact cell, pointing toward the node?
            ch = cv.at(r, c)
            heads = ARROW_INTO.get((dr, dc), set())
            if ch in heads and is_arrow_glyph(cv, r, c):
                info["arrow_in"] = True
            # Arrowhead one cell back pointing away from the node?
            back = cv.at(r - dr, c - dc)
            if back in ARROW_INTO.get((-dr, -dc), set()) \
                    and is_arrow_glyph(cv, r - dr, c - dc):
                info["arrow_out"] = True
    return out


def _edge_label(cells: Sequence[Tuple[int, int]], cv: Canvas,
                nodes: Sequence[Node]) -> str:
    """Text glyphs hugging the wire, excluding anything inside a node."""
    found: Dict[int, Set[int]] = {}
    for (r, c) in cells:
        for dr in (-1, 0, 1):
            for dc in (-2, -1, 0, 1, 2):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < cv.height and 0 <= cc < cv.width):
                    continue
                if cv.cls_at(rr, cc) not in (8, 9):
                    continue
                if _node_at(nodes, rr, cc):
                    continue
                if is_arrow_glyph(cv, rr, cc):
                    continue            # a bare 'v' terminating the connector
                found.setdefault(rr, set()).add(cc)
    words: List[str] = []
    for rr, cols in sorted(found.items()):
        # Grow each seed column outward across the contiguous word.
        taken: Set[int] = set()
        for cc in sorted(cols):
            if cc in taken:
                continue
            a = cc
            while a - 1 >= 0 and cv.cls_at(rr, a - 1) in (8, 9, 10) \
                    and cv.at(rr, a - 1) not in " " and not _node_at(nodes, rr, a - 1) \
                    and not is_arrow_glyph(cv, rr, a - 1):
                a -= 1
            b = cc
            while b + 1 < cv.width and cv.cls_at(rr, b + 1) in (8, 9, 10) \
                    and cv.at(rr, b + 1) not in " " and not _node_at(nodes, rr, b + 1) \
                    and not is_arrow_glyph(cv, rr, b + 1):
                b += 1
            taken.update(range(a, b + 1))
            frag = "".join(cv.at(rr, x) for x in range(a, b + 1)).strip()
            frag = re.sub(r"^[^\w]+|[^\w)]+$", "", frag)
            if frag and any(ch.isalnum() for ch in frag):
                words.append(frag)
    seen: Set[str] = set()
    uniq = [w for w in words if not (w.lower() in seen or seen.add(w.lower()))]
    return " ".join(uniq)[:60]


# ==========================================================================
#  Orchestration
# ==========================================================================
def parse_canvas(cv: Canvas) -> ParseResult:
    diagnostics: Dict = {}

    rects = _detect_rectangles(cv)
    brackets = _detect_brackets(cv, rects)
    nodes: List[Node] = rects + brackets
    diagnostics["rectangles"] = len(rects)
    diagnostics["brackets"] = len(brackets)

    if not nodes:
        nodes = _detect_text_clusters(cv, [])
        diagnostics["fallback_text_nodes"] = len(nodes)

    # De-duplicate labels so the graph reads naturally.
    label_counts: Dict[str, int] = {}
    for n in nodes:
        base = (n.label or "unnamed").strip() or "unnamed"
        label_counts[base] = label_counts.get(base, 0) + 1
        if label_counts[base] > 1:
            n.attrs["label_occurrence"] = label_counts[base]

    mask = _wire_mask(cv, nodes)
    comps = _components(mask, cv, nodes)
    diagnostics["wire_components"] = len(comps)

    edges: List[Edge] = []
    used_cells: Set[Tuple[int, int]] = set()
    for cells in comps:
        if len(cells) < 1:
            continue
        contacts = _contacts(cells, nodes, cv)
        if len(contacts) < 2:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        box = BBox(min(rs), min(cs), max(rs) + 1, max(cs) + 1)
        label = _edge_label(cells, cv, nodes)
        ids = list(contacts.keys())
        bus = len(ids) > 2
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ia, ib = contacts[a], contacts[b]
                # Direction: an arrowhead pointing INTO b means a -> b.
                if ib["arrow_in"] and not ia["arrow_in"]:
                    src, dst, directed = a, b, True
                elif ia["arrow_in"] and not ib["arrow_in"]:
                    src, dst, directed = b, a, True
                elif ia["arrow_in"] and ib["arrow_in"]:
                    src, dst, directed = a, b, False      # bidirectional
                else:
                    src, dst, directed = a, b, False
                edges.append(Edge(
                    id=_nid("e", src, dst, box.r0, box.c0),
                    source=src, target=dst, directed=directed,
                    label=label, bbox=box, cells=cells,
                    attrs={"bus": bus,
                           "bidirectional": ia["arrow_in"] and ib["arrow_in"],
                           "length": len(cells)},
                ))
        used_cells.update(cells)

    # Annotations: alphanumeric runs that belong to neither node nor edge.
    annotations: List[Annotation] = []
    for r in range(cv.height):
        c = 0
        while c < cv.width:
            if cv.cls_at(r, c) in (8, 9) and not _node_at(nodes, r, c):
                a = c
                while c < cv.width and cv.at(r, c) != " " and not _node_at(nodes, r, c):
                    c += 1
                frag = "".join(cv.at(r, x) for x in range(a, c)).strip()
                near_wire = any((r + dr, x) in used_cells
                                for x in range(a, c) for dr in (-1, 0, 1))
                if len(frag) >= 3 and any(ch.isalnum() for ch in frag) and not near_wire:
                    annotations.append(Annotation(
                        id=_nid("a", r, a), text=frag, bbox=BBox(r, a, r + 1, c)))
            else:
                c += 1
    diagnostics["annotations"] = len(annotations)
    diagnostics["nodes"] = len(nodes)
    diagnostics["edges"] = len(edges)
    return ParseResult(nodes=nodes, edges=edges, annotations=annotations,
                       diagnostics=diagnostics)
