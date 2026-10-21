"""
spatial_chunker.py — 2-D grid-matrix chunking.

THE CORE RESEARCH CONTRIBUTION OF THIS PROJECT.

A conventional RAG chunker does: `text.split("\\n\\n")` or a sliding window over
tokens.  For a network topology drawn in `+--|` characters this is catastrophic:
a box whose label sits on row 7 and whose outbound arrow sits on row 9 gets
severed into two unrelated chunks, and the fact that they were vertically
adjacent — the *entire* meaning of the diagram — is annihilated.

GLYPHMIND chunks in two dimensions at three simultaneous scales:

  1. TILE CHUNKS      overlapping (rows x cols) windows across the matrix.
                      Multi-scale: fine tiles capture individual components,
                      coarse tiles capture whole sub-systems.  50 % overlap
                      guarantees no feature is ever cut exactly in half at
                      every scale at once.

  2. REGION CHUNKS    connected components of non-whitespace glyphs, grown with
                      a gap tolerance so that `[ web ]--->[ api ]` is captured
                      both as two regions and (at tolerance 2) as one relation.

  3. BAND CHUNKS      full-width row bands and full-height column bands.  These
                      are what let the system answer "what is in the top third"
                      or "which column does the trunk line run down".

  4. STRUCTURAL CHUNKS  emitted by the parser: one chunk per detected node and
                      one per detected edge, with their exact geometry.

Every chunk carries its BBox.  Nothing is ever flattened.  Retrieval can
therefore reason about adjacency, containment, alignment and direction — all of
which are invisible to a 1-D chunker.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .canvas import BBox, Canvas, GLYPH_CLASSES, N_GLYPH_CLASSES

try:                                   # config import works both as pkg & script
    from ..config import (MIN_TILE_DENSITY, REGION_GAP_TOLERANCE, TILE_OVERLAP,
                          TILE_SIZES)
except ImportError:                    # pragma: no cover
    from config import (MIN_TILE_DENSITY, REGION_GAP_TOLERANCE, TILE_OVERLAP,
                        TILE_SIZES)


CHUNK_KINDS = ("tile", "region", "row_band", "col_band", "node", "edge", "document")


@dataclass
class SpatialChunk:
    """A 2-D chunk: text + the geometry that text occupied."""

    id: str
    doc_id: str
    kind: str
    bbox: BBox
    text: str                       # raw glyph content, newlines preserved
    readable: str                   # alphanumeric content, collapsed
    scale: int = 0                  # 0 = finest
    density: float = 0.0
    class_hist: List[float] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    # ---------------------------------------------------------------- utils
    @property
    def centroid(self) -> Tuple[float, float]:
        return self.bbox.centroid

    def describe(self) -> str:
        """A natural-language caption used as the semantic-encoder input.

        This is the bridge between the geometric world and the semantic world:
        we *verbalise* the geometry so that a pre-trained sentence encoder —
        which has never seen a box-drawing character in its life — can still
        place the chunk meaningfully in semantic space.
        """
        h, w = self.bbox.height, self.bbox.width
        parts = [f"{self.kind} region {h} rows by {w} columns"]
        if self.meta.get("label"):
            parts.append(f"labelled '{self.meta['label']}'")
        if self.readable:
            parts.append(f"containing text: {self.readable[:220]}")
        hints = []
        ch = self.class_hist or [0.0] * N_GLYPH_CLASSES
        names = {1: "corners", 2: "horizontal lines", 3: "vertical lines",
                 4: "junctions", 5: "diagonals", 6: "arrows", 7: "shading"}
        for idx, nm in names.items():
            if idx < len(ch) and ch[idx] > 0.04:
                hints.append(nm)
        if hints:
            parts.append("drawn with " + ", ".join(hints))
        if self.meta.get("position_phrase"):
            parts.append(f"located in the {self.meta['position_phrase']}")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "kind": self.kind,
            "bbox": self.bbox.to_dict(),
            "text": self.text,
            "readable": self.readable,
            "scale": self.scale,
            "density": round(self.density, 4),
            "meta": self.meta,
        }


# --------------------------------------------------------------------------
#  Helpers
# --------------------------------------------------------------------------
def _cid(doc_id: str, kind: str, box: BBox, salt: str = "") -> str:
    raw = f"{doc_id}|{kind}|{box.r0},{box.c0},{box.r1},{box.c1}|{salt}"
    return hashlib.blake2s(raw.encode(), digest_size=8).hexdigest()


def class_histogram(canvas: Canvas, box: BBox) -> List[float]:
    r0 = max(0, box.r0); c0 = max(0, box.c0)
    r1 = min(canvas.height, box.r1); c1 = min(canvas.width, box.c1)
    if r1 <= r0 or c1 <= c0:
        return [0.0] * N_GLYPH_CLASSES
    sub = canvas.classes[r0:r1, c0:c1]
    counts = np.bincount(sub.ravel(), minlength=N_GLYPH_CLASSES).astype(np.float64)
    total = counts.sum() or 1.0
    return (counts / total).tolist()


def position_phrase(canvas: Canvas, box: BBox) -> str:
    """Human-readable coarse position — 'upper-left quadrant', 'centre', ..."""
    cr, cc = box.centroid
    vr = cr / max(1, canvas.height)
    vc = cc / max(1, canvas.width)
    v = "top" if vr < 0.33 else ("bottom" if vr > 0.66 else "middle")
    h = "left" if vc < 0.33 else ("right" if vc > 0.66 else "centre")
    if v == "middle" and h == "centre":
        return "centre of the canvas"
    return f"{v}-{h} area"


# --------------------------------------------------------------------------
#  Tile chunking (multi-scale, overlapping)
# --------------------------------------------------------------------------
def tile_chunks(canvas: Canvas, doc_id: str) -> List[SpatialChunk]:
    out: List[SpatialChunk] = []
    for scale, (th, tw) in enumerate(TILE_SIZES):
        if th > canvas.height * 2 and tw > canvas.width * 2 and scale > 0:
            continue
        sr = max(1, int(th * (1.0 - TILE_OVERLAP)))
        sc = max(1, int(tw * (1.0 - TILE_OVERLAP)))
        for r in range(0, max(1, canvas.height - 1), sr):
            for c in range(0, max(1, canvas.width - 1), sc):
                box = BBox(r, c, min(r + th, canvas.height), min(c + tw, canvas.width))
                if box.area == 0:
                    continue
                d = canvas.density(box)
                if d < MIN_TILE_DENSITY:
                    continue
                text = canvas.text_of(box, strip=False)
                if not text.strip():
                    continue
                out.append(SpatialChunk(
                    id=_cid(doc_id, "tile", box, str(scale)),
                    doc_id=doc_id, kind="tile", bbox=box,
                    text=text, readable=canvas.readable_text(box),
                    scale=scale, density=d,
                    class_hist=class_histogram(canvas, box),
                    meta={"position_phrase": position_phrase(canvas, box)},
                ))
    return out


# --------------------------------------------------------------------------
#  Region chunking (gap-tolerant connected components)
# --------------------------------------------------------------------------
def _dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    """Binary dilation with a (2*tol+1) square kernel, pure numpy."""
    if tol <= 0:
        return mask.copy()
    out = mask.copy()
    for dr in range(-tol, tol + 1):
        for dc in range(-tol, tol + 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(mask)
            r_src = slice(max(0, -dr), mask.shape[0] - max(0, dr))
            r_dst = slice(max(0, dr), mask.shape[0] - max(0, -dr))
            c_src = slice(max(0, -dc), mask.shape[1] - max(0, dc))
            c_dst = slice(max(0, dc), mask.shape[1] - max(0, -dc))
            shifted[r_dst, c_dst] = mask[r_src, c_src]
            out |= shifted
    return out


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected labelling via iterative flood fill (no scipy dependency)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c]:
                continue
            cur += 1
            stack = [(r, c)]
            labels[r, c] = cur
            while stack:
                rr, cc = stack.pop()
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = cur
                        stack.append((nr, nc))
    return labels, cur


def region_chunks(canvas: Canvas, doc_id: str,
                  tolerances: Tuple[int, ...] = (0, REGION_GAP_TOLERANCE, 3)
                  ) -> List[SpatialChunk]:
    """Connected glyph regions at several gap tolerances.

    tolerance 0  -> individual words / glyph clusters
    tolerance 1  -> whole boxes including their borders
    tolerance 3  -> relations: a box, its connector, and its target fuse into
                    one region, which is exactly the unit you want to retrieve
                    when the question is "what talks to the database?"
    """
    out: List[SpatialChunk] = []
    base = canvas.occupancy
    seen: set = set()
    for tol in tolerances:
        mask = _dilate(base, tol)
        labels, n = _label_components(mask)
        for i in range(1, n + 1):
            idx = np.argwhere(labels == i)
            if idx.size == 0:
                continue
            r0, c0 = idx.min(axis=0)
            r1, c1 = idx.max(axis=0) + 1
            box = BBox(int(r0), int(c0), int(r1), int(c1))
            if box.area < 2:
                continue
            # Skip a region that is the whole canvas unless it is the only one
            if box.area >= canvas.height * canvas.width and n > 1:
                continue
            key = (box.r0, box.c0, box.r1, box.c1)
            if key in seen:
                continue
            seen.add(key)
            text = canvas.text_of(box, strip=False)
            if not text.strip():
                continue
            out.append(SpatialChunk(
                id=_cid(doc_id, "region", box, f"t{tol}"),
                doc_id=doc_id, kind="region", bbox=box,
                text=text, readable=canvas.readable_text(box),
                scale=tol, density=canvas.density(box),
                class_hist=class_histogram(canvas, box),
                meta={"gap_tolerance": tol,
                      "position_phrase": position_phrase(canvas, box)},
            ))
    return out


# --------------------------------------------------------------------------
#  Band chunking
# --------------------------------------------------------------------------
def band_chunks(canvas: Canvas, doc_id: str, n_bands: int = 6) -> List[SpatialChunk]:
    out: List[SpatialChunk] = []
    bh = max(1, canvas.height // n_bands)
    for i in range(0, canvas.height, bh):
        box = BBox(i, 0, min(i + bh, canvas.height), canvas.width)
        if canvas.density(box) < 0.005:
            continue
        out.append(SpatialChunk(
            id=_cid(doc_id, "row_band", box),
            doc_id=doc_id, kind="row_band", bbox=box,
            text=canvas.text_of(box, strip=False),
            readable=canvas.readable_text(box),
            scale=0, density=canvas.density(box),
            class_hist=class_histogram(canvas, box),
            meta={"band_index": i // bh,
                  "position_phrase": position_phrase(canvas, box)},
        ))
    bw = max(1, canvas.width // n_bands)
    for j in range(0, canvas.width, bw):
        box = BBox(0, j, canvas.height, min(j + bw, canvas.width))
        if canvas.density(box) < 0.005:
            continue
        out.append(SpatialChunk(
            id=_cid(doc_id, "col_band", box),
            doc_id=doc_id, kind="col_band", bbox=box,
            text=canvas.text_of(box, strip=False),
            readable=canvas.readable_text(box),
            scale=0, density=canvas.density(box),
            class_hist=class_histogram(canvas, box),
            meta={"band_index": j // bw,
                  "position_phrase": position_phrase(canvas, box)},
        ))
    return out


# --------------------------------------------------------------------------
#  Orchestration
# --------------------------------------------------------------------------
def chunk_canvas(canvas: Canvas, doc_id: str, *,
                 include_bands: bool = True) -> List[SpatialChunk]:
    chunks: List[SpatialChunk] = []
    full = canvas.bbox
    chunks.append(SpatialChunk(
        id=_cid(doc_id, "document", full),
        doc_id=doc_id, kind="document", bbox=full,
        text=canvas.to_text(), readable=canvas.readable_text(full),
        scale=9, density=canvas.density(full),
        class_hist=class_histogram(canvas, full),
        meta={"position_phrase": "entire canvas"},
    ))
    chunks.extend(tile_chunks(canvas, doc_id))
    chunks.extend(region_chunks(canvas, doc_id))
    if include_bands:
        chunks.extend(band_chunks(canvas, doc_id))
    return chunks


def chunk_statistics(chunks: List[SpatialChunk]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for ch in chunks:
        stats[ch.kind] = stats.get(ch.kind, 0) + 1
    stats["total"] = len(chunks)
    return stats
