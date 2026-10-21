"""
canvas.py — the 2-D substrate.

Standard RAG flattens a document into a 1-D token stream.  The instant you do
that to ASCII art you have destroyed the only signal that mattered: *where*
each glyph sits relative to every other glyph.

This module is the foundation that refuses to flatten.  A `Canvas` is a padded
rectangular character matrix with O(1) coordinate access, glyph-class maps,
occupancy maps, and a battery of geometric primitives that the chunker, the
parser and the spatial encoder all build on.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
#  Glyph taxonomy
# --------------------------------------------------------------------------
# Box-drawing characters (ASCII + Unicode), the skeleton of any diagram.
BOX_CORNERS = set("+┌┐└┘╔╗╚╝╭╮╯╰┏┓┗┛")
BOX_HORIZ = set("-─━═_~")
BOX_VERT = set("|│┃║¦")
BOX_JUNCTION = set("+┼├┤┬┴╬╠╣╦╩┣┫┳┻")
BOX_DIAG = set("/\\╱╲")
# NOTE: ASCII 'v' / 'V' are deliberately NOT in this set.  They are letters far
# more often than they are arrowheads ("Svc", "Archive", "VLAN").  They are
# promoted to arrowheads *contextually* by ascii_parser.is_arrow_glyph(), which
# requires a vertical wire directly above and no adjacent letters.
ARROWHEADS = set("><^▲▼◄►→←↑↓»«")
SOFT_ARROWHEADS = set("vV^")
FILL_SHADES = set("░▒▓█▄▀▌▐■□●○◆◇★☆*#@%&.:;'`\"")

ALL_BOX = BOX_CORNERS | BOX_HORIZ | BOX_VERT | BOX_JUNCTION | BOX_DIAG

#: Glyph classes, in the fixed order used by the spatial encoder.
GLYPH_CLASSES = (
    "space",      # 0
    "corner",     # 1
    "horizontal", # 2
    "vertical",   # 3
    "junction",   # 4
    "diagonal",   # 5
    "arrow",      # 6
    "shade",      # 7
    "alpha",      # 8
    "digit",      # 9
    "punct",      # 10
)
N_GLYPH_CLASSES = len(GLYPH_CLASSES)


def classify(ch: str) -> int:
    """Map a single character onto its glyph class index."""
    if ch == " " or ch == "\t" or ch == " ":
        return 0
    if ch in BOX_CORNERS:
        return 1
    if ch in BOX_JUNCTION:
        return 4
    if ch in BOX_HORIZ:
        return 2
    if ch in BOX_VERT:
        return 3
    if ch in BOX_DIAG:
        return 5
    if ch in ARROWHEADS:
        return 6
    if ch in FILL_SHADES:
        return 7
    if ch.isdigit():
        return 9
    if ch.isalpha():
        return 8
    return 10


# Vectorised classifier cache for ASCII fast-path
_ASCII_CLASS = np.array([classify(chr(i)) for i in range(128)], dtype=np.int8)


@dataclass
class BBox:
    """Inclusive-exclusive bounding box in (row, col) space."""
    r0: int
    c0: int
    r1: int   # exclusive
    c1: int   # exclusive

    @property
    def height(self) -> int:
        return max(0, self.r1 - self.r0)

    @property
    def width(self) -> int:
        return max(0, self.c1 - self.c0)

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.r0 + self.r1) / 2.0, (self.c0 + self.c1) / 2.0)

    def contains_point(self, r: int, c: int) -> bool:
        return self.r0 <= r < self.r1 and self.c0 <= c < self.c1

    def intersects(self, other: "BBox") -> bool:
        return not (
            self.r1 <= other.r0
            or other.r1 <= self.r0
            or self.c1 <= other.c0
            or other.c1 <= self.c0
        )

    def iou(self, other: "BBox") -> float:
        if not self.intersects(other):
            return 0.0
        ir0, ic0 = max(self.r0, other.r0), max(self.c0, other.c0)
        ir1, ic1 = min(self.r1, other.r1), min(self.c1, other.c1)
        inter = (ir1 - ir0) * (ic1 - ic0)
        union = self.area + other.area - inter
        return inter / union if union else 0.0

    def overlap_fraction(self, other: "BBox") -> float:
        """Fraction of *self* covered by *other*."""
        if not self.intersects(other) or self.area == 0:
            return 0.0
        ir0, ic0 = max(self.r0, other.r0), max(self.c0, other.c0)
        ir1, ic1 = min(self.r1, other.r1), min(self.c1, other.c1)
        return ((ir1 - ir0) * (ic1 - ic0)) / self.area

    def expand(self, pad: int) -> "BBox":
        return BBox(self.r0 - pad, self.c0 - pad, self.r1 + pad, self.c1 + pad)

    def to_dict(self) -> dict:
        return {"r0": self.r0, "c0": self.c0, "r1": self.r1, "c1": self.c1,
                "h": self.height, "w": self.width}

    @staticmethod
    def from_dict(d: dict) -> "BBox":
        return BBox(d["r0"], d["c0"], d["r1"], d["c1"])


@dataclass
class Canvas:
    """A rectangular character matrix with derived geometric maps."""

    grid: np.ndarray                 # (H, W) dtype='<U1'
    classes: np.ndarray = field(init=False)    # (H, W) int8 glyph classes
    occupancy: np.ndarray = field(init=False)  # (H, W) bool  — non-space
    structure: np.ndarray = field(init=False)  # (H, W) bool  — box-drawing only
    text_mask: np.ndarray = field(init=False)  # (H, W) bool  — alnum/punct

    def __post_init__(self) -> None:
        self._rebuild()

    # -- construction ------------------------------------------------------
    @staticmethod
    def from_text(text: str, *, expand_tabs: int = 4) -> "Canvas":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if expand_tabs:
            text = text.expandtabs(expand_tabs)
        lines = text.split("\n")
        # Trim trailing fully-blank lines but keep internal ones.
        while lines and lines[-1].strip() == "":
            lines.pop()
        while lines and lines[0].strip() == "":
            lines.pop(0)
        if not lines:
            lines = [""]
        # Normalise width-2 CJK away so column maths stays honest.
        lines = [_narrow(line) for line in lines]
        width = max((len(l) for l in lines), default=1) or 1
        height = len(lines)
        grid = np.full((height, width), " ", dtype="<U1")
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                grid[r, c] = ch
        return Canvas(grid)

    def _rebuild(self) -> None:
        h, w = self.grid.shape
        cls = np.zeros((h, w), dtype=np.int8)
        # Vectorised for the common ASCII case, python loop only for exotica.
        codes = np.vectorize(ord, otypes=[np.int32])(self.grid) if self.grid.size else np.zeros((h, w), np.int32)
        ascii_mask = codes < 128
        if ascii_mask.any():
            cls[ascii_mask] = _ASCII_CLASS[codes[ascii_mask]]
        if (~ascii_mask).any():
            idx = np.argwhere(~ascii_mask)
            for r, c in idx:
                cls[r, c] = classify(str(self.grid[r, c]))
        self.classes = cls
        self.occupancy = cls != 0
        self.structure = np.isin(cls, [1, 2, 3, 4, 5, 6])
        self.text_mask = np.isin(cls, [8, 9, 10])

    # -- geometry ----------------------------------------------------------
    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def bbox(self) -> BBox:
        return BBox(0, 0, self.height, self.width)

    def at(self, r: int, c: int) -> str:
        if 0 <= r < self.height and 0 <= c < self.width:
            return str(self.grid[r, c])
        return " "

    def cls_at(self, r: int, c: int) -> int:
        if 0 <= r < self.height and 0 <= c < self.width:
            return int(self.classes[r, c])
        return 0

    def slice(self, box: BBox) -> np.ndarray:
        r0 = max(0, box.r0); c0 = max(0, box.c0)
        r1 = min(self.height, box.r1); c1 = min(self.width, box.c1)
        if r1 <= r0 or c1 <= c0:
            return np.full((0, 0), " ", dtype="<U1")
        return self.grid[r0:r1, c0:c1]

    def text_of(self, box: BBox, *, strip: bool = True) -> str:
        sub = self.slice(box)
        if sub.size == 0:
            return ""
        lines = ["".join(row) for row in sub]
        if strip:
            lines = [l.rstrip() for l in lines]
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
        return "\n".join(lines)

    def readable_text(self, box: BBox) -> str:
        """Alphanumeric content inside a box, collapsed to a clean phrase."""
        sub_cls = self.classes[max(0, box.r0):box.r1, max(0, box.c0):box.c1]
        sub = self.slice(box)
        if sub.size == 0:
            return ""
        keep = np.isin(sub_cls, [8, 9]) | (sub == " ") | np.isin(sub, list("._-/()&,'"))
        chars = np.where(keep, sub, " ")
        words: List[str] = []
        for row in chars:
            words.extend("".join(row).split())
        out = " ".join(words).strip(" ._-")
        return out

    def density(self, box: Optional[BBox] = None) -> float:
        box = box or self.bbox
        r0 = max(0, box.r0); c0 = max(0, box.c0)
        r1 = min(self.height, box.r1); c1 = min(self.width, box.c1)
        if r1 <= r0 or c1 <= c0:
            return 0.0
        sub = self.occupancy[r0:r1, c0:c1]
        return float(sub.mean())

    def to_text(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self.grid)

    # -- directional probing ----------------------------------------------
    def ray(self, r: int, c: int, dr: int, dc: int, max_len: int = 4096
            ) -> Iterable[Tuple[int, int, str]]:
        """Walk a ray from (r,c) exclusive, yielding (row, col, char)."""
        rr, cc = r + dr, c + dc
        n = 0
        while 0 <= rr < self.height and 0 <= cc < self.width and n < max_len:
            yield rr, cc, str(self.grid[rr, cc])
            rr += dr
            cc += dc
            n += 1

    def row_text(self, r: int) -> str:
        if 0 <= r < self.height:
            return "".join(self.grid[r]).rstrip()
        return ""

    def col_text(self, c: int) -> str:
        if 0 <= c < self.width:
            return "".join(self.grid[:, c]).rstrip()
        return ""

    # -- stamping (used by the generative engines) -------------------------
    def stamp(self, block: Sequence[str], r: int, c: int, *,
              transparent: str = " ") -> None:
        for i, line in enumerate(block):
            rr = r + i
            if not (0 <= rr < self.height):
                continue
            for j, ch in enumerate(line):
                cc = c + j
                if not (0 <= cc < self.width):
                    continue
                if ch == transparent:
                    continue
                self.grid[rr, cc] = ch
        self._rebuild()

    @staticmethod
    def blank(height: int, width: int, fill: str = " ") -> "Canvas":
        return Canvas(np.full((max(1, height), max(1, width)), fill, dtype="<U1"))


def _narrow(line: str) -> str:
    """Replace full-width glyphs so that one char == one column, always."""
    out = []
    for ch in line:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            out.append(ch)
            out.append("​")       # zero-width filler keeps columns honest
        else:
            out.append(ch)
    return "".join(out).replace("​", " ") if any(
        unicodedata.east_asian_width(c) in ("W", "F") for c in line
    ) else line


def normalise_block(block: str) -> List[str]:
    """Split a multi-line string into a left-trimmed, padded block."""
    lines = block.replace("\r\n", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return [""]
    indent = min((len(l) - len(l.lstrip(" ")) for l in lines if l.strip()), default=0)
    lines = [l[indent:] if len(l) >= indent else l for l in lines]
    w = max(len(l) for l in lines)
    return [l.ljust(w) for l in lines]
