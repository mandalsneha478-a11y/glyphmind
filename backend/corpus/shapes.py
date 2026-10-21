"""
corpus/shapes.py — structural formulas for concrete poetry.

Apollinaire cut his calligrams by hand.  Here each shape is a *formula*: an
implicit function over normalised coordinates that answers "is this cell inside
the figure?", paired with an SVG outline so the same shape can be rendered as
vector typography.

The poetry engine retrieves a shape semantically — "grief" finds the vessel and
the rain, "freedom" finds the bird — then flows syllable-weighted text through
the mask the formula produces at the requested canvas size.

Coordinate convention for `fn(x, y)`:
    x ∈ [-1, 1]  left → right
    y ∈ [-1, 1]  top  → bottom
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List


def _tree(x: float, y: float) -> bool:
    if y > 0.55:                                   # trunk
        return abs(x) < 0.10
    if y > 0.42:
        return abs(x) < 0.14
    t = (y + 1.0) / 1.42                           # 0 at apex, 1 at canopy base
    tiers = 3
    seg = min(tiers - 1e-6, t * tiers)
    local = seg - math.floor(seg)
    half = 0.16 + 0.62 * (0.35 + 0.65 * local) * (0.45 + 0.55 * t)
    return abs(x) <= half


def _heart(x: float, y: float) -> bool:
    X, Y = x * 1.25, -(y * 1.25) + 0.25
    v = (X * X + Y * Y - 1.0)
    return v * v * v - X * X * Y * Y * Y <= 0.0


def _spiral(x: float, y: float) -> bool:
    r = math.hypot(x, y)
    if r < 0.05 or r > 0.98:
        return r < 0.05
    th = math.atan2(y, x)
    for k in range(-2, 4):
        target = (th + 2 * math.pi * k) * 0.118
        if abs(r - target) < 0.085:
            return True
    return False


def _wave(x: float, y: float) -> bool:
    band = 0.30 * math.sin(math.pi * 1.7 * x) + 0.12 * math.sin(math.pi * 4.3 * x + 1.1)
    return abs(y - band) < 0.34


def _hourglass(x: float, y: float) -> bool:
    if abs(y) > 0.96:
        return abs(x) < 0.92
    half = 0.12 + 0.80 * abs(y)
    return abs(x) <= half


def _bird(x: float, y: float) -> bool:
    if abs(x) < 0.12:                              # body
        return -0.25 < y < 0.55
    w = abs(x)
    top = -0.62 + 0.70 * (w - 0.12) / 0.88
    return top - 0.26 < y < top + 0.20 and w < 1.0


def _mountain(x: float, y: float) -> bool:
    h = 0.92 * math.exp(-((x + 0.22) ** 2) / 0.17) + \
        0.66 * math.exp(-((x - 0.46) ** 2) / 0.10)
    return y > (0.85 - 1.70 * h)


def _circle(x: float, y: float) -> bool:
    return x * x + y * y <= 0.92


def _ring(x: float, y: float) -> bool:
    r = math.hypot(x, y)
    return 0.52 <= r <= 0.95


def _cross(x: float, y: float) -> bool:
    return (abs(x) < 0.20 and abs(y) < 0.95) or (abs(y - (-0.22)) < 0.19 and abs(x) < 0.68)


def _flame(x: float, y: float) -> bool:
    t = (1.0 - y) / 2.0
    half = 0.72 * math.sin(math.pi * max(0.0, min(1.0, t)) ** 0.75)
    sway = 0.16 * math.sin(3.1 * y + 0.6)
    return abs(x - sway) <= half


def _vessel(x: float, y: float) -> bool:
    if y > 0.82:
        return abs(x) < 0.46
    if y < -0.78:
        return abs(x) < 0.30
    if y < -0.52:
        return abs(x) < 0.22
    t = (y + 0.52) / 1.34
    half = 0.24 + 0.52 * math.sin(math.pi * min(1.0, max(0.0, t)) ** 0.85)
    return abs(x) <= half


def _rain(x: float, y: float) -> bool:
    col = (x + 1.0) * 5.0
    k = math.floor(col)
    phase = (col - k)
    off = ((k * 37) % 13) / 13.0
    yy = (y + 1.0 + off) % 1.25
    return abs(phase - 0.5) < 0.30 and yy < 0.72


def _butterfly(x: float, y: float) -> bool:
    if abs(x) < 0.07:
        return abs(y) < 0.82
    X, Y = abs(x), y
    up = (X - 0.40) ** 2 / 0.14 + (Y + 0.34) ** 2 / 0.13 <= 1.0
    dn = (X - 0.34) ** 2 / 0.10 + (Y - 0.40) ** 2 / 0.11 <= 1.0
    return up or dn


def _infinity(x: float, y: float) -> bool:
    X = x * 1.45
    v = (X * X + y * y) ** 2 - 1.15 * (X * X - y * y)
    return abs(v) < 0.16


def _star(x: float, y: float) -> bool:
    r = math.hypot(x, y)
    if r < 1e-6:
        return True
    th = math.atan2(y, x)
    k = 0.46 + 0.52 * abs(math.cos(2.5 * th))
    return r <= k


def _moon(x: float, y: float) -> bool:
    return (x * x + y * y <= 0.92) and ((x - 0.52) ** 2 + y * y > 0.80)


def _leaf(x: float, y: float) -> bool:
    if abs(x) < 0.045 and y > -0.1:
        return True
    X, Y = x, y
    return (X * X) / 0.30 + (Y * Y) / 0.95 <= 1.0 and (Y - 2.2 * X * X) < 0.85


def _lightning(x: float, y: float) -> bool:
    t = (y + 1.0) / 2.0
    centre = -0.45 + 1.0 * t + (0.34 if t > 0.5 else -0.10)
    return abs(x - centre) < 0.20


def _door(x: float, y: float) -> bool:
    if y < -0.52:
        return abs(x) <= 0.58 * math.sqrt(max(0.0, 1 - ((y + 0.52) / 0.46) ** 2))
    return abs(x) <= 0.58


def _eye(x: float, y: float) -> bool:
    outer = (x * x) / 0.95 + (y * y) / 0.22 <= 1.0
    pupil = (x * x + y * y) <= 0.055
    return outer and not ((x * x + y * y) <= 0.105 and not pupil)


SHAPE_DEFS: List[Dict] = [
    {"id": "shape.tree", "name": "Tree", "fn": _tree, "aspect": 0.62,
     "tags": "tree growth nature roots family life forest season memory time patience",
     "svg": "M50 4 L70 30 L60 30 L78 54 L64 54 L84 80 L56 80 L56 96 L44 96 L44 80 L16 80 L36 54 L22 54 L40 30 L30 30 Z"},
    {"id": "shape.heart", "name": "Heart", "fn": _heart, "aspect": 0.80,
     "tags": "love heart romance affection longing desire tenderness valentine warmth",
     "svg": "M50 92 C10 62 6 32 24 20 C38 11 50 22 50 32 C50 22 62 11 76 20 C94 32 90 62 50 92 Z"},
    {"id": "shape.spiral", "name": "Spiral", "fn": _spiral, "aspect": 0.90,
     "tags": "spiral obsession recursion memory descent vertigo infinity thought dizzy",
     "svg": "M50 50 m0 -4 a4 4 0 1 1 -4 4 a10 10 0 1 1 10 10 a18 18 0 1 1 -18 -18 a28 28 0 1 1 28 28 a38 38 0 1 1 -38 -38"},
    {"id": "shape.wave", "name": "Wave", "fn": _wave, "aspect": 0.42,
     "tags": "sea ocean wave water tide rhythm flow grief crossing voyage distance",
     "svg": "M2 55 C20 25 34 85 50 55 C66 25 80 85 98 55 L98 78 C80 48 66 108 50 78 C34 48 20 108 2 78 Z"},
    {"id": "shape.hourglass", "name": "Hourglass", "fn": _hourglass, "aspect": 0.95,
     "tags": "time hourglass mortality waiting patience deadline age sand ending",
     "svg": "M12 4 L88 4 L56 50 L88 96 L12 96 L44 50 Z"},
    {"id": "shape.bird", "name": "Bird in Flight", "fn": _bird, "aspect": 0.50,
     "tags": "bird flight freedom escape wings migration departure soul hope leaving",
     "svg": "M4 24 C26 30 40 44 50 60 C60 44 74 30 96 24 C78 44 64 54 50 76 C36 54 22 44 4 24 Z"},
    {"id": "shape.mountain", "name": "Mountain", "fn": _mountain, "aspect": 0.55,
     "tags": "mountain climb struggle ambition height distance solitude stone endurance",
     "svg": "M2 96 L34 18 L52 52 L68 30 L98 96 Z"},
    {"id": "shape.circle", "name": "Circle", "fn": _circle, "aspect": 1.0,
     "tags": "circle whole unity completeness moon sun cycle return perfection",
     "svg": "M50 4 a46 46 0 1 0 0.1 0 Z"},
    {"id": "shape.ring", "name": "Ring", "fn": _ring, "aspect": 1.0,
     "tags": "ring void absence emptiness marriage orbit hollow halo loss",
     "svg": "M50 2 a48 48 0 1 0 0.1 0 Z M50 26 a24 24 0 1 1 -0.1 0 Z"},
    {"id": "shape.cross", "name": "Cross", "fn": _cross, "aspect": 0.80,
     "tags": "cross faith sacrifice burden religion crossroads decision church grief",
     "svg": "M40 2 L60 2 L60 34 L96 34 L96 54 L60 54 L60 98 L40 98 L40 54 L4 54 L4 34 L40 34 Z"},
    {"id": "shape.flame", "name": "Flame", "fn": _flame, "aspect": 0.62,
     "tags": "fire flame anger passion burning rage energy heat destruction hunger",
     "svg": "M50 2 C64 26 84 38 80 62 C77 84 64 96 50 96 C36 96 22 84 20 62 C17 38 36 26 50 2 Z"},
    {"id": "shape.vessel", "name": "Vessel", "fn": _vessel, "aspect": 0.72,
     "tags": "vessel vase urn container grief holding body memory fragile keeping",
     "svg": "M34 4 L66 4 L60 20 C86 34 86 72 66 92 L34 92 C14 72 14 34 40 20 Z"},
    {"id": "shape.rain", "name": "Rain", "fn": _rain, "aspect": 0.70,
     "tags": "rain sorrow weather melancholy falling grey tears autumn city window",
     "svg": "M10 6 L6 40 M30 14 L26 52 M50 4 L46 44 M70 16 L66 56 M90 8 L86 46"},
    {"id": "shape.butterfly", "name": "Butterfly", "fn": _butterfly, "aspect": 0.80,
     "tags": "butterfly transformation change rebirth fragility beauty summer brief",
     "svg": "M50 14 L50 86 M50 40 C28 6 4 18 10 42 C14 62 36 60 50 50 C64 60 86 62 90 42 C96 18 72 6 50 40 Z"},
    {"id": "shape.infinity", "name": "Infinity", "fn": _infinity, "aspect": 0.45,
     "tags": "infinity eternity forever loop endless repetition always bond",
     "svg": "M50 50 C40 26 6 26 6 50 C6 74 40 74 50 50 C60 26 94 26 94 50 C94 74 60 74 50 50 Z"},
    {"id": "shape.star", "name": "Star", "fn": _star, "aspect": 0.95,
     "tags": "star night wish hope guidance distance cosmos light navigation dream",
     "svg": "M50 2 L61 36 L97 36 L68 57 L79 92 L50 70 L21 92 L32 57 L3 36 L39 36 Z"},
    {"id": "shape.moon", "name": "Crescent", "fn": _moon, "aspect": 1.0,
     "tags": "moon night crescent sleep dream tide month quiet secret dark",
     "svg": "M50 2 a48 48 0 1 0 26 88 a40 40 0 1 1 -26 -88 Z"},
    {"id": "shape.leaf", "name": "Leaf", "fn": _leaf, "aspect": 0.55,
     "tags": "leaf autumn fall season fragile green plant small quiet detail",
     "svg": "M50 2 C84 30 80 74 50 98 C20 74 16 30 50 2 Z M50 20 L50 94"},
    {"id": "shape.lightning", "name": "Lightning", "fn": _lightning, "aspect": 0.50,
     "tags": "lightning shock sudden revelation storm anger violence strike insight",
     "svg": "M58 2 L22 56 L46 56 L34 98 L78 40 L52 40 Z"},
    {"id": "shape.door", "name": "Doorway", "fn": _door, "aspect": 0.72,
     "tags": "door threshold entrance leaving arrival choice home exile opening",
     "svg": "M12 98 L12 40 a38 38 0 0 1 76 0 L88 98 Z"},
    {"id": "shape.eye", "name": "Eye", "fn": _eye, "aspect": 0.40,
     "tags": "eye seeing watching surveillance witness vision attention knowing",
     "svg": "M2 50 C24 20 76 20 98 50 C76 80 24 80 2 50 Z M50 36 a14 14 0 1 0 0.1 0 Z"},
]


def all_shapes() -> List[Dict]:
    return [{**{k: v for k, v in s.items()}, "kind": "shape"} for s in SHAPE_DEFS]


def shape_by_id(sid: str) -> Dict:
    for s in SHAPE_DEFS:
        if s["id"] == sid:
            return s
    return SHAPE_DEFS[0]


def shape_catalog() -> List[Dict]:
    """JSON-safe listing (drops the callables) for the UI picker."""
    return [{"id": s["id"], "name": s["name"], "tags": s["tags"],
             "aspect": s["aspect"], "svg": s["svg"]} for s in SHAPE_DEFS]
