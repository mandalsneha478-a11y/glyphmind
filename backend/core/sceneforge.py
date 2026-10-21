"""
sceneforge.py — procedural ASCII scene synthesis.

Why this module replaced sprite stamping
----------------------------------------
A fixed library of hand-drawn sprites can only ever draw the things somebody
already drew. Ask it for "a lighthouse in a storm" and it has nothing to say.
Worse, stamping a 35x7 sprite onto a 104x36 canvas leaves 90 % of the picture
empty, which reads as thin rather than detailed.

So the Architect does not stamp pictures. It retrieves **parametric generators**
and runs them. Every element here is a function of (region, parameters, rng)
that synthesises detail at whatever resolution it is handed: a fractal ridge
line computes a new mountain silhouette every time, a skyline grows buildings
with lit windows, an ocean derives wave rows with correct perspective
compression, and a reflection is computed from whatever happens to be above it.

That is what makes the output dense and what makes arbitrary prompts viable:
retrieval chooses *which* generators run in *which* depth layer with *what*
parameters, and the generators fill the canvas.

Rendering model
---------------
Painter's algorithm over six depth layers — sky, far, mid, near, foreground,
overlay. Each cell records the glyph, a luminance value in [0,1], the layer,
and which scene element painted it. Luminance plus layer drives the ANSI
colouring downstream, so colour is assigned semantically rather than by
position.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
#  Ramps — ordered light → dark. Luminance indexes into these.
# --------------------------------------------------------------------------
RAMP_BLOCK = " ░▒▓█"
RAMP_FINE = " .:-=+*#%@"
RAMP_SOFT = " .·:∘•●"
RAMP_WAVE = " .~-≈≋"
RAMP_STAR = " .·*✦★"
RAMP_ORGANIC = " .,;'\"^&%@"
RAMP_HALF = " ▁▂▃▄▅▆▇█"

LAYERS = ("sky", "far", "mid", "near", "fg", "overlay")
LAYER_INDEX = {n: i for i, n in enumerate(LAYERS)}


def ink(ramp: str, v: float) -> str:
    """Map a luminance to a glyph on a ramp."""
    if v <= 0.0:
        return " "
    i = int(max(0.0, min(0.999, v)) * (len(ramp) - 1) + 0.5)
    return ramp[i]


# --------------------------------------------------------------------------
#  Noise and fractal primitives
# --------------------------------------------------------------------------
def value_noise(h: int, w: int, scale: float, rng: random.Random,
                octaves: int = 3) -> np.ndarray:
    """Fractal value noise in [0,1] via bilinear upsampling of coarse grids."""
    seed = rng.randrange(1 << 30)
    nprng = np.random.default_rng(seed)
    acc = np.zeros((h, w), dtype=np.float32)
    amp, total = 1.0, 0.0
    s = max(2.0, scale)
    for _ in range(max(1, octaves)):
        gh = max(2, int(h / s) + 2)
        gw = max(2, int(w / s) + 2)
        grid = nprng.random((gh, gw)).astype(np.float32)
        ys = np.linspace(0, gh - 1, h)
        xs = np.linspace(0, gw - 1, w)
        y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, gh - 1)
        x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, gw - 1)
        ty = (ys - y0)[:, None]; tx = (xs - x0)[None, :]
        # smoothstep for continuity
        ty = ty * ty * (3 - 2 * ty); tx = tx * tx * (3 - 2 * tx)
        a = grid[np.ix_(y0, x0)]; b = grid[np.ix_(y0, x1)]
        c = grid[np.ix_(y1, x0)]; d = grid[np.ix_(y1, x1)]
        acc += amp * ((a * (1 - tx) + b * tx) * (1 - ty) +
                      (c * (1 - tx) + d * tx) * ty)
        total += amp
        amp *= 0.5
        s *= 0.5
    return acc / max(1e-6, total)


def fractal_line(w: int, rng: random.Random, roughness: float = 0.55,
                 lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Midpoint-displacement ridge of length w, normalised into [lo, hi]."""
    n = 1
    while n < w:
        n *= 2
    arr = np.zeros(n + 1, dtype=np.float32)
    arr[0] = rng.random()
    arr[-1] = rng.random()
    step = n
    amp = 1.0
    while step > 1:
        half = step // 2
        for i in range(half, n, step):
            arr[i] = 0.5 * (arr[i - half] + arr[i + half]) + (rng.random() - 0.5) * amp
        step = half
        amp *= roughness
    out = arr[:w] if w <= len(arr) else np.pad(arr, (0, w - len(arr)), mode="edge")
    mn, mx = float(out.min()), float(out.max())
    if mx - mn < 1e-6:
        out = np.full(w, 0.5, dtype=np.float32)
    else:
        out = (out - mn) / (mx - mn)
    return lo + out * (hi - lo)


# --------------------------------------------------------------------------
#  Raster
# --------------------------------------------------------------------------
class Raster:
    def __init__(self, h: int, w: int) -> None:
        self.h, self.w = h, w
        self.ch = np.full((h, w), " ", dtype="<U1")
        self.v = np.zeros((h, w), dtype=np.float32)      # luminance
        self.lay = np.full((h, w), -1, dtype=np.int8)    # depth layer
        self.elem = np.full((h, w), -1, dtype=np.int16)  # scene element id
        self.horizon = int(h * 0.62)

    def put(self, r: int, c: int, glyph: str, v: float, lay: int, elem: int,
            *, over_space_only: bool = False) -> None:
        if not (0 <= r < self.h and 0 <= c < self.w):
            return
        if glyph == " " or glyph == "":
            return
        if over_space_only and self.ch[r, c] != " ":
            return
        self.ch[r, c] = glyph
        self.v[r, c] = v
        self.lay[r, c] = lay
        self.elem[r, c] = elem

    def hline(self, r: int, c0: int, c1: int, glyph: str, v: float,
              lay: int, elem: int) -> None:
        for c in range(max(0, c0), min(self.w, c1)):
            self.put(r, c, glyph, v, lay, elem)

    def vline(self, c: int, r0: int, r1: int, glyph: str, v: float,
              lay: int, elem: int) -> None:
        for r in range(max(0, r0), min(self.h, r1)):
            self.put(r, c, glyph, v, lay, elem)

    def text(self) -> str:
        return "\n".join("".join(row) for row in self.ch)

    def ink_count(self) -> int:
        return int((self.ch != " ").sum())


# --------------------------------------------------------------------------
#  Generator context
# --------------------------------------------------------------------------
class Ctx:
    """Everything a generator needs, packaged so signatures stay uniform."""

    def __init__(self, R: Raster, rng: random.Random, elem: int, layer: str,
                 params: Dict, mood: Dict) -> None:
        self.R = R
        self.rng = rng
        self.elem = elem
        self.layer = LAYER_INDEX.get(layer, 1)
        self.p = params or {}
        self.mood = mood or {}

    def f(self, key: str, default: float) -> float:
        try:
            return float(self.p.get(key, default))
        except Exception:
            return default

    def i(self, key: str, default: int) -> int:
        try:
            return int(self.p.get(key, default))
        except Exception:
            return default

    @property
    def night(self) -> bool:
        return bool(self.mood.get("night"))


# ==========================================================================
#  SKY LAYER
# ==========================================================================
def gen_stars(x: Ctx) -> None:
    R = x.R
    top, bot = 0, int(R.horizon * x.f("extent", 1.0))
    dens = x.f("density", 0.055)
    for r in range(top, max(1, bot)):
        fall = 1.0 - (r / max(1, bot)) * 0.55        # thinner toward the horizon
        for c in range(R.w):
            if x.rng.random() < dens * fall:
                b = x.rng.random()
                g = "·" if b < 0.52 else ("*" if b < 0.82 else ("✦" if b < 0.95 else "★"))
                R.put(r, c, g, 0.35 + 0.65 * b, x.layer, x.elem, over_space_only=True)


def gen_moon(x: Ctx) -> None:
    R = x.R
    rad = max(3.0, x.f("radius", min(R.w, R.h) * 0.11))
    cy = x.f("cy", rad + 2)
    cx = x.f("cx", R.w * x.f("cx_frac", 0.78))
    crescent = x.f("crescent", 0.55)
    for r in range(int(cy - rad - 1), int(cy + rad + 2)):
        for c in range(int(cx - rad * 2 - 2), int(cx + rad * 2 + 3)):
            dy = (r - cy)
            dx = (c - cx) * 0.5                       # cell aspect
            d = math.hypot(dx, dy)
            if d > rad:
                continue
            if crescent > 0:
                d2 = math.hypot((c - cx - rad * 2 * crescent) * 0.5, dy)
                if d2 < rad * 0.94:
                    continue
            v = 0.55 + 0.45 * (1.0 - d / rad)
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)


def gen_sun(x: Ctx) -> None:
    R = x.R
    rad = max(3.0, x.f("radius", min(R.w, R.h) * 0.12))
    cy = x.f("cy", rad + 3)
    cx = x.f("cx", R.w * x.f("cx_frac", 0.70))
    for r in range(int(cy - rad - 1), int(cy + rad + 2)):
        for c in range(int(cx - rad * 2 - 2), int(cx + rad * 2 + 3)):
            d = math.hypot((c - cx) * 0.5, r - cy)
            if d <= rad:
                v = 0.5 + 0.5 * (1 - d / rad)
                R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)
    if x.f("rays", 1.0) > 0:
        for k in range(12):
            a = k * math.pi / 6
            for t in range(int(rad) + 2, int(rad * 2.0) + 3):
                rr = int(cy + math.sin(a) * t)
                cc = int(cx + math.cos(a) * t * 2.0)
                g = "-" if abs(math.sin(a)) < 0.35 else ("|" if abs(math.cos(a)) < 0.35 else
                                                         ("\\" if math.sin(a) * math.cos(a) > 0 else "/"))
                R.put(rr, cc, g, 0.45, x.layer, x.elem, over_space_only=True)


def gen_clouds(x: Ctx) -> None:
    R = x.R
    bot = max(2, int(R.horizon * x.f("extent", 0.82)))
    n = value_noise(bot, R.w, x.f("scale", 11.0), x.rng, octaves=3)
    thr = x.f("threshold", 0.56)
    ramp = RAMP_BLOCK if x.f("solid", 0) > 0 else RAMP_SOFT
    for r in range(bot):
        band = 1.0 - abs((r / max(1, bot)) - x.f("band", 0.42)) * x.f("tight", 1.6)
        for c in range(R.w):
            val = n[r, c] * max(0.0, band)
            if val > thr:
                v = min(1.0, (val - thr) / max(1e-3, 1 - thr))
                R.put(r, c, ink(ramp, 0.25 + 0.75 * v), 0.25 + 0.6 * v,
                      x.layer, x.elem, over_space_only=True)


def gen_aurora(x: Ctx) -> None:
    R = x.R
    bot = max(3, int(R.horizon * 0.8))
    for band in range(x.i("bands", 3)):
        phase = x.rng.random() * math.tau
        amp = 2.0 + x.rng.random() * 4.0
        base = 3 + band * max(2, bot // 5)
        for c in range(R.w):
            y = base + amp * math.sin(c * 0.09 + phase) + 1.6 * math.sin(c * 0.027 + phase * 2)
            for k in range(x.i("thickness", 4)):
                rr = int(y + k)
                v = 0.85 - k * 0.18
                if v > 0.05:
                    R.put(rr, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)


def gen_planet(x: Ctx) -> None:
    R = x.R
    rad = max(4.0, x.f("radius", min(R.w * 0.5, R.h) * 0.30))
    cy = x.f("cy", R.h * 0.34)
    cx = x.f("cx", R.w * x.f("cx_frac", 0.70))
    n = value_noise(R.h, R.w, 7.0, x.rng, 3)
    for r in range(int(cy - rad - 2), int(cy + rad + 3)):
        for c in range(int(cx - rad * 2 - 3), int(cx + rad * 2 + 4)):
            d = math.hypot((c - cx) * 0.5, r - cy)
            if d > rad:
                continue
            lit = max(0.0, 1.0 - ((c - cx) * 0.5 - (r - cy) * 0.4 + rad) / (2 * rad))
            v = 0.2 + 0.8 * lit * (0.55 + 0.45 * float(n[r % R.h, c % R.w]))
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)
    if x.f("ring", 1.0) > 0:
        for c in range(R.w):
            dx = (c - cx) * 0.5
            if abs(dx) > rad * 2.1 or abs(dx) < rad * 1.02:
                continue
            rr = int(cy + dx * 0.22)
            R.put(rr, c, "─", 0.6, x.layer, x.elem)
            R.put(rr + 1, c, "·", 0.4, x.layer, x.elem, over_space_only=True)


def gen_storm_sky(x: Ctx) -> None:
    R = x.R
    bot = max(3, int(R.horizon * 0.80))
    n = value_noise(bot, R.w, 14.0, x.rng, 3)
    thr = x.f("threshold", 0.70)
    for r in range(bot):
        # weight toward the upper sky so the horizon stays readable
        band = 1.0 - (r / max(1, bot)) * 0.55
        for c in range(R.w):
            val = float(n[r, c]) * band
            if val <= thr:
                continue
            v = 0.25 + 0.55 * min(1.0, (val - thr) / max(1e-3, 1 - thr))
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem,
                  over_space_only=True)
    for _ in range(x.i("bolts", 2)):
        c = x.rng.randrange(6, max(7, R.w - 6))
        r = x.rng.randrange(1, max(2, bot // 2))
        while r < bot + 2:
            R.put(r, c, "\\" if x.rng.random() < 0.5 else "/", 1.0, x.layer, x.elem)
            c += x.rng.choice((-1, 0, 1, 1))
            r += 1


# ==========================================================================
#  FAR LAYER
# ==========================================================================
def _terrain(x: Ctx, ridge: np.ndarray, ramp: str, body_v: float,
             outline: bool = True, snow: float = 0.0) -> None:
    """Fill beneath a ridge line — but only as far as the horizon.

    Terrain belongs to the *far* layer, so it must stop where the ground plane
    begins. Letting it run to the bottom of the canvas was what turned every
    scene into a solid block: the near layer then had nothing left to say.
    """
    R = x.R
    floor = min(R.h, R.horizon + 1)
    for c in range(R.w):
        top = int(ridge[c])
        if top >= floor:
            continue
        if outline:
            prev = int(ridge[c - 1]) if c > 0 else top
            nxt = int(ridge[c + 1]) if c + 1 < R.w else top
            # A flat run gets a crisp half-block edge rather than a chain of
            # underscores: underscores read as noise once two ranges overlap,
            # whereas ▀ reads as a lit ridge and separates the depth planes.
            if prev > top and nxt > top:
                g = "^"
            elif nxt < top - 0:
                g = "/" if nxt < top else "▀"
            elif nxt > top:
                g = "\\"
            else:
                g = "▀"
            R.put(top, c, g, 0.95, x.layer, x.elem)
        span = max(1.0, floor - top)
        for r in range(top + 1, floor):
            depth = (r - top) / span
            v = body_v * (1.0 - depth * 0.35)
            if snow > 0 and (r - top) <= 1 and top < R.horizon * 0.55:
                R.put(r, c, "▒", 0.92, x.layer, x.elem)
            else:
                R.put(r, c, ink(ramp, v), v, x.layer, x.elem)


def gen_mountains(x: Ctx) -> None:
    R = x.R
    ranges = x.i("ranges", 3)
    for k in range(ranges):
        t = k / max(1, ranges - 1) if ranges > 1 else 0.0
        # Ridges live in the band just above the horizon. Letting them start
        # near the top of the canvas meant the fill below them swallowed the
        # sky, and every scene came out as one solid mass.
        hi = R.horizon * (0.40 + 0.20 * t)
        lo = R.horizon * (0.70 + 0.20 * t)
        ridge = fractal_line(R.w, x.rng, x.f("roughness", 0.48), hi, lo)
        # Distinct fill values per range are what make the depth planes
        # separate at a glance; identical greys collapse into one mass.
        _terrain(x, ridge, RAMP_BLOCK, 0.22 + 0.52 * t,
                 outline=True, snow=x.f("snow", 0.6) if k == 0 else 0.0)


def gen_hills(x: Ctx) -> None:
    R = x.R
    for k in range(x.i("ranges", 2)):
        t = k / max(1, x.i("ranges", 2) - 1)
        base = R.horizon * (0.76 + 0.14 * t)
        amp = R.horizon * 0.09
        ridge = np.array([base - amp * (math.sin(c * 0.055 + k * 1.9) +
                                        0.4 * math.sin(c * 0.021 + k))
                          for c in range(R.w)], dtype=np.float32)
        _terrain(x, ridge, RAMP_ORGANIC, 0.32 + 0.3 * t, outline=True)


def gen_dunes(x: Ctx) -> None:
    R = x.R
    for k in range(x.i("ranges", 3)):
        t = k / max(1, x.i("ranges", 3) - 1)
        base = R.horizon * (0.80 + 0.12 * t)
        ridge = np.array([base - R.horizon * 0.06 *
                          math.sin(c * (0.032 + 0.013 * k) + k * 2.2)
                          for c in range(R.w)], dtype=np.float32)
        _terrain(x, ridge, RAMP_SOFT, 0.3 + 0.3 * t, outline=True)


def gen_skyline(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * x.f("base", 0.98))
    c = 0
    lit = x.f("lit", 0.45)
    while c < R.w:
        bw = x.rng.randrange(x.i("min_w", 4), x.i("max_w", 11))
        bh = x.rng.randrange(int(R.horizon * 0.18), int(R.horizon * 0.72))
        top = max(1, base - bh)
        wall = "▓" if x.rng.random() < 0.6 else "█"
        for cc in range(c, min(R.w, c + bw)):
            R.put(top, cc, "▄", 0.85, x.layer, x.elem)
            for rr in range(top + 1, base + 1):
                R.put(rr, cc, wall, 0.42, x.layer, x.elem)
        # windows on a regular grid — this is what reads as "city" at a glance
        for rr in range(top + 2, base, 2):
            for cc in range(c + 1, min(R.w, c + bw - 1), 2):
                if x.rng.random() < lit:
                    R.put(rr, cc, "▪" if x.rng.random() < 0.7 else "□",
                          0.95, x.layer, x.elem)
        if bw > 5 and x.rng.random() < 0.4:                 # antenna
            ac = c + bw // 2
            for rr in range(max(0, top - x.rng.randrange(2, 6)), top):
                R.put(rr, ac, "|", 0.7, x.layer, x.elem)
        c += bw + x.rng.randrange(0, 2)


def gen_forest_line(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * x.f("base", 0.99))
    c = x.rng.randrange(0, 4)
    max_half = x.i("max_half", 4)
    while c < R.w + max_half:
        th = x.rng.randrange(x.i("min_h", 4), x.i("max_h", 9))
        top = max(0, base - th)
        half = 1
        for i, rr in enumerate(range(top, base)):
            half = min(max_half, int((i / max(1, th - 1)) * (th * 0.5)) + 1)
            for cc in range(c - half + 1, c + half):
                v = 0.26 + 0.34 * (i / max(1, th))
                R.put(rr, cc, ink(RAMP_ORGANIC, v), v, x.layer, x.elem)
            R.put(rr, c - half, "/", 0.82, x.layer, x.elem)
            R.put(rr, c + half, "\\", 0.82, x.layer, x.elem)
        R.put(base, c, "|", 0.7, x.layer, x.elem)
        # Step by a full crown width so neighbouring crowns touch rather than
        # overlap. Overlapping crowns merge into an unreadable band.
        c += max(3, 2 * half + x.rng.randrange(0, 3))


def gen_pyramids(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * 0.99)
    for k in range(x.i("count", 2)):
        cx = int(R.w * (0.26 + 0.42 * k) + x.rng.randrange(-6, 7))
        hgt = x.rng.randrange(int(R.horizon * 0.3), int(R.horizon * 0.62))
        for i in range(hgt):
            rr = base - hgt + i
            half = int(i * 1.35) + 1
            for cc in range(cx - half, cx + half + 1):
                lit = cc < cx
                v = 0.62 if lit else 0.28
                R.put(rr, cc, ink(RAMP_BLOCK, v), v, x.layer, x.elem)
            R.put(rr, cx - half, "/", 0.9, x.layer, x.elem)
            R.put(rr, cx + half, "\\", 0.9, x.layer, x.elem)


def gen_wall(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * 0.99)
    top = max(1, base - x.i("height", max(4, R.horizon // 3)))
    for c in range(R.w):
        for r in range(top, base + 1):
            brick = ((r // 1) + (c // 4) + (r % 2) * 2) % 8 == 0
            v = 0.5 if not brick else 0.22
            R.put(r, c, "▓" if not brick else "─", v, x.layer, x.elem)
    for c in range(0, R.w, 6):                      # crenellations
        for cc in range(c, min(R.w, c + 3)):
            R.put(top - 1, cc, "█", 0.7, x.layer, x.elem)


def gen_void(x: Ctx) -> None:
    """Deep space: nothing but a faint dust gradient. Lets the sky dominate."""
    R = x.R
    n = value_noise(R.h, R.w, 14.0, x.rng, 2)
    for r in range(R.h):
        for c in range(R.w):
            if n[r, c] > 0.80 and x.rng.random() < 0.25:
                R.put(r, c, "·", 0.25, x.layer, x.elem, over_space_only=True)


# ==========================================================================
#  MID LAYER — subjects
# ==========================================================================
def _branch(R: Raster, x: Ctx, r: float, c: float, ang: float, length: float,
            depth: int) -> None:
    if depth <= 0 or length < 1.2:
        return
    steps = max(1, int(length))
    for i in range(steps):
        rr = int(r - math.sin(ang) * i * 0.55)
        cc = int(c + math.cos(ang) * i)
        g = "|" if abs(math.cos(ang)) < 0.35 else ("/" if math.cos(ang) * math.sin(ang) > 0 else "\\")
        R.put(rr, cc, g, 0.45 + 0.1 * depth, x.layer, x.elem)
    nr = r - math.sin(ang) * steps * 0.55
    nc = c + math.cos(ang) * steps
    spread = 0.52 + x.rng.random() * 0.3
    _branch(R, x, nr, nc, ang + spread, length * 0.68, depth - 1)
    _branch(R, x, nr, nc, ang - spread, length * 0.68, depth - 1)
    if depth <= 2:
        rad = max(1.5, length * 0.9)
        for dr in range(int(-rad), int(rad) + 1):
            for dc in range(int(-rad * 1.8), int(rad * 1.8) + 1):
                if (dc / 1.8) ** 2 + dr ** 2 <= rad * rad:
                    v = 0.35 + 0.45 * x.rng.random()
                    R.put(int(nr + dr), int(nc + dc), ink(RAMP_ORGANIC, v), v,
                          x.layer, x.elem, over_space_only=True)


def gen_big_tree(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * x.f("base", 1.02))
    cx = int(R.w * x.f("cx_frac", 0.5))
    trunk_h = x.i("trunk", max(3, R.horizon // 5))
    for r in range(base - trunk_h, base + 1):
        for cc in range(cx - 1, cx + 2):
            R.put(r, cc, "█" if cc == cx else "▌", 0.55, x.layer, x.elem)
    _branch(R, x, base - trunk_h, cx, math.pi / 2, max(3.0, R.horizon * 0.16), 4)


def gen_castle(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * 1.0)
    cx = int(R.w * x.f("cx_frac", 0.5))
    bw = x.i("width", max(14, R.w // 4))
    bh = x.i("height", max(7, R.horizon // 2))
    l, r0 = cx - bw // 2, base - bh
    for r in range(r0, base + 1):
        for c in range(l, l + bw):
            R.put(r, c, "▓", 0.4, x.layer, x.elem)
    for c in range(l, l + bw, 3):
        R.put(r0 - 1, c, "█", 0.75, x.layer, x.elem)
        R.put(r0 - 1, c + 1, "█", 0.75, x.layer, x.elem)
    for tc in (l - 3, l + bw + 1):                      # flanking towers
        th = bh + x.rng.randrange(2, 5)
        for r in range(base - th, base + 1):
            for c in range(tc - 1, tc + 3):
                R.put(r, c, "▓", 0.45, x.layer, x.elem)
        for c in range(tc - 1, tc + 3):
            R.put(base - th - 1, c, "█", 0.8, x.layer, x.elem)
        R.put(base - th - 2, tc + 1, "|", 0.9, x.layer, x.elem)
        R.put(base - th - 2, tc + 2, "▶", 0.95, x.layer, x.elem)
    gw = max(3, bw // 5)
    for r in range(base - gw * 2, base + 1):
        for c in range(cx - gw // 2, cx + gw // 2 + 1):
            R.put(r, c, " " if False else "█", 0.08, x.layer, x.elem)
    for r in range(r0 + 2, base - gw * 2, 3):
        for c in range(l + 2, l + bw - 2, 4):
            R.put(r, c, "▪", 0.9, x.layer, x.elem)


def gen_tower(x: Ctx) -> None:
    """Lighthouse / spire — tapered body, lantern room, optional beam."""
    R = x.R
    base = int(R.horizon * 1.0)
    cx = int(R.w * x.f("cx_frac", 0.72))
    hgt = x.i("height", max(8, int(R.horizon * 0.8)))
    for i in range(hgt):
        r = base - i
        half = max(1, int(3.2 - i / max(1, hgt) * 1.9))
        for c in range(cx - half, cx + half + 1):
            band = (i // 3) % 2 == 0
            v = 0.62 if band else 0.3
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)
    top = base - hgt
    for c in range(cx - 3, cx + 4):
        R.put(top, c, "▄", 0.8, x.layer, x.elem)
        R.put(top - 1, c, "█" if abs(c - cx) > 1 else "☀", 0.98, x.layer, x.elem)
    R.put(top - 2, cx, "▲", 0.9, x.layer, x.elem)
    if x.f("beam", 1.0) > 0:
        for t in range(1, int(R.w * 0.45)):
            rr = top - 1 + int(t * 0.12)
            R.put(rr, cx + 4 + t, "-" if t % 3 else "=", max(0.15, 0.8 - t * 0.02),
                  x.layer, x.elem, over_space_only=True)


def gen_ship(x: Ctx) -> None:
    R = x.R
    wl = int(R.horizon * x.f("waterline", 1.06))
    cx = int(R.w * x.f("cx_frac", 0.35))
    hw = x.i("hull", max(6, R.w // 12))
    for i in range(3):
        r = wl - i
        half = hw - i
        for c in range(cx - half, cx + half + 1):
            R.put(r, c, "▓" if i else "▀", 0.5 - i * 0.1, x.layer, x.elem)
    R.put(wl - 2, cx - hw - 1, "\\", 0.7, x.layer, x.elem)
    R.put(wl - 2, cx + hw + 1, "/", 0.7, x.layer, x.elem)
    mast = x.i("mast", max(5, R.horizon // 4))
    for r in range(wl - 3 - mast, wl - 2):
        R.put(r, cx, "|", 0.8, x.layer, x.elem)
    for k, r in enumerate(range(wl - 2 - mast, wl - 3)):
        half = max(1, int((k + 1) * 0.9))
        for c in range(cx + 1, cx + half + 2):
            R.put(r, c, ink(RAMP_BLOCK, 0.72), 0.72, x.layer, x.elem)
        R.put(r, cx + half + 2, ")", 0.8, x.layer, x.elem)


def gen_rocket(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * 1.0)
    cx = int(R.w * x.f("cx_frac", 0.5))
    hgt = x.i("height", max(10, int(R.horizon * 0.85)))
    for i in range(hgt):
        r = base - i
        t = i / max(1, hgt)
        half = 1 if t > 0.86 else (2 if t > 0.72 else 3)
        for c in range(cx - half, cx + half + 1):
            edge = abs(c - cx) == half
            v = 0.85 if edge else (0.55 if (i // 4) % 2 else 0.3)
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)
    for i in range(4):                                   # fins
        R.put(base - i, cx - 4 - i // 2, "/", 0.8, x.layer, x.elem)
        R.put(base - i, cx + 4 + i // 2, "\\", 0.8, x.layer, x.elem)
    R.put(base - hgt - 1, cx, "▲", 0.95, x.layer, x.elem)
    for i in range(1, x.i("flame", 7)):                  # exhaust
        r = base + i
        half = max(0, 3 - i // 2)
        for c in range(cx - half, cx + half + 1):
            v = max(0.1, 1.0 - i * 0.14)
            R.put(r, c, ink(RAMP_FINE, v), v, x.layer, x.elem)


def gen_bridge(x: Ctx) -> None:
    R = x.R
    deck = int(R.horizon * x.f("deck", 0.96))
    for c in range(R.w):
        R.put(deck, c, "═", 0.7, x.layer, x.elem)
        R.put(deck + 1, c, "▀" if c % 4 else "╥", 0.4, x.layer, x.elem)
    for tx in (int(R.w * 0.25), int(R.w * 0.75)):
        th = x.i("tower", max(6, R.horizon // 3))
        for r in range(deck - th, deck):
            R.put(r, tx, "║", 0.8, x.layer, x.elem)
            R.put(r, tx + 1, "║", 0.8, x.layer, x.elem)
        R.put(deck - th - 1, tx, "╔", 0.9, x.layer, x.elem)
        R.put(deck - th - 1, tx + 1, "╗", 0.9, x.layer, x.elem)
    for c in range(R.w):
        t = abs(((c / R.w) * 2 - 1))
        y = deck - int((1 - t * t) * max(4, R.horizon // 3) * 0.9)
        R.put(y, c, "_" if c % 2 else ".", 0.55, x.layer, x.elem, over_space_only=True)


def gen_fire(x: Ctx) -> None:
    R = x.R
    base = int(R.horizon * x.f("base", 1.02))
    cx = int(R.w * x.f("cx_frac", 0.5))
    hgt = x.i("height", max(6, R.horizon // 3))
    for i in range(hgt):
        r = base - i
        t = i / max(1, hgt)
        half = max(0, int((1 - t) ** 0.7 * x.f("spread", 6)))
        sway = int(2.2 * math.sin(t * 3.0 + x.rng.random() * 0.4))
        for c in range(cx - half + sway, cx + half + sway + 1):
            if x.rng.random() < 0.18 + 0.4 * (1 - t):
                continue
            v = max(0.15, 1.0 - t * 0.85)
            R.put(r, c, ink(RAMP_FINE, v), v, x.layer, x.elem)
    for _ in range(x.i("sparks", 14)):
        R.put(base - x.rng.randrange(hgt, hgt + 6),
              cx + x.rng.randrange(-8, 9), "·", 0.9, x.layer, x.elem,
              over_space_only=True)


def gen_waterfall(x: Ctx) -> None:
    R = x.R
    cx = int(R.w * x.f("cx_frac", 0.5))
    wdt = x.i("width", max(5, R.w // 10))
    top = x.i("top", max(1, R.horizon // 4))
    bot = int(R.horizon * 1.0)
    for c in range(cx - wdt // 2, cx + wdt // 2 + 1):
        for r in range(top, bot):
            if x.rng.random() < 0.22:
                continue
            v = 0.35 + 0.55 * x.rng.random()
            R.put(r, c, "│" if x.rng.random() < 0.6 else "┆", v, x.layer, x.elem)
    for i in range(3):                                   # mist
        for c in range(cx - wdt, cx + wdt + 1):
            if x.rng.random() < 0.5:
                R.put(bot + i, c, ink(RAMP_SOFT, 0.5 - i * 0.12), 0.5, x.layer, x.elem)


def gen_portal(x: Ctx) -> None:
    R = x.R
    cy = int(R.horizon * 0.72)
    cx = int(R.w * x.f("cx_frac", 0.5))
    rad = x.f("radius", min(R.w * 0.5, R.h) * 0.26)
    for r in range(int(cy - rad - 1), int(cy + rad + 2)):
        for c in range(int(cx - rad * 2 - 2), int(cx + rad * 2 + 3)):
            d = math.hypot((c - cx) * 0.5, r - cy)
            if d > rad:
                continue
            edge = 1.0 - abs(d - rad * 0.86) / max(1.0, rad * 0.3)
            v = max(0.12, min(1.0, edge)) * (0.6 + 0.4 * x.rng.random())
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)


def gen_circuit(x: Ctx) -> None:
    R = x.R
    r0, r1 = int(R.h * 0.12), int(R.h * 0.9)
    for r in range(r0, r1, 3):
        c = x.rng.randrange(0, 6)
        while c < R.w:
            run = x.rng.randrange(4, 14)
            for cc in range(c, min(R.w, c + run)):
                R.put(r, cc, "─", 0.5, x.layer, x.elem)
            c += run
            if c < R.w:
                R.put(r, c, x.rng.choice("┼┬┴├┤"), 0.85, x.layer, x.elem)
                dr = x.rng.choice((-2, 2))
                for k in range(1, abs(dr) + 1):
                    R.put(r + (k if dr > 0 else -k), c, "│", 0.45, x.layer, x.elem)
                R.put(r + dr, c, "●" if x.rng.random() < 0.4 else "┐", 0.95,
                      x.layer, x.elem)
            c += x.rng.randrange(2, 7)


def gen_figure(x: Ctx) -> None:
    """A small human silhouette — gives a scene its sense of scale."""
    R = x.R
    base = int(R.horizon * x.f("base", 1.06))
    cx = int(R.w * x.f("cx_frac", 0.22))
    h = x.i("height", 6)
    R.put(base - h, cx, "●", 0.95, x.layer, x.elem)
    for i in range(1, h - 2):
        R.put(base - h + i, cx, "█", 0.8, x.layer, x.elem)
    R.put(base - h + 1, cx - 1, "/", 0.8, x.layer, x.elem)
    R.put(base - h + 1, cx + 1, "\\", 0.8, x.layer, x.elem)
    R.put(base - 1, cx - 1, "/", 0.8, x.layer, x.elem)
    R.put(base - 1, cx + 1, "\\", 0.8, x.layer, x.elem)


def gen_silhouette(x: Ctx) -> None:
    """Fill an implicit shape with volumetric shading.

    This is the universal subject renderer: any shape formula in the corpus —
    heart, butterfly, hourglass, bird, leaf — becomes a shaded solid lit from
    the upper left. It is what lets an unmatched noun still produce a figure.
    """
    R = x.R
    fn: Optional[Callable] = x.p.get("fn")
    if fn is None:
        return
    r0 = int(R.h * x.f("r0", 0.10)); r1 = int(R.h * x.f("r1", 0.92))
    cw = int(R.w * x.f("cw", 0.52))
    c0 = int(R.w * x.f("cx_frac", 0.5) - cw / 2)
    H = max(2, r1 - r0); W = max(2, cw)
    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            yy = (r + 0.5) / H * 2 - 1
            xx = (c + 0.5) / W * 2 - 1
            try:
                mask[r, c] = bool(fn(xx, yy))
            except Exception:
                mask[r, c] = False
    # distance-to-edge gives volume; the light sits upper-left
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            d = 0
            while d < 7:
                d += 1
                rr0, rr1 = max(0, r - d), min(H - 1, r + d)
                cc0, cc1 = max(0, c - d), min(W - 1, c + d)
                if (not mask[rr0, c] or not mask[rr1, c]
                        or not mask[r, cc0] or not mask[r, cc1]):
                    break
            lightness = 0.30 + 0.5 * min(1.0, d / 5.0)
            lightness += 0.18 * (1.0 - (r / H)) + 0.12 * (1.0 - (c / W))
            v = max(0.12, min(1.0, lightness))
            R.put(r0 + r, c0 + c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)


def gen_monument(x: Ctx) -> None:
    """Monumental typography as the subject.

    When retrieval finds no scene element for the noun the user asked for, the
    word itself becomes the architecture — which is the honest answer, and
    keeps the Architect's promise for genuinely arbitrary prompts.
    """
    R = x.R
    block: Sequence[str] = x.p.get("block") or []
    if not block:
        return
    bh = len(block); bw = max(len(l) for l in block)
    r0 = int(R.horizon * x.f("anchor", 0.92)) - bh
    c0 = (R.w - bw) // 2
    for i, line in enumerate(block):
        for j, g in enumerate(line):
            if g == " ":
                continue
            v = 0.55 + 0.4 * (1.0 - i / max(1, bh))
            R.put(r0 + i, c0 + j, g, v, x.layer, x.elem)
    for c in range(max(0, c0 - 2), min(R.w, c0 + bw + 2)):   # plinth
        R.put(r0 + bh, c, "▀", 0.6, x.layer, x.elem)
        R.put(r0 + bh + 1, c, "▒", 0.3, x.layer, x.elem)


# ==========================================================================
#  NEAR LAYER — ground planes
# ==========================================================================
def gen_ocean(x: Ctx) -> None:
    R = x.R
    top = int(R.horizon * x.f("top", 1.0))
    span = max(1, R.h - top)
    for r in range(top, R.h):
        t = (r - top) / span
        # Perspective: wave marks are short and tightly packed at the horizon,
        # long and widely spaced as the surface comes toward the viewer.
        period = max(4, int(14 - t * 9))
        seg = max(1, int(1 + t * 3))
        glyph = "~" if t < 0.32 else ("≈" if t < 0.70 else "≋")
        off = x.rng.randrange(0, period)
        for c in range(R.w):
            if (c + off + r * 5) % period < seg:
                v = 0.30 + 0.55 * t
                R.put(r, c, glyph, v, x.layer, x.elem)
            elif x.rng.random() < 0.02 + t * 0.05:
                R.put(r, c, "-", 0.18 + 0.25 * t, x.layer, x.elem)


def gen_reflection(x: Ctx) -> None:
    """Mirror whatever has already been drawn above the waterline."""
    R = x.R
    wl = int(R.horizon * x.f("waterline", 1.0))
    depth = R.h - wl
    for i in range(1, depth):
        src = wl - i * 1
        if src < 0:
            break
        fade = 1.0 - i / max(1, depth)
        for c in range(R.w):
            g = R.ch[src, c]
            if g == " " or R.lay[src, c] == LAYER_INDEX["overlay"]:
                continue
            if x.rng.random() > 0.30 + 0.5 * fade:
                continue
            jitter = x.rng.choice((-1, 0, 0, 1))
            v = R.v[src, c] * (0.25 + 0.45 * fade)
            R.put(wl + i, c + jitter, ink(RAMP_WAVE, v), v, x.layer, x.elem,
                  over_space_only=True)


def gen_grass_field(x: Ctx) -> None:
    R = x.R
    top = int(R.horizon * x.f("top", 1.0))
    n = value_noise(max(1, R.h - top), R.w, 6.0, x.rng, 2)
    for r in range(top, R.h):
        t = (r - top) / max(1, R.h - top)
        for c in range(R.w):
            # coverage grows toward the viewer: sparse at the horizon, dense
            # underfoot. A uniform fill reads as a wall, not a field.
            if x.rng.random() > 0.22 + 0.62 * t:
                continue
            v = 0.18 + 0.55 * t * (0.5 + 0.5 * float(n[r - top, c]))
            g = ink(RAMP_ORGANIC, v)
            if g != " ":
                R.put(r, c, g, v, x.layer, x.elem)


def gen_desert_floor(x: Ctx) -> None:
    R = x.R
    top = int(R.horizon * x.f("top", 1.0))
    for r in range(top, R.h):
        t = (r - top) / max(1, R.h - top)
        for c in range(R.w):
            if x.rng.random() < 0.08 + t * 0.35:
                R.put(r, c, ink(RAMP_SOFT, 0.2 + 0.5 * t), 0.2 + 0.5 * t,
                      x.layer, x.elem)
        if r % 3 == 0:
            for c in range(x.rng.randrange(0, 8), R.w, x.rng.randrange(7, 16)):
                R.put(r, c, "~", 0.35 + 0.3 * t, x.layer, x.elem)


def gen_snow_field(x: Ctx) -> None:
    R = x.R
    top = int(R.horizon * x.f("top", 1.0))
    for r in range(top, R.h):
        t = (r - top) / max(1, R.h - top)
        for c in range(R.w):
            v = 0.30 + 0.45 * t
            if x.rng.random() < 0.35 + 0.45 * t:
                R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)


def gen_road(x: Ctx) -> None:
    R = x.R
    top = int(R.horizon * x.f("top", 1.0))
    cx = R.w * x.f("cx_frac", 0.5)
    for r in range(top, R.h):
        t = (r - top) / max(1, R.h - top)
        half = max(1, int(1 + t * R.w * 0.22))
        for c in range(int(cx - half), int(cx + half) + 1):
            edge = abs(c - cx) > half - 1.5
            if edge:
                R.put(r, c, "│", 0.8, x.layer, x.elem)
            # The road surface stays mostly open: a solid pale triangle reads
            # as a spill of shading, not as tarmac receding to a vanishing point.
            elif x.rng.random() < 0.10 + 0.22 * t:
                R.put(r, c, ink(RAMP_BLOCK, 0.16 + 0.12 * t), 0.2, x.layer, x.elem)
        if int(t * 14) % 2 == 0:
            for c in range(int(cx - 1), int(cx + 2)):
                R.put(r, c, "▀", 0.85, x.layer, x.elem)


def gen_city_street(x: Ctx) -> None:
    R = x.R
    top = int(R.horizon * x.f("top", 1.0))
    for r in range(top, R.h):
        t = (r - top) / max(1, R.h - top)
        for c in range(R.w):
            if x.rng.random() > 0.25 + 0.5 * t:
                continue
            v = 0.12 + 0.30 * t
            R.put(r, c, ink(RAMP_BLOCK, v), v, x.layer, x.elem)
        if r % 2 == 0:
            for c in range(x.rng.randrange(0, 5), R.w, 6):
                R.put(r, c, "·", 0.5 + 0.3 * t, x.layer, x.elem)


# ==========================================================================
#  FOREGROUND
# ==========================================================================
def gen_grass_tufts(x: Ctx) -> None:
    R = x.R
    for r in range(R.h - x.i("rows", 3), R.h):
        d = (r - (R.h - x.i("rows", 3))) / max(1, x.i("rows", 3))
        for c in range(R.w):
            if x.rng.random() < 0.3 + 0.5 * d:
                R.put(r, c, x.rng.choice("|/\\,'"), 0.75 + 0.25 * d, x.layer, x.elem)


def gen_rocks(x: Ctx) -> None:
    R = x.R
    for _ in range(x.i("count", 7)):
        cc = x.rng.randrange(0, R.w)
        rr = R.h - x.rng.randrange(1, 4)
        w = x.rng.randrange(2, 6)
        for i in range(w):
            R.put(rr, cc + i, "▄", 0.8, x.layer, x.elem)
            R.put(rr + 1, cc + i, "█", 0.6, x.layer, x.elem)


def gen_reeds(x: Ctx) -> None:
    R = x.R
    for _ in range(x.i("count", 22)):
        cc = x.rng.randrange(0, R.w)
        h = x.rng.randrange(2, max(3, R.h // 7))
        for i in range(h):
            R.put(R.h - 1 - i, cc + (1 if i > h * 0.7 else 0),
                  "|" if i < h - 1 else "*", 0.85, x.layer, x.elem)


def gen_branch_frame(x: Ctx) -> None:
    """A foreground branch across the top corner — classic depth cue."""
    R = x.R
    r = 1
    for c in range(0, int(R.w * 0.55)):
        rr = r + int(math.sin(c * 0.05) * 1.6) + c // 26
        R.put(rr, c, "═" if c % 7 else "╤", 0.9, x.layer, x.elem)
        if x.rng.random() < 0.35:
            for k in range(1, x.rng.randrange(2, 4)):
                R.put(rr + k, c, x.rng.choice("v,'\""), 0.7, x.layer, x.elem)


def gen_fence(x: Ctx) -> None:
    R = x.R
    r = R.h - x.i("rows", 4)
    for c in range(0, R.w, 5):
        for i in range(x.i("rows", 4)):
            R.put(r + i, c, "║", 0.85, x.layer, x.elem)
    for c in range(R.w):
        R.put(r + 1, c, "═", 0.7, x.layer, x.elem, over_space_only=True)


# ==========================================================================
#  OVERLAY
# ==========================================================================
def gen_rain(x: Ctx) -> None:
    """Rain falls in sheets, not in confetti.

    Scattering individual strokes at random reads as noise over the whole
    picture. Real rain in ASCII reads when it is drawn as broken diagonal
    *lines* running the full height on a regular column period — the eye
    resolves the direction, and the gaps keep the scene underneath legible.
    """
    R = x.R
    slant = x.i("slant", 1)
    period = max(3, int(x.f("period", 7)))
    keep = x.f("keep", 0.48)
    for c0 in range(-R.h * abs(slant) - 1, R.w, period):
        jitter = x.rng.randrange(0, 2)
        for r in range(R.h):
            c = c0 + r * slant + jitter
            if 0 <= c < R.w and x.rng.random() < keep:
                R.put(r, c, "/" if slant > 0 else "\\", 0.85, x.layer, x.elem)


def gen_snowfall(x: Ctx) -> None:
    R = x.R
    for _ in range(int(R.h * R.w * x.f("density", 0.045))):
        R.put(x.rng.randrange(0, R.h), x.rng.randrange(0, R.w),
              x.rng.choice("*·.❄"), 0.95, x.layer, x.elem)


def gen_fog(x: Ctx) -> None:
    R = x.R
    n = value_noise(R.h, R.w, 13.0, x.rng, 2)
    band = x.f("band", 0.62)
    for r in range(R.h):
        prox = 1.0 - abs(r / max(1, R.h) - band) * x.f("tight", 2.4)
        if prox <= 0:
            continue
        for c in range(R.w):
            if n[r, c] * prox > 0.42:
                R.put(r, c, "░" if x.rng.random() < 0.7 else "▒", 0.4,
                      x.layer, x.elem)


def gen_birds(x: Ctx) -> None:
    R = x.R
    for _ in range(x.i("count", 9)):
        r = x.rng.randrange(1, max(2, R.horizon - 2))
        c = x.rng.randrange(2, max(3, R.w - 3))
        R.put(r, c, "v", 0.95, x.layer, x.elem)
        R.put(r - 1, c - 1, "\\", 0.9, x.layer, x.elem)
        R.put(r - 1, c + 1, "/", 0.9, x.layer, x.elem)


def gen_sparks(x: Ctx) -> None:
    R = x.R
    for _ in range(x.i("count", 40)):
        R.put(x.rng.randrange(0, R.h), x.rng.randrange(0, R.w),
              x.rng.choice("·.*"), 0.9, x.layer, x.elem, over_space_only=True)


def gen_scanlines(x: Ctx) -> None:
    R = x.R
    for r in range(0, R.h, x.i("every", 3)):
        for c in range(R.w):
            if R.ch[r, c] == " " and x.rng.random() < 0.55:
                R.put(r, c, "·", 0.18, x.layer, x.elem)


# ==========================================================================
#  Registry
# ==========================================================================
GENERATORS: Dict[str, Callable[[Ctx], None]] = {
    # sky
    "stars": gen_stars, "moon": gen_moon, "sun": gen_sun, "clouds": gen_clouds,
    "aurora": gen_aurora, "planet": gen_planet, "storm_sky": gen_storm_sky,
    # far
    "mountains": gen_mountains, "hills": gen_hills, "dunes": gen_dunes,
    "skyline": gen_skyline, "forest_line": gen_forest_line,
    "pyramids": gen_pyramids, "wall": gen_wall, "void": gen_void,
    # mid
    "big_tree": gen_big_tree, "castle": gen_castle, "tower": gen_tower,
    "ship": gen_ship, "rocket": gen_rocket, "bridge": gen_bridge,
    "fire": gen_fire, "waterfall": gen_waterfall, "portal": gen_portal,
    "circuit": gen_circuit, "figure": gen_figure, "silhouette": gen_silhouette,
    "monument": gen_monument,
    # near
    "ocean": gen_ocean, "reflection": gen_reflection, "grass_field": gen_grass_field,
    "desert_floor": gen_desert_floor, "snow_field": gen_snow_field,
    "road": gen_road, "city_street": gen_city_street,
    # fg
    "grass_tufts": gen_grass_tufts, "rocks": gen_rocks, "reeds": gen_reeds,
    "branch_frame": gen_branch_frame, "fence": gen_fence,
    # overlay
    "rain": gen_rain, "snowfall": gen_snowfall, "fog": gen_fog,
    "birds": gen_birds, "sparks": gen_sparks, "scanlines": gen_scanlines,
}


def render_plan(plan: Sequence[Dict], h: int, w: int, rng: random.Random,
                mood: Dict, horizon: float = 0.62) -> Raster:
    """Execute a retrieved scene plan back-to-front."""
    R = Raster(h, w)
    R.horizon = max(3, min(h - 2, int(h * horizon)))
    order = {n: i for i, n in enumerate(LAYERS)}
    for idx, item in enumerate(sorted(plan, key=lambda e: order.get(e["layer"], 9))):
        fn = GENERATORS.get(item["gen"])
        if fn is None:
            continue
        ctx = Ctx(R, rng, idx, item["layer"], item.get("params", {}), mood)
        try:
            fn(ctx)
        except Exception:
            # One bad generator must never take the whole picture down.
            continue
    return R
