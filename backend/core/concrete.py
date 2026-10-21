"""
concrete.py — Interactive Generative Concrete Poetry.

The problem
-----------
Making a poem *look like* a tree is not a styling task.  It is a constrained
packing problem over a non-convex region, where the atoms (words) have
indivisible widths, the reading order must stay linear, and the result has to
survive being read aloud.  A language model asked to "write a poem shaped like
a tree" produces a ragged approximation, because it is choosing characters, not
solving a layout.

The pipeline
------------
  1. RETRIEVE SHAPE      the theme is embedded and matched against structural
                         formulas (implicit functions over normalised space).
  2. RETRIEVE STRATEGY   a calligram exemplar — Apollinaire's falling rain,
                         Herbert's tapering wings, Mallarmé's scattered field —
                         supplies flow parameters, not words.
  3. RETRIEVE LEXICON    two or three thematically coherent word clusters form
                         the vocabulary the poem is allowed to draw on.
  4. RASTERISE           the formula is sampled into a boolean mask at the exact
                         requested character resolution, with the terminal cell
                         aspect ratio (~0.5) corrected so circles look round.
  5. SCANLINE SOLVE      each mask row is decomposed into runs.  Runs are filled
                         greedily with whole words under a syllable-weighted
                         cost that prefers heavy words where the figure is
                         thick and light words where it narrows — so the poem's
                         prosody tracks the silhouette.
  6. EMIT               ASCII grid, an SVG with per-row monospace tspans over
                         the shape outline, and a full provenance trace.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assets import get_asset_index

try:
    from ..corpus.lexicon import STRATEGIES, cluster_by_id, syllables
    from ..corpus.shapes import shape_by_id
except ImportError:                                      # pragma: no cover
    from corpus.lexicon import STRATEGIES, cluster_by_id, syllables
    from corpus.shapes import shape_by_id


CELL_ASPECT = 0.52          # a monospace cell is about twice as tall as wide

#: Short connectives used to fill runs too narrow for any content word.
#: Concrete poetry lives or dies on the silhouette, so a one- or two-cell limb
#: must still carry ink — and a function word carries it without lying.
FUNCTION_WORDS = ["a", "I", "of", "to", "in", "is", "it", "on", "as", "at",
                  "we", "so", "no", "up", "by", "or", "and", "the", "but",
                  "yet", "all", "its", "for", "not", "was", "one"]


# ==========================================================================
#  1. Rasterisation
# ==========================================================================
def rasterise(shape: Dict, height: int, width: int,
              supersample: int = 2) -> np.ndarray:
    """Sample an implicit shape function into a character-cell mask.

    Supersampling matters here: at one sample per cell a thin limb of the
    figure (a branch, a wing) alternates on and off and the silhouette breaks.
    """
    fn = shape["fn"]
    mask = np.zeros((height, width), dtype=bool)
    s = max(1, supersample)
    for r in range(height):
        for c in range(width):
            hits = 0
            for sr in range(s):
                for sc in range(s):
                    y = ((r + (sr + 0.5) / s) / height) * 2.0 - 1.0
                    x = ((c + (sc + 0.5) / s) / width) * 2.0 - 1.0
                    # correct for the cell aspect so a circle reads as a circle
                    x *= 1.0 / max(0.2, (width * CELL_ASPECT) / height) \
                        if (width * CELL_ASPECT) / height > 1 else 1.0
                    try:
                        if fn(x, y):
                            hits += 1
                    except Exception:
                        pass
            if hits >= (s * s) / 2.0:
                mask[r, c] = True
    return mask


def mask_runs(mask: np.ndarray, row: int, min_len: int = 2
              ) -> List[Tuple[int, int]]:
    """Contiguous inside-spans of one mask row, as (start, end_exclusive)."""
    runs: List[Tuple[int, int]] = []
    c = 0
    w = mask.shape[1]
    while c < w:
        if mask[row, c]:
            a = c
            while c < w and mask[row, c]:
                c += 1
            if c - a >= min_len:
                runs.append((a, c))
        else:
            c += 1
    return runs


# ==========================================================================
#  2. Text supply
# ==========================================================================
def tokenise_poem(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z'’\-]+|[.,;:!?—–]", text)
    return [t for t in toks if t]


def compose_lines(clusters: Sequence[Dict], theme: str, rng: random.Random,
                  n_tokens: int) -> List[str]:
    """Generate a vocabulary stream from the retrieved clusters.

    This is deliberately *not* free-form generation.  It is recombination
    inside a retrieved lexical field, which is what keeps a 400-word calligram
    thematically coherent instead of drifting.
    """
    nouns, verbs, mods, frags = [], [], [], []
    for cl in clusters:
        nouns += cl.get("nouns", [])
        verbs += cl.get("verbs", [])
        mods += cl.get("mods", [])
        frags += cl.get("fragments", [])
    if not nouns:
        nouns = ["thing"]; verbs = ["is"]; mods = ["quiet"]

    theme_words = [w for w in re.findall(r"[A-Za-z']{3,}", theme.lower())][:6]
    stream: List[str] = []
    patterns = [
        lambda: [rng.choice(mods), rng.choice(nouns)],
        lambda: ["the", rng.choice(nouns), rng.choice(verbs)],
        lambda: [rng.choice(nouns), rng.choice(verbs), "the", rng.choice(nouns)],
        lambda: ["and", rng.choice(mods), rng.choice(nouns)],
        lambda: [rng.choice(rng.choice(frags).split())] if frags else [rng.choice(nouns)],
        lambda: rng.choice(frags).split() if frags else [rng.choice(nouns)],
        lambda: [rng.choice(theme_words)] if theme_words else [rng.choice(nouns)],
    ]
    weights = [3, 3, 2, 2, 1, 4, 2]
    while len(stream) < n_tokens:
        p = rng.choices(patterns, weights=weights, k=1)[0]
        try:
            stream.extend(w for w in p() if w)
        except Exception:
            stream.append(rng.choice(nouns))
    return stream[:n_tokens]


# ==========================================================================
#  3. The scanline packing solver
# ==========================================================================
def fill_run(words: List[str], cursor: int, span: int, *,
             thickness: float, rng: random.Random,
             justify: str) -> Tuple[str, int, Dict]:
    """Fill one horizontal run with whole words.

    `thickness` ∈ [0,1] is how wide this run is relative to the widest run in
    the figure.  Where the figure is broad we prefer polysyllabic words; where
    it narrows to a few cells we prefer monosyllables, so the poem's rhythm
    thins as the silhouette thins.  That is the Herbert manoeuvre, automated.
    """
    chosen: List[str] = []
    used = 0
    attempts = 0
    stats = {"syllables": 0, "skipped": 0}
    n = len(words)
    while cursor < n and attempts < 64:
        best_i, best_cost = None, 1e9
        # Look ahead a few words and take the one whose weight best matches
        # the local thickness, provided it still fits.
        for look in range(0, min(5, n - cursor)):
            w = words[cursor + look]
            need = len(w) + (1 if chosen else 0)
            if used + need > span:
                continue
            syl = syllables(w)
            want = 1.0 + 2.0 * thickness
            cost = abs(syl - want) + 0.35 * look
            if cost < best_cost:
                best_cost, best_i = cost, cursor + look
        if best_i is None:
            break
        w = words[best_i]
        if best_i != cursor:
            words[cursor], words[best_i] = words[best_i], words[cursor]
            stats["skipped"] += 1
        used += len(w) + (1 if chosen else 0)
        chosen.append(w)
        stats["syllables"] += syllables(w)
        cursor += 1
        attempts += 1
        if used >= span - 1:
            break

    if not chosen:
        # The run is narrower than every nearby word.  Search further ahead for
        # anything that fits, then fall back to a function word, and only chop
        # a word (with a hyphen, honestly signalled) as the last resort — a
        # silhouette with a hole in it is worse than a hyphenated line.
        for look in range(0, min(40, n - cursor)):
            w = words[cursor + look]
            if len(w) <= span:
                words[cursor], words[cursor + look] = words[cursor + look], words[cursor]
                chosen = [words[cursor]]
                used = len(chosen[0])
                stats["syllables"] += syllables(chosen[0])
                cursor += 1
                stats["skipped"] += look
                break
    if not chosen:
        filler = [f for f in FUNCTION_WORDS if len(f) <= span]
        if filler:
            chosen = [rng.choice(filler)]
            used = len(chosen[0])
        elif cursor < n and span >= 3:
            w = words[cursor]
            chosen = [w[: span - 1] + "-"]
            used = span
            words[cursor] = w[span - 1:]
        else:
            return " " * span, cursor, stats

    text = " ".join(chosen)
    pad = span - len(text)
    if pad > 0:
        if justify == "left":
            text = text + " " * pad
        elif justify == "edge":
            text = text + " " * pad
        elif justify == "stagger":
            left = (pad * 2) // 3
            text = " " * left + text + " " * (pad - left)
        else:
            left = pad // 2
            text = " " * left + text + " " * (pad - left)
    return text[:span], cursor, stats


def flow(mask: np.ndarray, words: List[str], strategy: Dict,
         rng: random.Random) -> Tuple[List[str], Dict]:
    h, w = mask.shape
    density = float(strategy.get("density", 1.0))
    justify = strategy.get("justify", "center")
    grid = [[" "] * w for _ in range(h)]
    all_runs: List[Tuple[int, int, int]] = []
    for r in range(h):
        for (a, b) in mask_runs(mask, r):
            all_runs.append((r, a, b))
    if not all_runs:
        return ["" for _ in range(h)], {"runs": 0, "words_used": 0}
    widest = max(b - a for _, a, b in all_runs) or 1

    cursor = 0
    total_syl = 0
    used_runs = 0
    for (r, a, b) in all_runs:
        span = b - a
        if justify == "scatter" and rng.random() > density:
            continue
        if density < 1.0 and justify != "scatter" and rng.random() > density + 0.25:
            continue
        thickness = span / widest
        text, cursor, st = fill_run(words, cursor, span, thickness=thickness,
                                    rng=rng, justify=justify)
        total_syl += st["syllables"]
        used_runs += 1
        for i, ch in enumerate(text):
            if a + i < w:
                grid[r][a + i] = ch
        if cursor >= len(words):
            # Loop the vocabulary rather than truncating the figure.
            cursor = 0
    lines = ["".join(row).rstrip() for row in grid]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    return lines, {"runs": used_runs, "total_runs": len(all_runs),
                   "widest_run": widest, "syllables": total_syl,
                   "words_available": len(words), "cursor_end": cursor}


# ==========================================================================
#  4. SVG emission
# ==========================================================================
def to_svg(lines: Sequence[str], shape: Dict, *, cell_w: float = 8.4,
           cell_h: float = 16.0, fg: str = "#e9f7ff",
           accent: str = "#49e0ff") -> str:
    h = len(lines)
    w = max((len(l) for l in lines), default=1)
    W, H = w * cell_w, h * cell_h
    sx, sy = W / 100.0, H / 100.0
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        safe = (line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        rows.append(
            f'<text x="0" y="{(i + 0.8) * cell_h:.1f}" xml:space="preserve">{safe}</text>')
    body = "\n    ".join(rows)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" aria-label="concrete poem in the shape of {shape.get('name','a figure')}">
  <defs>
    <linearGradient id="gm-ink" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{accent}"/>
      <stop offset="100%" stop-color="{fg}"/>
    </linearGradient>
  </defs>
  <g transform="scale({sx:.4f} {sy:.4f})">
    <path d="{shape.get('svg','')}" fill="none" stroke="{accent}" stroke-opacity="0.22" stroke-width="1.2"/>
  </g>
  <g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="{cell_h * 0.78:.1f}" fill="url(#gm-ink)" letter-spacing="0">
    {body}
  </g>
</svg>"""


# ==========================================================================
#  Public API
# ==========================================================================
def generate(theme: str, *, poem: Optional[str] = None, shape_id: Optional[str] = None,
             strategy_id: Optional[str] = None, width: int = 72, height: int = 34,
             seed: Optional[int] = None,
             progress: Optional[callable] = None) -> Dict:
    emit = progress or (lambda *_a, **_k: None)
    t_all = time.perf_counter()
    width = max(24, min(160, int(width)))
    height = max(12, min(70, int(height)))
    if seed is None:
        seed = int(hashlib.blake2s((theme + (poem or "")).encode(),
                                   digest_size=4).hexdigest(), 16)
    rng = random.Random(seed)
    idx = get_asset_index()

    # ---- RETRIEVE ------------------------------------------------------
    t0 = time.perf_counter()
    shape_hits = idx.search(theme, kind="shape", k=4)
    if shape_id and idx.get(shape_id):
        chosen_shape_asset = idx.get(shape_id)
        shape_score = 1.0
    elif shape_hits:
        chosen_shape_asset, shape_score = shape_hits[0]
    else:
        chosen_shape_asset, shape_score = idx.get("shape.circle"), 0.0
    shape = shape_by_id(chosen_shape_asset["id"])

    cal_hits = idx.search(theme, kind="calligram", k=3)
    exemplar = cal_hits[0][0] if cal_hits else None
    strat_key = strategy_id or (exemplar or {}).get("strategy") or "fill"
    strategy = STRATEGIES.get(strat_key, STRATEGIES["fill"])

    lex_hits = idx.search(theme, kind="lexicon", k=3)
    clusters = [cluster_by_id(c["id"]) for c, _ in lex_hits] or [cluster_by_id("lex.hope")]
    t_retr = (time.perf_counter() - t0) * 1000

    retrieval = {
        "shape": {"id": shape["id"], "name": shape["name"],
                  "score": round(float(shape_score), 4),
                  "candidates": [{"id": s["id"], "name": s["name"],
                                  "score": round(sc, 4)} for s, sc in shape_hits]},
        "exemplar": ({"id": exemplar["id"], "name": exemplar["name"],
                      "strategy": exemplar["strategy"], "note": exemplar["note"],
                      "score": round(cal_hits[0][1], 4)} if exemplar else None),
        "exemplar_candidates": [{"id": c["id"], "name": c["name"],
                                 "score": round(s, 4)} for c, s in cal_hits],
        "strategy": {"key": strat_key, **strategy},
        "lexicon": [{"id": c["id"], "name": c["name"], "score": round(s, 4)}
                    for c, s in lex_hits],
        "corpus_size": idx.count(),
    }
    emit("retrieve", {**retrieval, "ms": round(t_retr, 2)})

    # ---- RASTERISE -----------------------------------------------------
    t0 = time.perf_counter()
    mask = rasterise(shape, height, width)
    inside = int(mask.sum())
    if inside < 20:                      # degenerate at this size: widen it
        mask = rasterise(shape_by_id("shape.circle"), height, width)
        inside = int(mask.sum())
    t_rast = (time.perf_counter() - t0) * 1000
    emit("rasterise", {"inside_cells": inside, "total_cells": height * width,
                       "coverage": round(inside / max(1, height * width), 4),
                       "ms": round(t_rast, 2)})

    # ---- TEXT ----------------------------------------------------------
    t0 = time.perf_counter()
    if poem and poem.strip():
        words = tokenise_poem(poem)
        source = "user-supplied poem"
        if len(words) < 12:
            words = words + compose_lines(clusters, theme, rng, 220)
            source = "user poem + retrieved lexicon"
    else:
        words = compose_lines(clusters, theme, rng, max(160, inside // 4))
        source = "retrieved lexical clusters"
    t_text = (time.perf_counter() - t0) * 1000
    emit("vocabulary", {"source": source, "tokens": len(words),
                        "clusters": [c["name"] for c in clusters],
                        "sample": " ".join(words[:14]), "ms": round(t_text, 2)})

    # ---- FLOW ----------------------------------------------------------
    t0 = time.perf_counter()
    lines, flow_stats = flow(mask, list(words), strategy, rng)
    t_flow = (time.perf_counter() - t0) * 1000
    emit("flow", {**flow_stats, "ms": round(t_flow, 2)})

    # ---- EMIT ----------------------------------------------------------
    t0 = time.perf_counter()
    svg = to_svg(lines, shape)
    ascii_art = "\n".join(lines)
    t_emit = (time.perf_counter() - t0) * 1000

    filled = sum(1 for l in lines for ch in l if ch != " ")
    fidelity = round(filled / max(1, inside), 4)
    emit("validate", {"glyphs_placed": filled, "mask_cells": inside,
                      "shape_fidelity": fidelity, "ms": round(t_emit, 2)})

    total = (time.perf_counter() - t_all) * 1000
    return {
        "theme": theme, "seed": seed,
        "ascii": ascii_art, "svg": svg,
        "lines": lines, "width": width, "height": height,
        "shape": {"id": shape["id"], "name": shape["name"], "svg": shape["svg"]},
        "retrieval": retrieval,
        "vocabulary": {"source": source, "tokens": len(words),
                       "clusters": [{"id": c["id"], "name": c["name"]} for c in clusters]},
        "flow": flow_stats,
        "validation": {"glyphs_placed": filled, "mask_cells": inside,
                       "shape_fidelity": fidelity},
        "timings_ms": {"retrieve": round(t_retr, 2), "rasterise": round(t_rast, 2),
                       "vocabulary": round(t_text, 2), "flow": round(t_flow, 2),
                       "emit": round(t_emit, 2), "total": round(total, 2)},
    }
