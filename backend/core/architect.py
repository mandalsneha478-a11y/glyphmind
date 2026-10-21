"""
architect.py — the Typographic ASCII / ANSI Architect.

What this subsystem is
----------------------
A **retrieval-augmented scene synthesiser**. The prompt is decomposed into
concept queries; each query is run against an indexed knowledge base of scene
recipes; the winning recipes are assembled into a depth-layered scene plan; and
procedural generators execute that plan at the requested resolution.

Where the RAG is
----------------
Everything that decides what the picture contains is retrieval:

  1. CONCEPT DECOMPOSITION  the prompt is split into spans and content words,
                            giving up to eight independent queries instead of
                            one blurry average of the whole sentence.
  2. SCENE RETRIEVAL        each query searches the scene knowledge base. Scores
                            are max-pooled per element and scaled by a prior.
  3. LAYER ASSIGNMENT       winners are slotted into sky / far / mid / near /
                            foreground / overlay under mutual-exclusion rules,
                            with mandatory layers back-filled so *every* prompt
                            produces a complete scene.
  4. STYLE RETRIEVAL        palette, frame, texture and the ranked typeface list
                            come from the same index.
  5. UNKNOWN-SUBJECT PATH   if no scene recipe matches the subject, retrieval
                            falls back to the shape-formula corpus and renders a
                            shaded silhouette; failing that, the subject word
                            itself is built as monumental typography.

Nothing is sampled from a model. Every element in the output is traceable to a
retrieved record with a score, and the UI prints that trace.

Why the geometry cannot shear
-----------------------------
Generators write into integer cells of a fixed-width raster, and the typography
solver measures every candidate glyph block against an integer slot rectangle
before anything is drawn. Candidates that would overrun are rejected, not
clipped. A final validation pass asserts rectangularity and counts out-of-slot
writes, so a broken composition is detectable rather than merely unfortunate.
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
from .canvas import BBox
from .sceneforge import LAYER_INDEX, LAYERS, Raster, render_plan

try:
    from ..corpus.scene_kb import (DAY_WORDS, HIGH_HORIZON, LOW_HORIZON,
                                   NIGHT_WORDS, element_by_id)
    from ..corpus.shapes import shape_by_id
except ImportError:                                      # pragma: no cover
    from corpus.scene_kb import (DAY_WORDS, HIGH_HORIZON, LOW_HORIZON,
                                 NIGHT_WORDS, element_by_id)
    from corpus.shapes import shape_by_id

try:
    import pyfiglet
    _HAS_FIGLET = True
except Exception:                                        # pragma: no cover
    _HAS_FIGLET = False


STOPWORDS = set("""a an the of and or for with to in on at from by as is are be
make create draw generate render build show me some please that this it into
using style styled looking like art ascii ansi picture image scene view of
very really quite just about over under above below near beside""".split())

# Acceptance thresholds, expressed as a FRACTION OF THE TOP POOLED SCORE.
#
# Absolute cosine thresholds do not survive a change of encoder: the hashed
# fallback and a transformer put their scores on completely different scales,
# and a cut-off tuned for one admits junk on the other. ("a neon city skyline"
# was pulling in a lighthouse, because 0.19 happened to clear a fixed 0.17.)
# A relative cut-off asks the only question that actually matters — is this
# element competitive with the best thing the prompt retrieved? — and is
# invariant to the scale the encoder happens to use.
REL_SUBJECT = 0.50
REL_CELESTIAL = 0.52
REL_OPTIONAL = 0.60
REL_WEATHER = 0.58
REL_SHAPE = 0.45
REL_SCALE_FIGURE = 0.70


# ==========================================================================
#  Fallback typeface when pyfiglet is unavailable
# ==========================================================================
_FALLBACK_FONT = {
    "A": ["█▀█", "█▀█", "▀ ▀"], "B": ["█▀▄", "█▀▄", "▀▀ "], "C": ["█▀▀", "█  ", "▀▀▀"],
    "D": ["█▀▄", "█ █", "▀▀ "], "E": ["█▀▀", "█▀▀", "▀▀▀"], "F": ["█▀▀", "█▀▀", "▀  "],
    "G": ["█▀▀", "█ █", "▀▀▀"], "H": ["█ █", "█▀█", "▀ ▀"], "I": ["█", "█", "▀"],
    "J": ["  █", "  █", "▀▀ "], "K": ["█ █", "█▀▄", "▀ ▀"], "L": ["█  ", "█  ", "▀▀▀"],
    "M": ["█▄▀▄█", "█ ▀ █", "▀   ▀"], "N": ["█▄ █", "█ ▀█", "▀  ▀"], "O": ["█▀█", "█ █", "▀▀▀"],
    "P": ["█▀█", "█▀▀", "▀  "], "Q": ["█▀█", "█ █", "▀▀▄"], "R": ["█▀█", "█▀▄", "▀ ▀"],
    "S": ["█▀▀", "▀▀█", "▀▀▀"], "T": ["▀█▀", " █ ", " ▀ "], "U": ["█ █", "█ █", "▀▀▀"],
    "V": ["█ █", "█ █", " ▀ "], "W": ["█   █", "█ ▄ █", "▀▀ ▀▀"], "X": ["▀▄▀", "▄▀▄", "▀ ▀"],
    "Y": ["█ █", "▀█▀", " ▀ "], "Z": ["▀▀█", "▄▀ ", "▀▀▀"], "0": ["█▀█", "█ █", "▀▀▀"],
    "1": ["▄█ ", " █ ", "▀▀▀"], "2": ["▀▀█", "█▀▀", "▀▀▀"], "3": ["▀▀█", " ▀█", "▀▀▀"],
    "4": ["█ █", "▀▀█", "  ▀"], "5": ["█▀▀", "▀▀█", "▀▀▀"], "6": ["█▀▀", "█▀█", "▀▀▀"],
    "7": ["▀▀█", "  █", "  ▀"], "8": ["█▀█", "█▀█", "▀▀▀"], "9": ["█▀█", "▀▀█", "▀▀▀"],
    " ": ["  ", "  ", "  "], "-": ["   ", "▀▀▀", "   "], ".": ["  ", "  ", "▀ "],
    "!": ["█", "█", "▀"], "?": ["▀▀█", " ▀ ", " ▀ "], ":": [" ", "▀", "▀"],
}


def _fallback_render(text: str) -> List[str]:
    rows = ["", "", ""]
    for ch in text.upper():
        glyph = _FALLBACK_FONT.get(ch, _FALLBACK_FONT[" "])
        for i in range(3):
            rows[i] += glyph[i] + " "
    w = max(len(r) for r in rows)
    return [r.ljust(w) for r in rows]


def render_figlet(text: str, font: str) -> Optional[List[str]]:
    if not text.strip():
        return None
    if _HAS_FIGLET:
        try:
            raw = pyfiglet.figlet_format(text, font=font, width=500)
            lines = [l.rstrip() for l in raw.split("\n")]
            while lines and not lines[0].strip():
                lines.pop(0)
            while lines and not lines[-1].strip():
                lines.pop()
            if not lines:
                return None
            w = max(len(l) for l in lines)
            return [l.ljust(w) for l in lines]
        except Exception:
            return None
    return _fallback_render(text)


# ==========================================================================
#  Prompt analysis
# ==========================================================================
SPLIT_RE = re.compile(r",|\band\b|\bwith\b|\bover\b|\bunder\b|\bbeneath\b|"
                      r"\bin\b|\bon\b|\bat\b|\bnear\b|\bduring\b|\bbeside\b", re.I)


def decompose(prompt: str) -> List[str]:
    """Split a prompt into independent retrieval queries.

    One embedding of a whole sentence is an average, and averages lose the
    minority concept. "a lighthouse in a storm over the sea" averaged is mostly
    'weather'; decomposed it is three precise queries that each retrieve their
    own element. This is the single biggest accuracy win in the subsystem.
    """
    base = prompt.strip()
    spans = [s.strip(" .;:") for s in SPLIT_RE.split(base) if s.strip(" .;:")]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", base)
             if w.lower() not in STOPWORDS]
    queries: List[str] = [base]
    for s in spans:
        if len(s) > 2 and s.lower() != base.lower():
            queries.append(s)
    for w in words:
        if w.lower() not in [q.lower() for q in queries]:
            queries.append(w)
    seen, out = set(), []
    for q in queries:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:9]


def detect_mood(prompt: str, chosen: Sequence[Dict]) -> Dict:
    low = " " + prompt.lower() + " "
    night = sum(1 for w in NIGHT_WORDS if w in low)
    day = sum(1 for w in DAY_WORDS if w in low)
    for e in chosen:
        if (e.get("mood") or {}).get("night"):
            night += 1
    return {"night": night > day, "night_score": night, "day_score": day}


def detect_horizon(prompt: str, default: float = 0.62) -> float:
    low = " " + prompt.lower() + " "
    hi = sum(1 for w in HIGH_HORIZON if w in low)
    lo = sum(1 for w in LOW_HORIZON if w in low)
    h = default + 0.07 * min(2, hi) - 0.06 * min(2, lo)
    return max(0.34, min(0.80, h))


def derive_title(prompt: str, explicit: Optional[str] = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()[:26]
    quoted = re.findall(r"[\"'“”‘’]([^\"'“”‘’]{2,26})[\"'“”‘’]", prompt)
    if quoted:
        return quoted[0].strip()
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", prompt)
             if w.lower() not in STOPWORDS]
    if not words:
        return "GLYPHMIND"
    pick, total = [], 0
    for w in words:
        if total + len(w) + 1 > 18 and pick:
            break
        pick.append(w); total += len(w) + 1
    return " ".join(pick).upper()[:22]


# ==========================================================================
#  Scene planning — this is the retrieval core of the subsystem
# ==========================================================================
def plan_scene(prompt: str, *, forced: Optional[Dict] = None,
               trace: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
    idx = get_asset_index()
    queries = decompose(prompt)

    # ---- multi-query max-pooled retrieval ------------------------------
    pooled: Dict[str, Dict] = {}
    per_query: List[Dict] = []
    for q in queries:
        hits = idx.search(q, kind="scene", k=8, mmr=0.18)
        per_query.append({"query": q,
                          "hits": [{"id": a["id"], "name": a["name"],
                                    "layer": a["layer"], "score": round(s, 4)}
                                   for a, s in hits[:4]]})
        for a, s in hits:
            s = float(s) * float(a.get("weight", 1.0))
            cur = pooled.get(a["id"])
            if cur is None or s > cur["score"]:
                pooled[a["id"]] = {"el": a, "score": s, "via": q}

    ranked = sorted(pooled.values(), key=lambda r: -r["score"])
    top_score = ranked[0]["score"] if ranked else 1.0
    bar = {k: round(top_score * v, 4) for k, v in
           (("subject", REL_SUBJECT), ("celestial", REL_CELESTIAL),
            ("optional", REL_OPTIONAL), ("weather", REL_WEATHER),
            ("shape", REL_SHAPE), ("scale", REL_SCALE_FIGURE))}

    def best(layer: str, slot: Optional[str] = None,
             exclude_slots: Sequence[str] = ()) -> Optional[Dict]:
        for r in ranked:
            e = r["el"]
            if e["layer"] != layer:
                continue
            if slot is not None and e.get("slot") != slot:
                continue
            if e.get("slot") in exclude_slots:
                continue
            return r
        return None

    plan: List[Dict] = []
    picks: List[Dict] = []
    used_slots: set = set()

    def add(rec: Optional[Dict], role: str, *, forced_id: Optional[str] = None,
            reason: str = "retrieved") -> Optional[Dict]:
        if forced_id:
            e = idx.get(forced_id) or element_by_id(forced_id)
            rec = {"el": e, "score": 1.0, "via": "manual override"}
            reason = "manual override"
        if rec is None:
            return None
        e = rec["el"]
        slot = e.get("slot")
        if slot and slot in used_slots:
            return None
        if slot:
            used_slots.add(slot)
        item = {"id": e["id"], "name": e["name"], "layer": e["layer"],
                "gen": e["gen"], "params": dict(e.get("params") or {}),
                "mood": e.get("mood") or {}}
        plan.append(item)
        picks.append({"role": role, "id": e["id"], "name": e["name"],
                      "layer": e["layer"], "gen": e["gen"],
                      "score": round(float(rec["score"]), 4),
                      "matched_query": rec["via"], "reason": reason})
        return item

    forced = forced or {}

    # ---- SKY: an atmosphere, plus at most one celestial body -----------
    atmos = best("sky", exclude_slots=("celestial",))
    if forced.get("sky_id"):
        add(None, "atmosphere", forced_id=forced["sky_id"])
    elif atmos and atmos["score"] >= bar["optional"] * 0.6:
        add(atmos, "atmosphere")
    else:
        default_sky = "scene.stars" if any(w in prompt.lower() for w in NIGHT_WORDS) \
            else "scene.clouds"
        add({"el": idx.get(default_sky) or element_by_id(default_sky),
             "score": 0.0, "via": "default"}, "atmosphere",
            reason="layer back-filled: no sky element cleared threshold")

    celest = best("sky", slot="celestial")
    if celest and celest["score"] >= bar["celestial"]:
        add(celest, "celestial body")

    # ---- FAR: the horizon --------------------------------------------
    hor = best("far", slot="horizon")
    horizon_pick = hor
    if forced.get("far_id"):
        add(None, "horizon", forced_id=forced["far_id"])
    elif hor is not None:
        add(hor, "horizon")
    else:
        horizon_pick = None
        add({"el": idx.get("scene.hills") or element_by_id("scene.hills"),
             "score": 0.0, "via": "default"}, "horizon",
            reason="layer back-filled: no horizon element cleared threshold")

    # A second far band may sit in front of the horizon — a treeline before
    # mountains, a wall before a skyline. One slot per depth plane would make
    # "snowy mountains with pine forest" choose between its own two nouns.
    second = None
    for r in ranked:
        e = r["el"]
        if e["layer"] != "far" or any(p["id"] == e["id"] for p in picks):
            continue
        if e["gen"] == (horizon_pick["el"]["gen"] if horizon_pick else None):
            continue
        if r["score"] >= bar["optional"]:
            second = r
        break
    if second is not None:
        e = second["el"]
        plan.append({"id": e["id"], "name": e["name"], "layer": "far",
                     "gen": e["gen"], "params": dict(e.get("params") or {}),
                     "mood": e.get("mood") or {}})
        picks.append({"role": "near horizon band", "id": e["id"], "name": e["name"],
                      "layer": "far", "gen": e["gen"],
                      "score": round(float(second["score"]), 4),
                      "matched_query": second["via"],
                      "reason": "second far band, drawn in front of the horizon"})

    # ---- MID: the subject ---------------------------------------------
    subj_fallback: Dict = {}
    subj = best("mid", slot="subject")
    if forced.get("subject_id"):
        add(None, "subject", forced_id=forced["subject_id"])
    elif subj and subj["score"] >= bar["subject"]:
        add(subj, "subject")
    elif horizon_pick is not None and horizon_pick["score"] >= bar["subject"]:
        # The horizon element IS the subject. "a neon city skyline at night"
        # retrieved a skyline at 0.78; planting a second focal object in front
        # of it (or, worse, a monument spelling out NEON) does not add content,
        # it occludes the one thing the prompt actually asked for.
        subj_fallback = {"mode": "skipped",
                         "reason": "horizon element is the subject",
                         "horizon": horizon_pick["el"]["id"]}
    else:
        # Nothing in the scene KB matches the noun. Fall back to the shape
        # formula corpus, then to monumental typography of the word itself.
        subject_words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", prompt)
                         if w.lower() not in STOPWORDS]
        sq = " ".join(subject_words[:4]) or prompt
        shots = idx.search(sq, kind="shape", k=3)
        if shots and float(shots[0][1]) >= bar["shape"]:
            sh = shape_by_id(shots[0][0]["id"])
            plan.append({"id": "fallback.silhouette", "name": f"Silhouette · {sh['name']}",
                         "layer": "mid", "gen": "silhouette",
                         "params": {"fn": sh["fn"], "r0": 0.14, "r1": 0.72, "cw": 0.46},
                         "mood": {}})
            picks.append({"role": "subject", "id": sh["id"],
                          "name": f"Silhouette · {sh['name']}", "layer": "mid",
                          "gen": "silhouette", "score": round(float(shots[0][1]), 4),
                          "matched_query": sq,
                          "reason": "no scene recipe matched — fell back to the "
                                    "shape-formula corpus"})
            subj_fallback = {"mode": "silhouette", "shape": sh["id"],
                             "score": round(float(shots[0][1]), 4)}
        else:
            word = (subject_words[0] if subject_words else "GLYPHMIND").upper()[:12]
            block = render_figlet(word, "small") or _fallback_render(word)
            plan.append({"id": "fallback.monument", "name": f"Monument · {word}",
                         "layer": "mid", "gen": "monument",
                         "params": {"block": block, "anchor": 0.94}, "mood": {}})
            picks.append({"role": "subject", "id": "fallback.monument",
                          "name": f"Monument · {word}", "layer": "mid",
                          "gen": "monument", "score": 0.0, "matched_query": word,
                          "reason": "no scene recipe and no shape matched — the "
                                    "subject word is built as architecture"})
            subj_fallback = {"mode": "monument", "word": word}

    scale_fig = best("mid", slot="scale")
    if scale_fig and scale_fig["score"] >= bar["scale"]:
        add(scale_fig, "human scale")

    # ---- NEAR: the ground plane ---------------------------------------
    # Deep space has no ground plane. Letting a ground element in anyway is how
    # you end up with ocean waves lapping at the bottom of an orbital scene.
    horizon_is_void = any(p["id"] == "scene.void" for p in picks)
    if horizon_is_void:
        picks.append({"role": "ground plane", "id": "—", "name": "none (vacuum)",
                      "layer": "near", "gen": "—", "score": 0.0,
                      "matched_query": "", "reason": "suppressed: the horizon "
                      "element is deep space, which has no ground plane"})
    elif forced.get("ground_id"):
        add(None, "ground plane", forced_id=forced["ground_id"])
    else:
        grnd = best("near", slot="ground")
        if grnd is not None:
            add(grnd, "ground plane")
        else:
            add({"el": idx.get("scene.grass_field") or element_by_id("scene.grass_field"),
                 "score": 0.0, "via": "default"}, "ground plane",
                reason="layer back-filled: no ground element cleared threshold")
    ground_id = next((p["id"] for p in picks if p["role"] == "ground plane"), "")
    refl = pooled.get("scene.reflection")
    wants_reflection = (not horizon_is_void) and (
        ground_id in ("scene.ocean",)
        or (refl is not None and refl["score"] >= bar["optional"]))
    if wants_reflection:
        add(refl or {"el": idx.get("scene.reflection") or element_by_id("scene.reflection"),
                     "score": 0.0, "via": "implied by water surface"},
            "reflection",
            reason="retrieved" if refl else "implied by the chosen water surface")

    # ---- FOREGROUND ----------------------------------------------------
    fg = best("fg")
    if fg and fg["score"] >= bar["optional"]:
        add(fg, "foreground")

    # ---- OVERLAY -------------------------------------------------------
    weather = best("overlay", slot="weather")
    if weather and weather["score"] >= bar["weather"]:
        add(weather, "weather")
    extra = 0
    for r in ranked:
        if extra >= 2:
            break
        e = r["el"]
        if e["layer"] != "overlay" or e.get("slot") == "weather":
            continue
        if any(p["id"] == e["id"] for p in picks):
            continue
        if r["score"] >= bar["optional"]:
            add(r, "atmosphere detail")
            extra += 1

    meta = {
        "queries": queries,
        "per_query": per_query,
        "candidates": len(pooled),
        "picks": picks,
        "subject_fallback": subj_fallback,
        "top_pool": [{"id": r["el"]["id"], "name": r["el"]["name"],
                      "layer": r["el"]["layer"], "score": round(r["score"], 4)}
                     for r in ranked[:12]],
        "top_score": round(top_score, 4),
        "thresholds": bar,
    }
    if trace is not None:
        trace.update(meta)
    return plan, meta


# ==========================================================================
#  Frame, title and caption
# ==========================================================================
def draw_frame(R: Raster, frame: Dict, pad: int = 0) -> None:
    h, w = R.h, R.w
    r0, c0, r1, c1 = pad, pad, h - 1 - pad, w - 1 - pad
    lay = LAYER_INDEX["fg"]
    R.hline(r0, c0 + 1, c1, frame["h"], 0.9, lay, 90)
    R.hline(r1, c0 + 1, c1, frame["h"], 0.9, lay, 90)
    R.put(r0, c0, frame["tl"], 0.95, lay, 90)
    R.put(r0, c1, frame["tr"], 0.95, lay, 90)
    R.put(r1, c0, frame["bl"], 0.95, lay, 90)
    R.put(r1, c1, frame["br"], 0.95, lay, 90)
    for r in range(r0 + 1, r1):
        R.put(r, c0, frame["v"], 0.9, lay, 90)
        R.put(r, c1, frame["v"], 0.9, lay, 90)


def _split_words(words: Sequence[str], n: int) -> List[List[str]]:
    if n <= 1:
        return [list(words)]
    total = sum(len(w) + 1 for w in words)
    target = total / n
    groups: List[List[str]] = [[]]
    acc = 0.0
    for w in words:
        if acc >= target * len(groups) and len(groups) < n:
            groups.append([])
        groups[-1].append(w)
        acc += len(w) + 1
    return [g for g in groups if g]


def _stack(blocks: Sequence[Sequence[str]], gap: int = 0) -> List[str]:
    rows: List[str] = []
    width = max((max((len(l) for l in b), default=0) for b in blocks), default=0)
    for i, b in enumerate(blocks):
        if i and gap:
            rows.extend([" " * width] * gap)
        for line in b:
            pad = width - len(line)
            left = pad // 2
            rows.append(" " * left + line + " " * (pad - left))
    return rows


def fit_title(title: str, fonts: Sequence[Dict], slot: BBox,
              trace: List[Dict]) -> Tuple[Optional[List[str]], Optional[Dict]]:
    """Choose a typeface by measurement, not by guesswork.

    Every (typeface x line-count) combination is rendered and measured against
    the integer slot rectangle. Overrunning candidates are rejected outright.
    Among those that fit, the winner maximises slot coverage, tempered by
    retrieval rank — first-fit would hand a poster to a three-row face in an
    eight-row slot.
    """
    chain = list(fonts)
    for fb in ("font.small", "font.mini"):
        if not any(f["id"] == fb for f in chain):
            a = get_asset_index().get(fb)
            if a:
                chain.append(a)

    words = title.split()
    slot_area = max(1, slot.width * slot.height)
    best = None
    for rank, f in enumerate(chain):
        for nlines in (1, 2, 3):
            if nlines > len(words):
                break
            groups = _split_words(words, nlines)
            blocks = [render_figlet(" ".join(g), f.get("figlet", "small"))
                      for g in groups]
            if any(b is None for b in blocks):
                trace.append({"font": f["id"], "lines": nlines, "status": "render-failed"})
                break
            block = _stack(blocks)
            bw = max(len(l) for l in block)
            bh = len(block)
            fits = bw <= slot.width and bh <= slot.height
            coverage = (bw * bh) / slot_area
            score = coverage + 0.3 * (1.0 - rank / max(1, len(chain)))
            trace.append({"font": f["id"], "name": f.get("name"), "lines": nlines,
                          "measured": {"w": bw, "h": bh},
                          "slot": {"w": slot.width, "h": slot.height},
                          "coverage": round(coverage, 3), "score": round(score, 3),
                          "status": "accepted" if fits else "rejected: overruns slot"})
            if fits and (best is None or score > best[0]):
                best = (score, block, {**f, "lines": nlines,
                                       "coverage": round(coverage, 3)})
    if best is not None:
        for t in trace:
            if t.get("font") == best[2]["id"] and t.get("lines") == best[2]["lines"]:
                t["status"] = "accepted — selected (best slot coverage)"
        return best[1], best[2]

    plain = [title[: max(1, slot.width)]]
    trace.append({"font": "plain-text", "measured": {"w": len(plain[0]), "h": 1},
                  "slot": {"w": slot.width, "h": slot.height},
                  "status": "accepted (degraded to plain text)"})
    return plain, {"id": "font.plain", "name": "Plain", "figlet": None}


def stamp_block(R: Raster, block: Sequence[str], r: int, c: int, lay: int,
                elem: int, v: float = 0.98) -> int:
    over = 0
    for i, line in enumerate(block):
        for j, g in enumerate(line):
            if g == " ":
                continue
            rr, cc = r + i, c + j
            if not (0 <= rr < R.h and 0 <= cc < R.w):
                over += 1
                continue
            R.put(rr, cc, g, v, lay, elem)
    return over


def clear_band(R: Raster, r0: int, r1: int, c0: int, c1: int) -> None:
    for r in range(max(0, r0), min(R.h, r1)):
        for c in range(max(0, c0), min(R.w, c1)):
            R.ch[r, c] = " "
            R.v[r, c] = 0.0
            R.lay[r, c] = -1
            R.elem[r, c] = -1


# ==========================================================================
#  ANSI colourisation
# ==========================================================================
LAYER_TINT = {"sky": -1, "far": 0, "mid": 1, "near": 0, "fg": 1, "overlay": 2}


def colourise(R: Raster, palette: Dict) -> str:
    colors: List[int] = palette.get("colors", [250])
    n = len(colors)
    tints = [LAYER_TINT.get(name, 0) for name in LAYERS]
    out: List[str] = []
    prev: Optional[int] = None
    for r in range(R.h):
        line: List[str] = []
        for c in range(R.w):
            g = str(R.ch[r, c])
            if g == " ":
                line.append(" ")
                continue
            lay = int(R.lay[r, c])
            tint = tints[lay] if 0 <= lay < len(tints) else 0
            idx = int(round(float(R.v[r, c]) * (n - 1))) + tint
            idx = max(0, min(n - 1, idx))
            col = colors[idx]
            if col != prev:
                line.append(f"\x1b[38;5;{col}m")
                prev = col
            line.append(g)
        out.append("".join(line))
    return "\x1b[0m" + "\n".join(out) + "\x1b[0m"


# ==========================================================================
#  Public API
# ==========================================================================
def compose(prompt: str, *, width: int = 110, height: int = 40,
            title: Optional[str] = None, seed: Optional[int] = None,
            mode: Optional[str] = None,
            frame_id: Optional[str] = None, palette_id: Optional[str] = None,
            font_id: Optional[str] = None,
            subject_id: Optional[str] = None, sky_id: Optional[str] = None,
            far_id: Optional[str] = None, ground_id: Optional[str] = None,
            horizon: Optional[float] = None,
            # accepted for backwards compatibility with the earlier API
            layout_id: Optional[str] = None, motif_id: Optional[str] = None,
            texture_id: Optional[str] = None,
            progress: Optional[callable] = None) -> Dict:
    emit = progress or (lambda *_a, **_k: None)
    t_all = time.perf_counter()
    width = max(48, min(240, int(width)))
    height = max(16, min(96, int(height)))
    if seed is None:
        seed = int(hashlib.blake2s(prompt.encode(), digest_size=4).hexdigest(), 16)
    rng = random.Random(seed)
    idx = get_asset_index()

    low = prompt.lower()
    if mode not in ("scene", "poster"):
        mode = "poster" if (title or any(w in low for w in
                                         ("poster", "banner", "headline", "title",
                                          "logo", "cover", "flyer"))) else "scene"

    # ---- 1. SCENE RETRIEVAL + PLANNING --------------------------------
    t0 = time.perf_counter()
    forced = {k: v for k, v in (("subject_id", subject_id), ("sky_id", sky_id),
                                ("far_id", far_id), ("ground_id", ground_id))
              if v}
    plan, plan_meta = plan_scene(prompt, forced=forced)
    mood = detect_mood(prompt, plan)
    hz = float(horizon) if horizon else detect_horizon(prompt)
    t_plan = (time.perf_counter() - t0) * 1000
    emit("scene", {**{k: v for k, v in plan_meta.items() if k != "per_query"},
                   "per_query": plan_meta["per_query"],
                   "mood": mood, "horizon": round(hz, 3),
                   "layers": [p["layer"] for p in plan],
                   "ms": round(t_plan, 2)})

    # ---- 2. STYLE RETRIEVAL -------------------------------------------
    t0 = time.perf_counter()

    # Query expansion: the style retrieval is conditioned on what the scene
    # retrieval already returned. Asking the palette index about the raw prompt
    # alone gave near-ties it had no way to break — "a lighthouse in a storm"
    # picked an ember palette and rendered a night seascape in fire red. Adding
    # the names of the chosen elements and the detected time of day gives the
    # second retrieval the context the first one established.
    style_query = " ".join([
        prompt,
        " ".join(p["name"] for p in plan_meta["picks"]),
        "night dark moonlit" if mood["night"] else "day bright daylight",
    ])

    def pick(kind: str, forced_id: Optional[str], fallback: str):
        cands = idx.search(style_query, kind=kind, k=4)
        listing = [{"id": c["id"], "name": c.get("name"), "score": round(s, 4)}
                   for c, s in cands]
        if forced_id and idx.get(forced_id):
            return idx.get(forced_id), 1.0, listing
        if cands:
            return cands[0][0], cands[0][1], listing
        a = idx.get(fallback) or {}
        return a, 0.0, listing

    palette, s_pal, cand_pal = pick("palette", palette_id,
                                    "pal.matrix" if mood["night"] else "pal.sunset")
    frame, s_frame, cand_frame = pick("frame", frame_id, "frame.unicode.light")
    font_hits = idx.search(style_query, kind="font", k=5)
    if font_id and idx.get(font_id):
        font_hits = [(idx.get(font_id), 1.0)] + [h for h in font_hits
                                                 if h[0]["id"] != font_id]
    t_style = (time.perf_counter() - t0) * 1000
    style_trace = {
        "palette": {"chosen": palette.get("id"), "score": round(s_pal, 4),
                    "candidates": cand_pal},
        "frame": {"chosen": frame.get("id"), "score": round(s_frame, 4),
                  "candidates": cand_frame},
        "fonts": [{"id": f["id"], "name": f["name"], "score": round(s, 4)}
                  for f, s in font_hits],
        "style_query": style_query[:240],
        "corpus_size": idx.count(), "corpus_kinds": idx.kinds(),
    }
    emit("style", {**style_trace, "ms": round(t_style, 2)})

    # ---- 3. RENDER THE SCENE -------------------------------------------
    t0 = time.perf_counter()
    R = render_plan(plan, height, width, rng, mood, horizon=hz)
    t_render = (time.perf_counter() - t0) * 1000
    emit("render", {"elements": len(plan), "ink_cells": R.ink_count(),
                    "ink_ratio": round(R.ink_count() / max(1, height * width), 4),
                    "horizon_row": R.horizon, "ms": round(t_render, 2)})

    # ---- 4. TYPOGRAPHY --------------------------------------------------
    t0 = time.perf_counter()
    the_title = derive_title(prompt, title)
    font_trace: List[Dict] = []
    title_block: Optional[List[str]] = None
    chosen_font: Optional[Dict] = None
    overflow = 0
    title_box: Optional[BBox] = None

    if mode == "poster":
        band_h = max(5, int(height * 0.30))
        slot = BBox(2, 3, 2 + band_h, width - 3)
        title_block, chosen_font = fit_title(the_title, [f for f, _ in font_hits],
                                             slot, font_trace)
        if title_block:
            clear_band(R, 1, 2 + band_h + 1, 1, width - 1)
            bh = len(title_block); bw = max(len(l) for l in title_block)
            r = slot.r0 + (slot.height - bh) // 2
            c = slot.c0 + (slot.width - bw) // 2
            overflow += stamp_block(R, title_block, r, c, LAYER_INDEX["fg"], 91)
            title_box = BBox(r, c, r + bh, c + bw)
            rule = max(0, min(height - 2, slot.r1))
            R.hline(rule, 4, width - 4, frame["h"], 0.65, LAYER_INDEX["fg"], 92)
    else:
        # Scene mode keeps the art full-bleed and sets the title on a plate.
        # The plate is cleared first: typography laid straight over texture is
        # unreadable, and unreadable is the same as absent.
        plate = f"  {the_title}  "
        c0 = max(2, (width - len(plate)) // 2)
        r0 = 1
        clear_band(R, r0 - 1, r0 + 3, c0 - 1, c0 + len(plate) + 1)
        for i, g in enumerate(plate):
            R.put(r0, c0 + i, g, 0.99, LAYER_INDEX["fg"], 91)
        R.hline(r0 + 1, c0 + 1, c0 + len(plate) - 1, frame["h"], 0.6,
                LAYER_INDEX["fg"], 92)
        title_box = BBox(r0, c0, r0 + 1, c0 + len(plate))
        chosen_font = {"id": "plate", "name": "Scene plate", "lines": 1}
    t_type = (time.perf_counter() - t0) * 1000
    emit("typography", {"mode": mode, "title": the_title, "attempts": font_trace,
                        "chosen": (chosen_font or {}).get("id"),
                        "ms": round(t_type, 2)})

    # ---- 5. FRAME + CAPTION ---------------------------------------------
    t0 = time.perf_counter()
    draw_frame(R, frame, pad=0)
    picks = plan_meta["picks"]
    caption = " · ".join(p["name"].lower() for p in picks[:4])
    caption = f"  {caption}  "[: width - 6]
    cr = height - 2
    cc = max(2, (width - len(caption)) // 2)
    clear_band(R, cr, cr + 1, cc - 1, cc + len(caption) + 1)
    for i, g in enumerate(caption):
        R.put(cr, cc + i, g, 0.9, LAYER_INDEX["fg"], 93)

    ansi = colourise(R, palette)
    plain = R.text()
    t_fin = (time.perf_counter() - t0) * 1000

    # ---- 6. VALIDATE -----------------------------------------------------
    rows = plain.split("\n")
    widths = {len(r) for r in rows}
    ink = sum(1 for r in rows for ch in r if ch != " ")
    validation = {
        "rectangular": len(widths) == 1,
        "row_count": len(rows),
        "row_width": max(widths) if widths else 0,
        "out_of_slot_writes": overflow,
        "ink_cells": ink,
        "ink_ratio": round(ink / max(1, height * width), 4),
        "alignment_score": round(
            1.0 - min(1.0, overflow / max(1.0, height * width * 0.02)), 4),
        "title_fits": bool(title_block) if mode == "poster" else True,
        "scene_elements": len(plan),
        "layers_covered": sorted({p["layer"] for p in plan}),
    }
    emit("validate", {**validation, "ms": round(t_fin, 2)})

    total = (time.perf_counter() - t_all) * 1000
    return {
        "prompt": prompt, "title": the_title, "seed": seed, "mode": mode,
        "plain": plain, "ansi": ansi, "width": width, "height": height,
        "scene": {k: v for k, v in plan_meta.items()},
        "mood": mood, "horizon": round(hz, 3),
        "assets": {"palette": palette, "frame": frame, "font": chosen_font},
        "retrieval": {**style_trace, "scene": plan_meta},
        "typography": {"title": the_title, "attempts": font_trace,
                       "chosen": chosen_font, "mode": mode,
                       "box": title_box.to_dict() if title_box else None},
        "validation": validation,
        "timings_ms": {"scene_retrieval": round(t_plan, 2),
                       "style_retrieval": round(t_style, 2),
                       "render": round(t_render, 2),
                       "typography": round(t_type, 2),
                       "finish": round(t_fin, 2), "total": round(total, 2)},
    }
