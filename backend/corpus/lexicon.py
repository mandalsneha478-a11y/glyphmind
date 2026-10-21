"""
corpus/lexicon.py — semantic word clusters + calligram exemplars.

Two retrievable corpora for the Concrete Poetry engine:

  1. CLUSTERS   thematically coherent pools of nouns, verbs, modifiers and
                ready-made fragments.  The generator retrieves two or three
                clusters for a theme and composes inside that vocabulary,
                which is what keeps an emotion-driven poem lexically coherent
                instead of drifting into generic verse.

  2. EXEMPLARS  structural notes on real calligrams and concrete poems.  These
                do not supply words — they supply *strategies*: how Apollinaire
                let the rain fall in slanted columns, how Herbert narrowed the
                line to mime diminishment.  The retrieved strategy parameterises
                the text-flow solver.
"""
from __future__ import annotations

import re
from typing import Dict, List

CLUSTERS: List[Dict] = [
    {
        "id": "lex.grief", "name": "Grief and Loss",
        "tags": "grief loss mourning sorrow death absence funeral bereavement "
                "sadness ache empty missing gone goodbye farewell",
        "nouns": ["ash", "hollow", "vessel", "winter", "silence", "threshold", "stone",
                  "echo", "shadow", "hour", "coat", "chair", "letter", "doorway", "dust"],
        "verbs": ["keeps", "holds", "forgets", "waits", "empties", "settles", "lingers",
                  "folds", "closes", "remains", "cools", "fades"],
        "mods": ["quiet", "unopened", "cold", "narrow", "grey", "still", "unanswered",
                 "small", "patient", "hollow", "late"],
        "fragments": ["the chair still faces the window", "I keep your coat",
                      "nothing in the house has moved", "the hour will not close",
                      "winter does the remembering now", "an unopened letter",
                      "the house keeps its own weather", "grief is a room with the light left on"],
    },
    {
        "id": "lex.love", "name": "Love and Longing",
        "tags": "love longing desire romance intimacy tenderness yearning devotion "
                "affection heart passion closeness want",
        "nouns": ["pulse", "wrist", "breath", "orchard", "ember", "harbour", "palm",
                  "honey", "thread", "window", "morning", "name", "mouth", "hand"],
        "verbs": ["answers", "opens", "returns", "gathers", "burns", "leans", "reaches",
                  "stays", "finds", "warms", "calls", "chooses"],
        "mods": ["bright", "near", "patient", "sudden", "warm", "certain", "golden",
                 "unhurried", "wordless", "steady"],
        "fragments": ["your name in my mouth like weather", "the orchard says yes",
                      "I keep choosing the same door", "come nearer, the fire is low",
                      "we are the harbour and the leaving", "a hand that knows the way back"],
    },
    {
        "id": "lex.time", "name": "Time and Mortality",
        "tags": "time mortality age ageing clock hourglass deadline ending finitude "
                "sand years decay transience brief patience",
        "nouns": ["sand", "clock", "year", "thread", "harvest", "rust", "hinge", "dusk",
                  "almanac", "mirror", "calendar", "bone", "river", "candle"],
        "verbs": ["narrows", "runs", "thins", "counts", "turns", "spends", "slips",
                  "measures", "empties", "closes", "carries"],
        "mods": ["slow", "narrow", "late", "borrowed", "thin", "ordinary", "brief",
                 "unhurried", "final", "patient"],
        "fragments": ["the glass narrows at the waist", "every hour spends itself",
                      "we are borrowed from the river", "the hinge remembers the door",
                      "time is a thing that thins", "count what is left, then stop counting"],
    },
    {
        "id": "lex.nature", "name": "Growth and Nature",
        "tags": "nature growth tree forest garden season spring roots green leaves "
                "earth soil bloom organic wild living",
        "nouns": ["root", "canopy", "seed", "bark", "orchard", "moss", "rain", "ring",
                  "branch", "soil", "sap", "meadow", "lichen", "grain"],
        "verbs": ["reaches", "widens", "drinks", "climbs", "returns", "splits", "greens",
                  "holds", "thickens", "unfolds", "branches"],
        "mods": ["patient", "green", "wide", "deep", "slow", "stubborn", "rooted",
                 "unhurried", "old", "quiet"],
        "fragments": ["the root does the arguing", "every ring is a year of weather",
                      "green is a kind of insistence", "the canopy keeps its own council",
                      "what reaches down also reaches up"],
    },
    {
        "id": "lex.sea", "name": "Sea and Voyage",
        "tags": "sea ocean voyage sailing water tide harbour distance crossing journey "
                "salt shore wave maritime navigation exile",
        "nouns": ["tide", "keel", "salt", "harbour", "horizon", "gull", "rope", "shore",
                  "compass", "fathom", "swell", "lantern", "chart", "anchor"],
        "verbs": ["carries", "returns", "pulls", "crosses", "breaks", "drifts", "turns",
                  "rises", "leaves", "answers", "swells"],
        "mods": ["grey", "wide", "far", "unlit", "restless", "open", "salt", "deep",
                 "uncharted", "cold"],
        "fragments": ["the tide keeps no accounts", "every harbour is a kind of promise",
                      "salt learns the shape of everything", "we crossed and were not the same",
                      "the horizon is not a place"],
    },
    {
        "id": "lex.fire", "name": "Fire and Anger",
        "tags": "fire anger rage fury burning heat destruction passion violence "
                "intensity wrath flame temper consuming",
        "nouns": ["ember", "forge", "tinder", "ash", "spark", "furnace", "smoke", "coal",
                  "kindling", "blaze", "iron", "heat"],
        "verbs": ["takes", "climbs", "eats", "cracks", "roars", "spreads", "consumes",
                  "flares", "burns", "refuses"],
        "mods": ["bright", "loud", "quick", "hungry", "white", "unapologetic", "raw",
                 "sudden", "fierce"],
        "fragments": ["the forge does not apologise", "anger is a clean fuel",
                      "it took the house and asked nothing", "every ember is an argument",
                      "burn, then build"],
    },
    {
        "id": "lex.freedom", "name": "Freedom and Flight",
        "tags": "freedom flight escape wings bird liberty release leaving independence "
                "soaring open sky departure unbound",
        "nouns": ["wing", "thermal", "sky", "feather", "migration", "window", "hinge",
                  "latitude", "kite", "air", "distance", "gate"],
        "verbs": ["lifts", "leaves", "opens", "rises", "slips", "carries", "refuses",
                  "widens", "climbs", "goes"],
        "mods": ["open", "high", "untethered", "clear", "far", "light", "wide",
                 "unasked", "weightless"],
        "fragments": ["the window was never locked", "a bird is mostly intention",
                      "we are what the air allows", "leaving is a kind of arithmetic",
                      "the sky does not need permission"],
    },
    {
        "id": "lex.night", "name": "Night and Dream",
        "tags": "night dream sleep moon stars darkness insomnia quiet midnight "
                "shadow subconscious lunar secret",
        "nouns": ["moon", "lamp", "street", "pillow", "constellation", "hour", "fog",
                  "owl", "curtain", "frost", "silence", "dream"],
        "verbs": ["drifts", "turns", "watches", "settles", "forgets", "wakes", "keeps",
                  "dissolves", "circles", "waits"],
        "mods": ["blue", "soft", "late", "unlit", "quiet", "strange", "half", "wide",
                 "sleepless", "silver"],
        "fragments": ["the moon audits everything", "sleep arrives like a small country",
                      "the street is a corridor of lamps", "night does the editing",
                      "nothing is decided before dawn"],
    },
    {
        "id": "lex.city", "name": "City and Alienation",
        "tags": "city urban alienation crowd loneliness commute concrete traffic "
                "modern isolation apartment noise stranger office",
        "nouns": ["platform", "lobby", "window", "traffic", "elevator", "pavement",
                  "stranger", "receipt", "wire", "tower", "queue", "signal"],
        "verbs": ["passes", "waits", "counts", "repeats", "ignores", "hums", "moves",
                  "closes", "arrives", "forgets"],
        "mods": ["grey", "fluorescent", "identical", "hurried", "glass", "anonymous",
                 "vertical", "cold", "numbered"],
        "fragments": ["eight million separate weathers", "the lobby remembers no one",
                      "we agreed to be strangers", "the city keeps its receipts",
                      "a window is a kind of alibi"],
    },
    {
        "id": "lex.hope", "name": "Hope and Dawn",
        "tags": "hope dawn morning beginning renewal optimism light recovery future "
                "possibility start fresh healing",
        "nouns": ["dawn", "hinge", "seed", "doorway", "signal", "bread", "match",
                  "thaw", "pulse", "first light", "path", "window"],
        "verbs": ["begins", "opens", "returns", "warms", "insists", "thaws", "arrives",
                  "answers", "lifts", "greens"],
        "mods": ["early", "small", "stubborn", "pale", "new", "ordinary", "quiet",
                 "possible", "rising"],
        "fragments": ["hope is small and extremely stubborn", "the thaw starts underneath",
                      "one match is a whole argument", "morning does not ask how the night went",
                      "something is already beginning"],
    },
    {
        "id": "lex.memory", "name": "Memory and Childhood",
        "tags": "memory childhood nostalgia past remembering home family photograph "
                "old house summer recollection",
        "nouns": ["photograph", "stair", "hedge", "kitchen", "summer", "bicycle",
                  "banister", "jar", "wallpaper", "field", "radio", "name"],
        "verbs": ["keeps", "returns", "fades", "repeats", "holds", "softens", "recalls",
                  "stays", "curls", "opens"],
        "mods": ["yellowed", "warm", "distant", "summer", "half", "familiar", "worn",
                 "small", "particular"],
        "fragments": ["the stair knew every one of us", "summer was a longer unit then",
                      "memory keeps the wrong details", "the kitchen is still that colour",
                      "we were small and entirely certain"],
    },
    {
        "id": "lex.change", "name": "Transformation",
        "tags": "transformation change metamorphosis becoming rebirth evolution shift "
                "chrysalis growth identity new self",
        "nouns": ["chrysalis", "hinge", "moult", "threshold", "tide", "draft", "skin",
                  "wing", "edge", "version", "seam", "rehearsal"],
        "verbs": ["becomes", "splits", "sheds", "turns", "rewrites", "emerges", "learns",
                  "breaks", "folds", "begins"],
        "mods": ["new", "unfinished", "raw", "provisional", "second", "strange",
                 "different", "wet", "unpractised"],
        "fragments": ["the chrysalis is mostly waiting", "you will not recognise the hands",
                      "every draft believes it is final", "change arrives already wearing your face",
                      "what sheds is not what is lost"],
    },
]

EXEMPLARS: List[Dict] = [
    {"id": "cal.apollinaire.ilpleut", "name": "Il Pleut — Apollinaire (1918)",
     "tags": "rain falling vertical columns slanted descent melancholy weather calligramme "
             "grief sorrow drizzle lines of rain",
     "strategy": "slanted-columns", "shape_hint": "shape.rain",
     "note": "Five slanted columns of text falling like rain; line breaks obey the fall, "
             "not the syntax, so reading enacts the downpour."},
    {"id": "cal.apollinaire.coeur", "name": "Cœur Couronne et Miroir — Apollinaire",
     "tags": "heart crown mirror love emblem devotion circular reading figure calligramme",
     "strategy": "closed-outline", "shape_hint": "shape.heart",
     "note": "Text traces the outline of the figure rather than filling it; the reader "
             "circles the emblem instead of scanning it."},
    {"id": "cal.herbert.wings", "name": "Easter Wings — George Herbert (1633)",
     "tags": "wings flight diminishment recovery faith symmetry narrowing widening hope",
     "strategy": "taper-and-recover", "shape_hint": "shape.bird",
     "note": "Line length shrinks to a single syllable at the point of greatest "
             "spiritual poverty, then expands again — form performs the argument."},
    {"id": "cal.herbert.altar", "name": "The Altar — George Herbert",
     "tags": "altar stone sacrifice devotion architecture solid base column religion",
     "strategy": "architectural-block", "shape_hint": "shape.cross",
     "note": "A broad base, a narrow column, a broad capital. Stability encoded in width."},
    {"id": "cal.mallarme.coup", "name": "Un Coup de Dés — Mallarmé (1897)",
     "tags": "chance dice scattering constellation whitespace silence typography void "
             "shipwreck fragmentation",
     "strategy": "scattered-field", "shape_hint": "shape.star",
     "note": "Whitespace is load-bearing. Fragments are positioned across the spread so "
             "that silence has measurable duration."},
    {"id": "cal.gomringer.silencio", "name": "Silencio — Gomringer (1953)",
     "tags": "silence repetition absence minimal grid void centre emptiness concrete",
     "strategy": "grid-with-hole", "shape_hint": "shape.ring",
     "note": "One word repeated in a rectangle with the centre left blank: the only real "
             "silence in the poem is the hole."},
    {"id": "cal.carroll.tale", "name": "The Mouse's Tale — Carroll (1865)",
     "tags": "tail mouse winding narrow serpentine diminishing whimsy narrative spiral",
     "strategy": "serpentine-thread", "shape_hint": "shape.spiral",
     "note": "A single narrowing thread of text that winds down the page, the font "
             "shrinking as the tale does."},
    {"id": "cal.finlay.wave", "name": "Wave / Rock — Ian Hamilton Finlay",
     "tags": "sea wave rock erosion repetition interference pattern landscape minimal",
     "strategy": "interference-bands", "shape_hint": "shape.wave",
     "note": "Two words interleaved until one erodes the other; meaning emerges from the "
             "collision pattern, not the lexicon."},
]

STRATEGIES: Dict[str, Dict] = {
    "fill":                {"label": "Solid Fill", "density": 1.00, "justify": "center"},
    "slanted-columns":     {"label": "Slanted Columns", "density": 0.70, "justify": "stagger"},
    "closed-outline":      {"label": "Outline Trace", "density": 0.55, "justify": "edge"},
    "taper-and-recover":   {"label": "Taper & Recover", "density": 0.95, "justify": "center"},
    "architectural-block": {"label": "Architectural Block", "density": 1.00, "justify": "center"},
    "scattered-field":     {"label": "Scattered Field", "density": 0.45, "justify": "scatter"},
    "grid-with-hole":      {"label": "Grid with Void", "density": 0.90, "justify": "left"},
    "serpentine-thread":   {"label": "Serpentine Thread", "density": 0.60, "justify": "center"},
    "interference-bands":  {"label": "Interference Bands", "density": 0.80, "justify": "center"},
}


# --------------------------------------------------------------------------
#  Syllable estimation — drives the width-fitting solver.
# --------------------------------------------------------------------------
_VOWELS = "aeiouy"


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    n = len(re.findall(r"[aeiouy]{1,2}", w))
    return max(1, n)


def all_lexicon() -> List[Dict]:
    out: List[Dict] = []
    for c in CLUSTERS:
        out.append({**c, "kind": "lexicon"})
    for e in EXEMPLARS:
        out.append({**e, "kind": "calligram"})
    return out


def cluster_by_id(cid: str) -> Dict:
    for c in CLUSTERS:
        if c["id"] == cid:
            return c
    return CLUSTERS[0]
