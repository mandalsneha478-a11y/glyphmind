"""
corpus/components.py — the retrieval corpus for the ASCII / ANSI Architect.

Every entry carries, besides its art, the thing that actually matters for
generation: **exact geometry**.  A pure LLM asked for ASCII art drifts by a
column and the whole picture shears.  Here the generator never guesses a width:
it retrieves a component whose `w` and `h` are known integers, and the
compositor solves placement as a packing problem over those integers.

Entry schema
------------
    id          stable identifier
    kind        frame | banner | texture | motif | palette | layout | font
    name        human name
    tags        retrieval surface (free text, embedded)
    ...         kind-specific payload
"""
from __future__ import annotations

from typing import Dict, List

# ==========================================================================
#  FRAMES — border glyph sets.  Geometry is implied by the glyph tuple, so a
#  frame can be drawn at any size with a guaranteed-closed outline.
# ==========================================================================
FRAMES: List[Dict] = [
    {"id": "frame.ascii.classic", "name": "Classic ASCII",
     "tags": "plain ascii retro terminal simple portable safe teletype",
     "tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"},
    {"id": "frame.unicode.light", "name": "Unicode Light",
     "tags": "clean modern thin minimal crisp technical documentation",
     "tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│"},
    {"id": "frame.unicode.heavy", "name": "Unicode Heavy",
     "tags": "bold strong industrial heavy weight emphatic brutalist",
     "tl": "┏", "tr": "┓", "bl": "┗", "br": "┛", "h": "━", "v": "┃"},
    {"id": "frame.unicode.double", "name": "Double Rule",
     "tags": "formal ceremonial certificate classical bbs dos ansi retro",
     "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║"},
    {"id": "frame.unicode.round", "name": "Rounded",
     "tags": "soft friendly gentle calm rounded approachable organic",
     "tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"},
    {"id": "frame.block.solid", "name": "Solid Block",
     "tags": "massive pixel blocky chunky arcade videogame 8-bit brutal",
     "tl": "█", "tr": "█", "bl": "█", "br": "█", "h": "█", "v": "█"},
    {"id": "frame.block.shade", "name": "Shaded Block",
     "tags": "dithered gradient vaporwave demoscene fade atmospheric",
     "tl": "▓", "tr": "▓", "bl": "▓", "br": "▓", "h": "▒", "v": "▒"},
    {"id": "frame.stars", "name": "Star Rule",
     "tags": "celebratory magic night sky festive award decorative",
     "tl": "*", "tr": "*", "bl": "*", "br": "*", "h": "*", "v": "*"},
    {"id": "frame.dots", "name": "Dotted",
     "tags": "quiet understated draft sketch provisional whisper",
     "tl": ".", "tr": ".", "bl": "'", "br": "'", "h": ".", "v": ":"},
    {"id": "frame.wave", "name": "Wave Rule",
     "tags": "ocean water sea liquid flowing tide nautical",
     "tl": "~", "tr": "~", "bl": "~", "br": "~", "h": "~", "v": "}"},
]

# ==========================================================================
#  TEXTURES — character ramps for shading and fills, ordered light -> dark.
# ==========================================================================
TEXTURES: List[Dict] = [
    {"id": "tex.ramp.block", "name": "Block Ramp",
     "tags": "gradient shading smooth fade volume depth demoscene",
     "ramp": " ░▒▓█"},
    {"id": "tex.ramp.classic", "name": "Classic Ramp",
     "tags": "ascii gradient tonal photographic halftone grayscale",
     "ramp": " .:-=+*#%@"},
    {"id": "tex.ramp.dots", "name": "Dot Ramp",
     "tags": "stipple pointillist sparse airy dust particles",
     "ramp": " .·:∙•●"},
    {"id": "tex.ramp.binary", "name": "Binary Rain",
     "tags": "matrix code cyber digital hacker data stream glitch",
     "ramp": " .01█"},
    {"id": "tex.ramp.wave", "name": "Wave Ramp",
     "tags": "water ocean ripple liquid flow current sea",
     "ramp": " .~≈≋█"},
    {"id": "tex.ramp.star", "name": "Starfield",
     "tags": "space cosmos night sky stars astronomy void",
     "ramp": " .·*✦★"},
    {"id": "tex.ramp.organic", "name": "Organic",
     "tags": "forest foliage nature growth leaves moss garden",
     "ramp": " ,;'^&%@"},
]

# ==========================================================================
#  MOTIFS — fixed-geometry sprites.  `w`/`h` are authoritative.
# ==========================================================================
def _blk(s: str) -> List[str]:
    lines = [l for l in s.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    w = max((len(l) for l in lines), default=0)
    return [l.ljust(w) for l in lines]


_MOTIF_SRC = {
    "motif.mountain": ("Mountain Range",
                       "mountain peak alps range summit nature landscape climb height",
                       r"""
          /\
         /  \      /\
        /    \    /  \
       /      \  /    \   /\
      /        \/      \ /  \
     /                  v    \
    /__________________________\
"""),
    "motif.tree": ("Pine Tree",
                   "tree forest pine nature growth woods evergreen organic life",
                   r"""
        /\
       /  \
      /____\
      /    \
     /______\
     /      \
    /________\
       |  |
       |__|
"""),
    "motif.wave": ("Ocean Wave",
                   "wave ocean sea water surf tide maritime liquid motion",
                   r"""
      .-~~~-.
    .'       `.      .-~~-.
   /    ~~     \   .'      `.
  ;  ~~~~~~~~~  ; /   ~~~~   \
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""),
    "motif.skyline": ("City Skyline",
                      "city urban skyline buildings metropolis downtown towers night",
                      r"""
            ___
     ___   |   |        ___
    |   |__|   |   __  |   |___
    |   |  |   |__|  |_|   |   |
  __|   |  |   |  |  | |   |   |__
 |  |   |  |   |  |  | |   |   |  |
 |__|___|__|___|__|__|_|___|___|__|
"""),
    "motif.terminal": ("CRT Terminal",
                       "computer terminal crt monitor retro machine console hacker screen",
                       r"""
   .------------------------.
   |  > _                   |
   |                        |
   |  GLYPHMIND READY       |
   |                        |
   '------------------------'
       \__________________/
        /                \
"""),
    "motif.rocket": ("Rocket",
                     "rocket space launch ship flight sci-fi mission ascend",
                     r"""
        /\
       /  \
      |    |
      |    |
     /|    |\
    / |    | \
   '  |____|  '
      /\  /\
     /  \/  \
      \ || /
       \||/
        \/
"""),
    "motif.sun": ("Sun",
                  "sun light warmth day solar bright star dawn heat",
                  r"""
      \   |   /
       \  |  /
        .-'-.
   ---  |   |  ---
        '-.-'
       /  |  \
      /   |   \
"""),
    "motif.cloud": ("Cloud",
                    "cloud sky weather soft drift calm air atmosphere",
                    r"""
       .--.
    .-(    ).
   (___.__)__)
"""),
    "motif.heart": ("Heart",
                    "heart love emotion affection romance feeling warmth",
                    r"""
    **     **
  *    * *    *
 *      *      *
 *             *
  *           *
    *       *
      *   *
        *
"""),
    "motif.circuit": ("Circuit",
                      "circuit electronics technology chip board wire silicon logic",
                      r"""
  o---[===]---+---o
              |
       o--+--[R]--o
          |
       +--+--||--+
       |         |
       o---------o
"""),
    "motif.castle": ("Castle",
                     "castle fortress medieval keep tower stronghold fantasy",
                     r"""
   |^|   |^|   |^|
   |_|___|_|___|_|
   |  _   _   _  |
   | |_| |_| |_| |
   |   _______   |
   |  |  _ _  |  |
   |__|_|_|_|_|__|
"""),
    "motif.skull": ("Skull",
                    "skull bone death pirate warning danger dark gothic",
                    r"""
     .-----.
    /  _ _  \
   |  (o|o)  |
   |    ^    |
    \ \___/ /
     '-----'
      | | |
"""),
    "motif.bird": ("Bird",
                   "bird flight wings freedom sky soar swallow migration",
                   r"""
    \        /
     \      /
   ___\    /___
  '    \  /    '
        \/
"""),
    "motif.gear": ("Gear",
                   "gear machine mechanical engine industry cog factory system",
                   r"""
     _.-'-._
   .'  ___  '.
  /   /   \   \
 |   |     |   |
  \   \___/   /
   '._     _.'
      '---'
"""),
}

MOTIFS: List[Dict] = []
for _id, (_name, _tags, _src) in _MOTIF_SRC.items():
    _b = _blk(_src)
    MOTIFS.append({"id": _id, "name": _name, "tags": _tags, "art": _b,
                   "h": len(_b), "w": len(_b[0]) if _b else 0})


# ==========================================================================
#  PALETTES — xterm-256 colour sequences with mood tags.
# ==========================================================================
PALETTES: List[Dict] = [
    {"id": "pal.neon", "name": "Neon Grid",
     "tags": "neon cyberpunk synthwave electric vivid night club arcade",
     "colors": [201, 165, 129, 93, 57, 51, 45, 39], "bg": 233},
    {"id": "pal.sunset", "name": "Sunset",
     "tags": "sunset warm orange dusk evening golden calm romantic",
     "colors": [52, 88, 124, 166, 202, 208, 214, 220], "bg": 234},
    {"id": "pal.forest", "name": "Deep Forest",
     "tags": "forest green nature growth moss woods organic calm",
     "colors": [22, 28, 34, 40, 46, 82, 118, 154], "bg": 233},
    {"id": "pal.ice", "name": "Glacier",
     "tags": "ice cold blue winter arctic frost clean clinical",
     "colors": [17, 18, 19, 25, 31, 38, 45, 51], "bg": 233},
    {"id": "pal.amber", "name": "Amber CRT",
     "tags": "amber terminal retro vintage monochrome phosphor computer dos",
     "colors": [58, 94, 130, 166, 172, 178, 214, 220], "bg": 232},
    {"id": "pal.vapor", "name": "Vaporwave",
     "tags": "vaporwave pastel pink purple dreamy nostalgic aesthetic 80s",
     "colors": [219, 213, 207, 171, 141, 111, 117, 159], "bg": 53},
    {"id": "pal.mono", "name": "Monochrome",
     "tags": "monochrome grayscale minimal print newspaper classic restrained",
     "colors": [236, 240, 244, 248, 250, 252, 254, 255], "bg": 232},
    {"id": "pal.blood", "name": "Ember",
     "tags": "red fire blood danger heat intense aggressive volcanic",
     "colors": [52, 88, 124, 160, 196, 202, 208, 215], "bg": 232},
    {"id": "pal.matrix", "name": "Matrix",
     "tags": "matrix hacker green code digital terminal cyber data",
     "colors": [22, 28, 34, 40, 46, 47, 48, 85], "bg": 232},
]

# ==========================================================================
#  LAYOUTS — normalised slot rectangles.  These are the "structural bounding
#  boxes" the architecture retrieves so that typography never collides.
#  Coordinates are fractions of the canvas: (r0, c0, r1, c1).
# ==========================================================================
LAYOUTS: List[Dict] = [
    {"id": "layout.poster", "name": "Poster",
     "tags": "poster title hero announcement banner headline big statement",
     "slots": [
         {"role": "title",   "rect": [0.08, 0.05, 0.42, 0.95], "align": "center"},
         {"role": "motif",   "rect": [0.44, 0.15, 0.80, 0.85], "align": "center"},
         {"role": "caption", "rect": [0.84, 0.10, 0.94, 0.90], "align": "center"},
     ]},
    {"id": "layout.plaque", "name": "Centred Plaque",
     "tags": "plaque award certificate centred formal badge emblem ceremonial",
     "slots": [
         {"role": "motif",   "rect": [0.10, 0.30, 0.42, 0.70], "align": "center"},
         {"role": "title",   "rect": [0.44, 0.05, 0.74, 0.95], "align": "center"},
         {"role": "caption", "rect": [0.78, 0.15, 0.92, 0.85], "align": "center"},
     ]},
    {"id": "layout.split", "name": "Split Panel",
     "tags": "split two column side by side comparison duo panel diptych",
     "slots": [
         {"role": "title",   "rect": [0.05, 0.05, 0.28, 0.95], "align": "left"},
         {"role": "motif",   "rect": [0.32, 0.04, 0.92, 0.46], "align": "center"},
         {"role": "caption", "rect": [0.40, 0.54, 0.80, 0.96], "align": "left"},
     ]},
    {"id": "layout.scene", "name": "Landscape Scene",
     "tags": "scene landscape horizon environment vista panorama world setting",
     "slots": [
         {"role": "title",   "rect": [0.04, 0.05, 0.22, 0.95], "align": "center"},
         {"role": "motif",   "rect": [0.26, 0.02, 0.86, 0.98], "align": "bottom"},
         {"role": "caption", "rect": [0.88, 0.05, 0.97, 0.95], "align": "right"},
     ]},
    {"id": "layout.bbs", "name": "BBS Welcome Screen",
     "tags": "bbs dos ansi bulletin board 90s modem door game login retro",
     "slots": [
         {"role": "title",   "rect": [0.06, 0.03, 0.40, 0.97], "align": "center"},
         {"role": "caption", "rect": [0.44, 0.08, 0.56, 0.92], "align": "center"},
         {"role": "motif",   "rect": [0.60, 0.20, 0.94, 0.80], "align": "center"},
     ]},
    {"id": "layout.stack", "name": "Vertical Stack",
     "tags": "stack list vertical sequence steps flow ordered column",
     "slots": [
         {"role": "title",   "rect": [0.03, 0.05, 0.26, 0.95], "align": "center"},
         {"role": "caption", "rect": [0.30, 0.10, 0.44, 0.90], "align": "center"},
         {"role": "motif",   "rect": [0.48, 0.10, 0.97, 0.90], "align": "center"},
     ]},
]

# ==========================================================================
#  FONTS — figlet font names with measured aspect characteristics.
#  `cols_per_char` is the empirical width cost, used by the fitting solver.
# ==========================================================================
FONTS: List[Dict] = [
    {"id": "font.big", "figlet": "big", "name": "Big",
     "tags": "big bold loud headline impact poster shout maximal",
     "cols_per_char": 8.2, "rows": 8},
    {"id": "font.standard", "figlet": "standard", "name": "Standard",
     "tags": "standard default neutral classic balanced readable",
     "cols_per_char": 7.0, "rows": 6},
    {"id": "font.slant", "figlet": "slant", "name": "Slant",
     "tags": "slant italic motion speed dynamic leaning fast",
     "cols_per_char": 7.4, "rows": 6},
    {"id": "font.small", "figlet": "small", "name": "Small",
     "tags": "small compact tight narrow economical dense fits",
     "cols_per_char": 5.2, "rows": 5},
    {"id": "font.mini", "figlet": "mini", "name": "Mini",
     "tags": "mini tiny minimal smallest cramped micro",
     "cols_per_char": 4.1, "rows": 4},
    {"id": "font.banner", "figlet": "banner", "name": "Banner",
     "tags": "banner block solid heavy wide proclamation monumental",
     "cols_per_char": 8.0, "rows": 7},
    {"id": "font.block", "figlet": "block", "name": "Block",
     "tags": "block cube chunky pixel 3d arcade videogame",
     "cols_per_char": 9.0, "rows": 8},
    {"id": "font.shadow", "figlet": "shadow", "name": "Shadow",
     "tags": "shadow depth drop elegant engraved subtle dimensional",
     "cols_per_char": 6.4, "rows": 6},
    {"id": "font.digital", "figlet": "digital", "name": "Digital",
     "tags": "lcd seven segment clock readout gauge meter numeric display panel",
     "cols_per_char": 3.2, "rows": 3},
    {"id": "font.doom", "figlet": "doom", "name": "Doom",
     "tags": "doom heavy metal aggressive game brutal dark",
     "cols_per_char": 7.8, "rows": 8},
]


def all_components() -> List[Dict]:
    """Flatten every component into one retrievable list."""
    out: List[Dict] = []
    for c in FRAMES:
        out.append({**c, "kind": "frame"})
    for c in TEXTURES:
        out.append({**c, "kind": "texture"})
    for c in MOTIFS:
        out.append({**c, "kind": "motif"})
    for c in PALETTES:
        out.append({**c, "kind": "palette"})
    for c in LAYOUTS:
        out.append({**c, "kind": "layout"})
    for c in FONTS:
        out.append({**c, "kind": "font"})
    return out
