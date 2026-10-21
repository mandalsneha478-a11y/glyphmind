"""
corpus/scene_kb.py — the retrievable scene knowledge base.

This is the corpus that makes the Architect a RAG system rather than a template
engine. Each entry is not a picture: it is a **recipe** — a depth layer, a
procedural generator, and the parameters that generator should be called with —
indexed by a deliberately broad tag surface.

Retrieval does the work the prompt implies:

    "a lighthouse in a storm over the sea"
        → scene.storm_sky   (sky)      storm clouds + bolts
        → scene.cliffs      (far)      jagged fractal ridge
        → scene.lighthouse  (mid)      tapered tower, lantern, beam
        → scene.ocean       (near)     perspective-compressed swell
        → scene.rain        (overlay)  slanted strokes
        → scene.reflection  (near)     computed from what is already drawn

Nothing in that list was authored for lighthouses. Every element was retrieved
independently, assigned to a depth layer, and composited. Change one noun in
the prompt and a different recipe comes back.

`slot` enforces mutual exclusion where physics demands it: a scene gets one
celestial body, one ground plane, one subject.
"""
from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
#  layer:  sky | far | mid | near | fg | overlay
#  slot:   optional mutual-exclusion group
#  weight: prior nudge for ambiguous prompts
# ---------------------------------------------------------------------------
SCENE_ELEMENTS: List[Dict] = [

    # ====================== SKY =========================================
    {"id": "scene.stars", "name": "Starfield", "layer": "sky", "gen": "stars",
     "tags": "night stars starry sky space cosmos galaxy dark midnight astronomy "
             "constellation universe nebula deep space evening nocturnal",
     "params": {"density": 0.055}, "mood": {"night": True}, "weight": 1.0},

    {"id": "scene.moon", "name": "Moon", "layer": "sky", "gen": "moon",
     "slot": "celestial",
     "tags": "moon moonlight crescent lunar night midnight tide silver "
             "moonlit nocturne full moon sleep dream",
     "params": {"crescent": 0.55}, "mood": {"night": True}, "weight": 1.0},

    {"id": "scene.sun", "name": "Sun", "layer": "sky", "gen": "sun",
     "slot": "celestial",
     "tags": "sun sunrise sunset dawn dusk daylight day bright solar golden "
             "morning evening warm summer heat noon",
     "params": {"rays": 1}, "weight": 1.0},

    {"id": "scene.clouds", "name": "Cloud Bank", "layer": "sky", "gen": "clouds",
     "tags": "clouds cloudy sky overcast weather soft drifting air atmosphere "
             "cumulus haze calm daytime",
     "params": {"scale": 11.0, "threshold": 0.55}, "weight": 0.9},

    {"id": "scene.storm_sky", "name": "Storm Sky", "layer": "sky", "gen": "storm_sky",
     "tags": "storm thunder lightning tempest dark clouds rain gale violent "
             "thunderstorm dramatic ominous angry sky hurricane",
     "params": {"bolts": 2}, "mood": {"night": True}, "weight": 1.1},

    {"id": "scene.aurora", "name": "Aurora", "layer": "sky", "gen": "aurora",
     "tags": "aurora northern lights polar arctic borealis shimmer magic "
             "ethereal green curtain iceland norway winter sky",
     "params": {"bands": 3, "thickness": 4}, "mood": {"night": True}, "weight": 1.0},

    {"id": "scene.planet", "name": "Planet and Ring", "layer": "sky", "gen": "planet",
     "slot": "celestial",
     "tags": "planet saturn ring orbit space sci-fi alien world moon base "
             "interstellar cosmic astronomy outer space",
     "params": {"ring": 1}, "mood": {"night": True}, "weight": 1.0},

    # ====================== FAR =========================================
    {"id": "scene.mountains", "name": "Mountain Range", "layer": "far",
     "gen": "mountains", "slot": "horizon",
     "tags": "mountain mountains range peaks alps himalaya summit ridge alpine "
             "highlands crag snowcap valley climbing wilderness",
     "params": {"ranges": 3, "roughness": 0.55, "snow": 0.6}, "weight": 1.0},

    {"id": "scene.cliffs", "name": "Jagged Cliffs", "layer": "far",
     "gen": "mountains", "slot": "horizon",
     "tags": "cliff cliffs coast rugged rocky headland crag shore bluff "
             "precipice sea wall storm coastline",
     "params": {"ranges": 2, "roughness": 0.75, "snow": 0.0}, "weight": 0.95},

    {"id": "scene.hills", "name": "Rolling Hills", "layer": "far", "gen": "hills",
     "slot": "horizon",
     "tags": "hills countryside rolling meadow pastoral farmland gentle green "
             "downs valley rural landscape calm",
     "params": {"ranges": 2}, "weight": 0.8},

    {"id": "scene.dunes", "name": "Sand Dunes", "layer": "far", "gen": "dunes",
     "slot": "horizon",
     "tags": "desert dune dunes sand sahara arid wasteland drought heat "
             "camel oasis dry barren",
     "params": {"ranges": 3}, "weight": 1.0},

    {"id": "scene.skyline", "name": "City Skyline", "layer": "far", "gen": "skyline",
     "slot": "horizon",
     "tags": "city skyline urban buildings metropolis downtown towers manhattan "
             "cyberpunk neon night city skyscraper tokyo blade runner",
     "params": {"min_w": 4, "max_w": 11, "lit": 0.45}, "weight": 1.15},

    {"id": "scene.forest_line", "name": "Tree Line", "layer": "far",
     "gen": "forest_line", "slot": "horizon",
     "tags": "forest woods trees pine conifer woodland taiga jungle grove "
             "nature wilderness evergreen canopy",
     "params": {"min_h": 4, "max_h": 10}, "weight": 1.0},

    {"id": "scene.pyramids", "name": "Pyramids", "layer": "far", "gen": "pyramids",
     "slot": "horizon",
     "tags": "pyramid pyramids egypt giza ancient desert tomb pharaoh "
             "archaeology monument civilisation",
     "params": {"count": 2}, "weight": 1.0},

    {"id": "scene.wall", "name": "Fortified Wall", "layer": "far", "gen": "wall",
     "slot": "horizon",
     "tags": "wall rampart fortification battlement siege stone barrier "
             "medieval defence keep border",
     "params": {}, "weight": 0.9},

    {"id": "scene.void", "name": "Deep Space", "layer": "far", "gen": "void",
     "slot": "horizon",
     "tags": "space void vacuum emptiness cosmos interstellar zero gravity "
             "orbit starship nebula dust nothing",
     "params": {}, "weight": 0.85},

    # ====================== MID (subjects) ==============================
    {"id": "scene.big_tree", "name": "Great Tree", "layer": "mid", "gen": "big_tree",
     "slot": "subject",
     "tags": "tree oak great tree branches canopy roots single tree ancient "
             "growth nature solitary life seasons",
     "params": {}, "weight": 1.0},

    {"id": "scene.castle", "name": "Castle", "layer": "mid", "gen": "castle",
     "slot": "subject",
     "tags": "castle fortress keep citadel stronghold medieval fantasy palace "
             "kingdom tower gate knight dragon",
     "params": {}, "weight": 1.05},

    {"id": "scene.lighthouse", "name": "Lighthouse", "layer": "mid", "gen": "tower",
     "slot": "subject",
     "tags": "lighthouse beacon tower light coast harbour guidance signal "
             "warning cape maritime lamp spire",
     "params": {"beam": 1}, "weight": 1.05},

    {"id": "scene.ship", "name": "Sailing Ship", "layer": "mid", "gen": "ship",
     "slot": "subject",
     "tags": "ship boat sail sailing vessel schooner voyage sea ocean "
             "pirate harbour crossing maritime navy yacht",
     "params": {}, "weight": 1.05},

    {"id": "scene.rocket", "name": "Rocket", "layer": "mid", "gen": "rocket",
     "slot": "subject",
     "tags": "rocket launch spacecraft missile apollo nasa liftoff mission "
             "space program booster countdown astronaut",
     "params": {"flame": 7}, "weight": 1.1},

    {"id": "scene.bridge", "name": "Suspension Bridge", "layer": "mid", "gen": "bridge",
     "slot": "subject",
     "tags": "bridge suspension crossing span golden gate river engineering "
             "cable viaduct connection infrastructure",
     "params": {}, "weight": 1.0},

    {"id": "scene.fire", "name": "Fire", "layer": "mid", "gen": "fire",
     "slot": "subject",
     "tags": "fire flame campfire bonfire burning blaze heat warmth hearth "
             "inferno ember torch destruction",
     "params": {"spread": 6}, "weight": 1.0},

    {"id": "scene.waterfall", "name": "Waterfall", "layer": "mid", "gen": "waterfall",
     "slot": "subject",
     "tags": "waterfall cascade falls river gorge torrent rapids plunge "
             "mist water cataract jungle",
     "params": {}, "weight": 1.0},

    {"id": "scene.portal", "name": "Portal", "layer": "mid", "gen": "portal",
     "slot": "subject",
     "tags": "portal gateway wormhole vortex magic doorway dimension rift "
             "threshold arcane sci-fi singularity",
     "params": {}, "weight": 0.95},

    {"id": "scene.circuit", "name": "Circuit Lattice", "layer": "mid", "gen": "circuit",
     "slot": "subject",
     "tags": "circuit board electronics chip silicon technology computer "
             "motherboard wires logic hardware machine data network",
     "params": {}, "weight": 1.0},

    {"id": "scene.figure", "name": "Lone Figure", "layer": "mid", "gen": "figure",
     "slot": "scale",
     "tags": "person figure human silhouette lone traveller walker standing "
             "someone man woman watcher solitude scale",
     "params": {}, "weight": 0.7},

    # ====================== NEAR (ground planes) ========================
    {"id": "scene.ocean", "name": "Open Water", "layer": "near", "gen": "ocean",
     "slot": "ground",
     "tags": "sea ocean water waves lake surf tide maritime bay swell "
             "shore coast sailing deep blue harbour",
     "params": {}, "weight": 1.1},

    {"id": "scene.grass_field", "name": "Grassland", "layer": "near",
     "gen": "grass_field", "slot": "ground",
     "tags": "grass field meadow plain prairie lawn green pasture "
             "countryside summer picnic earth soil",
     "params": {}, "weight": 0.9},

    {"id": "scene.desert_floor", "name": "Desert Floor", "layer": "near",
     "gen": "desert_floor", "slot": "ground",
     "tags": "sand desert dust arid dry ground wasteland barren heat "
             "salt flat badlands",
     "params": {}, "weight": 1.0},

    {"id": "scene.snow_field", "name": "Snowfield", "layer": "near",
     "gen": "snow_field", "slot": "ground",
     "tags": "snow ice winter frozen tundra arctic glacier cold white "
             "frost blizzard polar",
     "params": {}, "weight": 1.0},

    {"id": "scene.road", "name": "Vanishing Road", "layer": "near", "gen": "road",
     "slot": "ground",
     "tags": "road highway route journey travel asphalt drive perspective "
             "horizon trip departure freeway",
     "params": {}, "weight": 1.0},

    {"id": "scene.city_street", "name": "Wet Street", "layer": "near",
     "gen": "city_street", "slot": "ground",
     "tags": "street pavement sidewalk urban asphalt alley city ground "
             "downtown plaza concrete",
     "params": {}, "weight": 0.9},

    {"id": "scene.reflection", "name": "Reflection", "layer": "near",
     "gen": "reflection",
     "tags": "reflection mirror still water lake calm glass surface "
             "symmetry double inverted reflecting pool",
     "params": {}, "weight": 0.9},

    # ====================== FOREGROUND ==================================
    {"id": "scene.grass_tufts", "name": "Foreground Grass", "layer": "fg",
     "gen": "grass_tufts",
     "tags": "grass tufts foreground weeds undergrowth field close up "
             "depth meadow blades",
     "params": {"rows": 3}, "weight": 0.7},

    {"id": "scene.rocks", "name": "Rocks", "layer": "fg", "gen": "rocks",
     "tags": "rocks stones boulders rubble scree shore rocky ground "
             "gravel foreground",
     "params": {"count": 7}, "weight": 0.7},

    {"id": "scene.reeds", "name": "Reeds", "layer": "fg", "gen": "reeds",
     "tags": "reeds rushes marsh wetland pond bank riverbank cattails "
             "swamp lakeside",
     "params": {"count": 22}, "weight": 0.75},

    {"id": "scene.branch_frame", "name": "Overhanging Branch", "layer": "fg",
     "gen": "branch_frame",
     "tags": "branch leaves overhang framing foreground tree bough "
             "depth composition canopy",
     "params": {}, "weight": 0.7},

    {"id": "scene.fence", "name": "Fence", "layer": "fg", "gen": "fence",
     "tags": "fence railing barrier farm boundary gate posts "
             "enclosure rural edge",
     "params": {"rows": 4}, "weight": 0.65},

    # ====================== OVERLAY =====================================
    {"id": "scene.rain", "name": "Rain", "layer": "overlay", "gen": "rain",
     "slot": "weather",
     "tags": "rain rainy downpour drizzle shower wet storm monsoon "
             "melancholy grey weather umbrella",
     "params": {"density": 0.1, "slant": 1}, "weight": 1.05},

    {"id": "scene.snowfall", "name": "Snowfall", "layer": "overlay", "gen": "snowfall",
     "slot": "weather",
     "tags": "snow snowing snowfall blizzard flurry winter christmas "
             "cold white falling flakes",
     "params": {"density": 0.045}, "weight": 1.05},

    {"id": "scene.fog", "name": "Fog", "layer": "overlay", "gen": "fog",
     "slot": "weather",
     "tags": "fog mist haze smog gloom obscured morning damp "
             "atmospheric moody veil",
     "params": {"band": 0.62}, "weight": 1.0},

    {"id": "scene.birds", "name": "Birds", "layer": "overlay", "gen": "birds",
     "tags": "birds flock gulls flying flight wings migration sky "
             "freedom crows swallows",
     "params": {"count": 9}, "weight": 0.8},

    {"id": "scene.sparks", "name": "Embers", "layer": "overlay", "gen": "sparks",
     "tags": "sparks embers fireflies dust motes glow particles magic "
             "ash floating light",
     "params": {"count": 40}, "weight": 0.75},

    {"id": "scene.scanlines", "name": "CRT Scanlines", "layer": "overlay",
     "gen": "scanlines",
     "tags": "crt scanlines glitch retro terminal vhs analogue monitor "
             "computer screen cyber digital hacker bbs",
     "params": {"every": 3}, "weight": 0.8},
]


# ---------------------------------------------------------------------------
#  Mood vocabulary — retrieval also decides time of day and horizon height.
# ---------------------------------------------------------------------------
NIGHT_WORDS = ("night", "midnight", "dark", "nocturnal", "star", "moon",
               "evening", "dusk", "neon", "space", "cosmos", "aurora",
               "sleep", "dream", "shadow", "noir", "black")
DAY_WORDS = ("day", "morning", "noon", "sun", "sunny", "bright", "dawn",
             "daylight", "summer", "afternoon", "blue sky")

HIGH_HORIZON = ("sky", "space", "stars", "cosmos", "clouds", "aurora",
                "flying", "flight", "bird", "mountain", "storm")
LOW_HORIZON = ("ocean", "sea", "water", "lake", "field", "road", "desert",
               "street", "ground", "floor", "reflection")


def all_scene_elements() -> List[Dict]:
    return [{**e, "kind": "scene"} for e in SCENE_ELEMENTS]


def element_by_id(eid: str) -> Dict:
    for e in SCENE_ELEMENTS:
        if e["id"] == eid:
            return e
    return SCENE_ELEMENTS[0]


def scene_catalog() -> List[Dict]:
    return [{"id": e["id"], "name": e["name"], "layer": e["layer"],
             "gen": e["gen"], "tags": e["tags"]} for e in SCENE_ELEMENTS]
