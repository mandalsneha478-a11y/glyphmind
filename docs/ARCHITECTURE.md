# GLYPHMIND — Architecture & Design Rationale

> A retrieval-augmented system for text-native art: reading the geometry out of
> ASCII blueprints, and writing new geometry back in.

---

## Table of contents

1. [The problem, stated precisely](#1-the-problem-stated-precisely)
2. [System overview](#2-system-overview)
3. [The 2-D substrate](#3-the-2-d-substrate)
4. [Spatial chunking — the core contribution](#4-spatial-chunking--the-core-contribution)
5. [Structural vision: recovering the graph](#5-structural-vision-recovering-the-graph)
6. [Dual-stream embeddings](#6-dual-stream-embeddings)
7. [Storage: dense, sparse, symbolic](#7-storage-dense-sparse-symbolic)
8. [Retrieval: intent routing and four-way fusion](#8-retrieval-intent-routing-and-four-way-fusion)
9. [Grounding: derive, then phrase](#9-grounding-derive-then-phrase)
10. [Subsystem 2 — the Typographic Architect](#10-subsystem-2--the-typographic-architect)
11. [Subsystem 3 — Concrete Poetry](#11-subsystem-3--concrete-poetry)
12. [The interface and the Observatory](#12-the-interface-and-the-observatory)
13. [Performance](#13-performance)
14. [Design decisions and the arguments against them](#14-design-decisions-and-the-arguments-against-them)
15. [Limitations and honest failure modes](#15-limitations-and-honest-failure-modes)
16. [Extending the system](#16-extending-the-system)

---

## 1. The problem, stated precisely

A retrieval-augmented generation pipeline has a hidden premise: that a document
is a **sequence**. Split it on paragraphs, embed the pieces, search the
embeddings. That premise is correct for prose and catastrophically wrong for
anything whose meaning lives in two dimensions.

Consider three characters from a network diagram:

```
+---+
| A |
+-+-+
  |
  v
+-+-+
| B |
+---+
```

Every fact a human extracts here — *A is above B*, *A points to B*, *the arrow
leaves A's bottom edge and enters B's top edge* — is encoded entirely in
**column alignment and row adjacency**. Tokenise this and you get something
like `+ - - - + | A | + - + - + | v + - + - + | B | + - - - +`. The vertical
relationship has been deleted. No amount of model capacity recovers it, because
the information is no longer in the input.

The same failure runs in the opposite direction when *generating*. Ask a
language model for ASCII art and it predicts characters one at a time with no
representation of a column. The border loses a cell halfway down; the figlet
title overruns its frame by three characters; the caption lands one off centre.
The output is *almost* right, which in typography is the same as wrong.

GLYPHMIND attacks both directions of this problem with the same conviction:

> **Geometry must be represented explicitly, retrieved explicitly, and solved —
> never inferred from a flattened sequence.**

---

## 2. System overview

```
                        ┌──────────────────────────────────────────┐
   raw text drawing ───►│  Canvas          2-D character matrix    │
                        │                  glyph-class maps        │
                        └───────────────┬──────────────────────────┘
                                        │
                ┌───────────────────────┼───────────────────────┐
                ▼                       ▼                       ▼
     ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
     │  SpatialChunker    │  │   AsciiParser      │  │  (geometry reused  │
     │  tiles · regions   │  │   rects · wires    │  │   by the Architect │
     │  bands · scales    │  │   arrows · labels  │  │   and the Poetry   │
     └─────────┬──────────┘  └─────────┬──────────┘  │   compositors)     │
               │                       │             └────────────────────┘
               ▼                       ▼
     ┌────────────────────┐  ┌────────────────────┐
     │  FusedEmbedder     │  │   DiagramGraph     │
     │  [ sem ‖ spatial ] │  │   networkx         │
     └─────────┬──────────┘  └─────────┬──────────┘
               ▼                       │
     ┌────────────────────┐            │
     │ VectorStore + BM25 │            │
     └─────────┬──────────┘            │
               └───────────┬───────────┘
                           ▼
              ┌────────────────────────────┐
              │     HybridRetriever        │
              │  intent route → R1·R2·R3·R4│
              │  → RRF fusion → grounding  │
              └─────────────┬──────────────┘
                            ▼
              ┌────────────────────────────┐
              │   GroundedComposer         │
              │   (+ optional local LLM    │
              │    narration on top)       │
              └────────────────────────────┘
```

Three user-facing subsystems sit on this spine:

- **§1 Blueprint Reader** uses the whole pipeline as drawn.
- **§2 ASCII / ANSI Architect** reverses it: retrieval returns *parts with
  geometry*, and a solver composes them.
- **§3 Concrete Poetry** reuses the retrieval layer against a third corpus —
  shape formulas, calligram strategies, lexical clusters — and solves a
  constrained packing problem.

---

## 3. The 2-D substrate

`core/canvas.py`

A `Canvas` is a padded rectangular `numpy` array of single characters, plus
derived maps computed once at construction:

| map | meaning |
|---|---|
| `grid` | `(H, W)` of characters |
| `classes` | `(H, W)` glyph-class indices |
| `occupancy` | non-whitespace |
| `structure` | box-drawing glyphs only |
| `text_mask` | alphanumeric and punctuation |

**Glyph taxonomy.** Eleven classes: space, corner, horizontal, vertical,
junction, diagonal, arrow, shade, alpha, digit, punct. This covers ASCII
(`+-|/\`) and the full Unicode box-drawing range (light, heavy, double,
rounded) in one vocabulary, so a diagram drawn with `┌─┐` and one drawn with
`+-+` parse identically.

**Column honesty.** East-Asian-wide characters occupy two terminal columns.
`_narrow()` compensates so that one array index always equals one rendered
column. Without this every downstream coordinate silently drifts on any canvas
containing CJK text.

**`BBox` algebra.** Inclusive-exclusive `(r0, c0, r1, c1)` rectangles with
intersection, IoU, directional overlap fraction, centroid and padding. Every
chunk, node, edge, slot and highlight in the entire system is a `BBox`. It is
the lingua franca that lets the retriever, the graph and the UI agree on what
"here" means.

---

## 4. Spatial chunking — the core contribution

`core/spatial_chunker.py`

Four chunk families are emitted **simultaneously**, because different questions
want different units and you cannot know in advance which.

### 4.1 Tile chunks — multi-scale overlapping windows

Three window sizes — `(6×24)`, `(10×40)`, `(16×64)` — swept at 50 % stride.

Why three scales: a fine tile captures one component; a coarse tile captures a
whole subsystem and the relationships inside it. Why 50 % overlap: a feature cut
exactly in half at one scale and one offset is intact at another. The union
guarantees no feature is severed at *every* scale at once.

### 4.2 Region chunks — gap-tolerant connected components

Binary dilation with a `(2t+1)` square kernel, then 8-connected labelling, at
three tolerances:

| tolerance | what fuses | the unit you get |
|---|---|---|
| `0` | nothing | individual words, glyph clusters |
| `1` | adjacent glyphs | a whole box including its border |
| `3` | across short gaps | **a relation**: box + connector + target |

Tolerance 3 is the interesting one. `[ web ]--->[ api ]` becomes a *single
retrievable chunk*, which is exactly the unit you want when the question is
"what talks to the database?".

### 4.3 Band chunks

Full-width row bands and full-height column bands. These are what make "what is
in the top third" or "which column does the trunk line run down" answerable at
all.

### 4.4 Structural chunks

One chunk per parsed node and per parsed edge, carrying the symbolic id in its
payload. These are the chunks that make hybrid retrieval genuinely hybrid: they
are addressable **both** by vector similarity and by graph traversal.

### 4.5 Verbalising geometry

A pre-trained sentence encoder has never seen a box-drawing character. So each
chunk generates a natural-language *caption* of its own geometry:

```
node region 3 rows by 24 columns; labelled 'API Gateway';
containing text: API Gateway; drawn with corners, horizontal lines,
vertical lines; located in the top-centre area
```

The caption is what gets embedded semantically. This is the bridge between the
geometric world and the semantic one, and it costs nothing at query time.

A 60×100 diagram yields roughly **190 chunks**.

---

## 5. Structural vision: recovering the graph

`core/ascii_parser.py`

This module performs computer vision on a character matrix. Five stages.

### 5.1 Node detection

**Rectangle scan.** For each top-left corner glyph, walk the top edge to a
top-right corner, drop both verticals, and accept when a bottom edge closes with
≥ 60 % solidity on all four sides. The tolerance matters: real diagrams have
junction characters embedded in their borders where connectors attach.

**Bracket scan.** `[ label ]`, `( label )`, `{ label }`, `< label >` — the idiom
of quick sketches and finite-state diagrams.

**Fallback.** A canvas with no rectangles at all (an indented tree, a legend)
falls back to isolated text clusters, so the graph layer always has something to
reason over.

**Nested-rectangle label correction.** A container rectangle naively inherits
the text of everything it encloses — a rack slot would be labelled with the name
of the server inside it. Each label is therefore recomputed from only those
glyphs belonging to no child rectangle. A wrapper holding exactly one child and
carrying no text of its own is discarded as noise.

### 5.2 Wire segmentation

Mask every structural glyph *not* inside a node border, then take 8-connected
components. Each component is one physical wire run.

**Inline-label bridging.** `[ DRAFT ] ---- submit ----> [ PENDING ]` draws one
logical connector, but the word `submit` physically severs the glyph run. A
naive component pass sees two half-wires that each touch only one node and
silently drops the edge — and dropped edges are how a diagram pipeline becomes
useless without anyone noticing. So the flood fill may *jump* along a cardinal
axis across cells that are non-node and non-wire, provided at least one of them
is alphanumeric (it really is a label, not empty space). Horizontal budget 14
columns, vertical 4 rows.

### 5.3 Incidence and direction

For each wire run, find every node whose border it touches — distance 1 in all
eight directions, plus distance 2 on the cardinals so that `[ NODE ] --->` with
one space of padding still registers.

**Contextual arrowhead disambiguation.** `v` is a letter in *Svc*, *Archive* and
*VLAN*, and an arrowhead under a pipe. `is_arrow_glyph()` promotes a soft
arrowhead only when a wire glyph sits on the matching side **and** no letter
touches it horizontally. Without this rule the parser reports a node called
`Auth S c`. With it, `Auth Svc`.

A run touching two nodes produces one edge. A run touching N nodes is a bus and
expands into pairwise edges flagged `bus=True`.

### 5.4 Labelling

Text glyphs within two columns of a wire, belonging to no node and not
themselves arrowheads, are grown outward across whole words and become the edge
label.

### 5.5 Co-reference

A finite-state diagram draws `[ PENDING ]` twice when two transitions re-enter
it. Geometrically those are two rectangles; logically one state. Both instances
are kept — position still matters for spatial questions — but stitched together
in the undirected view by `alias` edges so reachability behaves correctly.
Alias edges are filtered out of neighbour listings so they never pollute an
answer.

**Measured recovery on the seed corpus:**

| drawing | nodes | edges | style |
|---|---|---|---|
| Payments Platform | 10 | 11 | box-and-arrow, directed, labelled |
| Campus Network | 10 | 11 | bus topology with a policy loop |
| Level 2 Floor Plan | 7 | 9 | rooms on a corridor bus |
| Order Lifecycle | 8 | 9 | bracket FSM with repeated states |
| Rack A14 Elevation | 6 | 15 | nested rectangles |

---

## 6. Dual-stream embeddings

`core/embedder.py`

A pre-trained sentence encoder is extraordinary at *what does this mean* and
blind to *where was this*. A geometric descriptor is the exact mirror. Run both.

```
v(chunk) = [ W_sem · L2(SBERT(caption))  ‖  W_spa · L2(Φ(geometry)) ]
```

Because each half is L2-normalised **before** weighting, the dot product
decomposes exactly:

```
sim(a,b) = W_sem² · cos_sem(a,b) + W_spa² · cos_spa(a,b)
```

So the contribution of each stream to any score is recoverable — and the
Observatory displays it per hit, live. Retrieval here is not a black box; it is
an audited one.

### Φ — the 72-dimensional spatial descriptor

| dims | feature |
|---|---|
| 0–3 | normalised bbox `(r0, c0, h, w)` |
| 4–5 | normalised centroid |
| 6 | log-squashed aspect ratio |
| 7 | area fraction of the canvas |
| 8 | ink density |
| 9–19 | glyph-class histogram (11 classes) |
| 20–35 | 4×4 occupancy grid — the chunk's own thumbnail |
| 36–51 | 4×4 structural-ink grid |
| 52–55 | border-contact flags |
| 56–57 | horizontal / vertical symmetry |
| 58–59 | mean run lengths |
| 60–63 | directional gradient energies (↕ ↔ ╱ ╲) |
| 64–67 | row / column profile moments |
| 68–71 | corner, junction, arrow, shade saliency |

These are hand-engineered, and deliberately so. Nothing in the pre-training
distribution of any text encoder teaches it that a 4×4 occupancy thumbnail
discriminates a box from a connector, but it does, and it costs 72 floats.

### Graceful degradation

`SentenceEncoder` lazily loads `all-MiniLM-L6-v2`. If the download fails, the
machine is offline, or `GLYPHMIND_OFFLINE=1` is set, `HashedEncoder` takes over:
hashed word + character-trigram + bigram projection into 384 dimensions with
sublinear damping. Weaker recall, identical interface, zero download, instant
boot. **The system is never unavailable because a model was unavailable.**

---

## 7. Storage: dense, sparse, symbolic

`core/vector_store.py`, `core/graph_store.py`

### VectorStore

An exact brute-force cosine scan over a contiguous `float32` matrix, namespaced
(`diagram:<id>`, `components`, `lexicon`, `shapes`).

*Why not FAISS or Chroma?* The corpora here are thousands of vectors, not
billions. Exact scan is faster in wall-clock (no index build, no warm-up), has
no recall loss, and — decisively — is **fully introspectable**. `search_split()`
returns the semantic and spatial components of every score separately, which is
what the UI renders. An ANN index would have bought approximation in exchange
for the one property this project most wanted to demonstrate.

### BM25

Classic Okapi (`k1=1.4`, `b=0.72`) over chunk captions plus readable text. It
catches exact identifiers — `sw-a14-01`, `vlan 20`, `cn-a14-001` — that dense
retrieval smooths away. Dense and sparse fail in different directions; that is
precisely why both are present.

### DiagramGraph

`networkx.MultiDiGraph` for directed semantics, plus an undirected view for
reachability. It answers:

- neighbours (in / out / undirected), degree
- all simple paths, shortest path
- k-hop expansion
- betweenness centrality blended with normalised degree
- connected components, cycles, roots, sinks, isolated nodes
- **spatial relations** computed from bbox centroids: `nodes_in_direction`,
  `nodes_in_quadrant`, `spatial_relation`

Everything this module returns is *derived*. It is the grounding layer.

---

## 8. Retrieval: intent routing and four-way fusion

`core/retriever.py`

### 8.1 Intent routing

Regex families classify each question into `path`, `topology`, `containment`,
`direction`, `inventory`, `connectivity`, or `semantic`, resolved by a
most-specific-wins priority. *"What is inside the north wing?"* mentions a
compass word but is a containment question, so containment outranks direction.

### 8.2 Entity resolution

Longest-label-first exact containment, then fuzzy token-overlap with a Jaccard
floor, then ordering by position in the question — so *"from A to B"* keeps A
before B, which matters because `paths(a, b)` is not symmetric.

### 8.3 Spatial prior synthesis

This is the step that makes the spatial half of the vector *do* something at
query time.

Locative language is resolved into an actual region of the canvas:

- *"what sits below the message broker"* → a `BBox` underneath the broker's
  rectangle, padded sideways
- *"what is in the top-right"* → the top-right quadrant `BBox`
- a named entity with no direction → the entity's own `BBox`

That rectangle is pushed through the **same Φ descriptor used at index time**.
The ANN search is then literally searching *a place*, not a phrase.

### 8.4 The four retrievers

| | retriever | what it is good at |
|---|---|---|
| **R1** | dense cosine over `[sem ‖ spa]` | meaning and form together |
| **R2** | BM25 over captions | exact identifiers, rare tokens |
| **R3** | spatial-only scan against the prior | pure position |
| **R4** | graph expansion: resolve → k-hop → chunks overlapping that neighbourhood | relational structure |

**R4 is the one ordinary RAG cannot have**, because ordinary RAG never recovered
a graph in the first place.

### 8.5 Fusion

Reciprocal rank fusion, robust to the fact that a cosine score and a BM25 score
live on incomparable scales:

```
RRF(d) = Σ_r  w_r / (K + rank_r(d)),     K = 60
```

Weights are intent-dependent:

| intent | adjustment |
|---|---|
| `direction`, `containment` | spatial ↑ 1.35, lexical ↓ 0.5 |
| `path`, `connectivity`, `topology`, `inventory` | graph ↑ 1.5 |

Then a **kind prior** (structural chunks are the most citable units — node 1.25,
edge 1.20, document 0.75), then a **diversity pass** that drops any candidate
with IoU > 0.72 against an already-selected chunk of the same kind, so you never
get eight near-identical overlapping tiles.

---

## 9. Grounding: derive, then phrase

`core/generator.py`

**The answer is computed before it is worded.**

`derive_facts()` executes real operations against the graph and the grid and
returns typed, structured facts — `path`, `shortest_path`, `connectivity`,
`direction`, `quadrant`, `topology`, `cycles`, `inventory`, `containment` —
alongside a trace of the operations performed and a list of highlight
rectangles.

`GroundedComposer` renders those facts into prose **deterministically**. Each
intent has its own renderer. It cannot hallucinate, because it has no generative
freedom: every sentence is a rendering of a computed fact, and every claim
carries the chunk id it came from.

The optional `LocalLLM` sits strictly *on top*. If an Ollama daemon is
reachable, it receives the facts and the already-correct derived answer and is
asked only to rephrase under a strict prompt. If it is absent, unreachable, slow
or returns something degenerate, the deterministic answer stands. The system is
fully functional with zero models and merely more fluent with one.

This inverts the usual arrangement — retrieve, then hope — and is the reason the
Reader can assert *"every route passes through Edge CDN"* and be right.

---

## 10. Subsystem 2 — the Typographic Architect

`core/architect.py`, `core/sceneforge.py`, `corpus/scene_kb.py`

Same engine, pointed the other way. Instead of retrieving regions *out of* a
drawing, retrieve recipes to build one.

### 10.1 Why sprite stamping was abandoned

The first version of this subsystem retrieved hand-drawn sprites and stamped
them onto a canvas. It failed on both axes the brief cares about:

- **Arbitrary prompts.** A fixed sprite library can only draw the things
  somebody already drew. "A lighthouse in a storm" had nothing to say.
- **Detail.** A 35×7 sprite on a 110×40 canvas leaves ninety per cent of the
  picture empty, which reads as thin rather than as art.

So the Architect no longer stamps pictures. It retrieves **parametric
generators** and runs them.

### 10.2 The scene knowledge base

`corpus/scene_kb.py` holds ~45 **scene recipes**. A recipe is not an image; it
is a depth layer, a generator name, the parameters that generator should be
called with, and a deliberately broad tag surface for retrieval. Mutual
exclusion is declared with `slot`, so a scene gets one celestial body, one
horizon, one ground plane, one subject.

### 10.3 The retrieval pipeline

**1 · Prompt decomposition.** The prompt is split on conjunctions and
prepositions and then on content words, giving up to nine independent queries.

> This is the single biggest accuracy win in the subsystem. One embedding of a
> whole sentence is an *average*, and averages destroy the minority concept.
> "A lighthouse in a storm over the sea" averaged is mostly weather; decomposed
> it is three precise queries that each retrieve their own element.

**2 · Multi-query max-pooled retrieval.** Every query searches the scene KB.
Scores are max-pooled per element and scaled by a per-element prior.

**3 · Layer assignment.** Winners are slotted into sky / far / mid / near /
foreground / overlay. Mandatory layers are back-filled from defaults, so *no
prompt can produce an empty canvas*. A second far band is admitted when it uses
a different generator — otherwise "snowy mountains **with pine forest**" would
have to choose between its own two nouns.

**4 · Relative acceptance thresholds.** Thresholds are expressed as a fraction
of the top pooled score, never as absolute cosine values.

> Absolute cut-offs do not survive a change of encoder: the hashed fallback and
> a transformer put their scores on completely different scales, and a
> threshold tuned for one admits junk on the other. This was observable — "a
> neon city skyline" was pulling a lighthouse into the mid layer because 0.19
> happened to clear a fixed 0.17. A relative cut-off asks the only question
> that matters: *is this element competitive with the best thing the prompt
> retrieved?*

**5 · Principled suppression.** If the horizon element is deep space, the
ground plane is suppressed entirely — vacuum has no floor. If the horizon
element is itself the subject (a city skyline at 0.78), no second focal object
is planted in front of it.

**6 · The unknown-subject ladder.** When no scene recipe clears the bar:

| tier | source | result |
|---|---|---|
| 1 | scene KB | a procedural subject |
| 2 | shape-formula corpus | a volumetrically shaded silhouette |
| 3 | typography | the subject word built as monumental architecture |

Every prompt therefore terminates in something deliberate, and the scene graph
in the UI names which tier fired and why.

### 10.4 Procedural generators

`core/sceneforge.py` implements ~45 generators across six depth layers. They
are *functions of resolution*, not fixed art:

- **fractal midpoint displacement** computes a new mountain ridge every run
- **skylines** grow buildings with setbacks, antennas and lit window grids
- **oceans** derive wave rows with perspective compression — short marks packed
  tightly at the horizon, long marks widely spaced underfoot
- **reflections** are computed from whatever has already been drawn above the
  waterline, with fade and jitter
- **value noise** (bilinear-upsampled, multi-octave) drives clouds, fog and
  terrain mottling
- **silhouettes** fill any implicit shape with distance-to-edge shading lit from
  the upper left

Rendering is painter's algorithm back to front. Each cell records a glyph, a
luminance, its depth layer and the element that painted it — so ANSI colour is
assigned **semantically**, by layer and luminance, rather than by position.

### 10.5 Density discipline

Three rules keep a scene legible, each of which was a visible bug first:

1. **Terrain stops at the horizon.** Letting a far-layer fill run to the bottom
   of the canvas left the near layer nothing to say and turned every scene into
   a solid block.
2. **Ground planes thin toward the horizon.** A uniform fill reads as a wall;
   coverage that grows toward the viewer reads as a field.
3. **Weather falls in sheets.** Scattering individual rain strokes at random
   reads as confetti over the whole picture; broken diagonal lines on a regular
   column period read as rain, and the gaps keep the scene underneath visible.

The validation pass reports `ink_ratio`, and the self-test asserts it lands in
a sane band — too sparse and too dense are both failures.

### 10.6 Typography

Composition runs in one of two modes. **Scene** keeps the art full-bleed and
sets the title on a cleared plate. **Poster** reserves a title band and runs the
typography solver:

Every (typeface × line-count) combination is rendered and **measured** against
the integer slot rectangle. Overrunning candidates are rejected outright. Among
those that fit, the winner maximises slot coverage tempered by retrieval rank —
first-fit would hand a poster to a three-row face in an eight-row slot and the
result would look timid. The UI prints the full accept/reject trace.

## 11. Subsystem 3 — Concrete Poetry

`core/concrete.py`, `corpus/shapes.py`, `corpus/lexicon.py`

Making a poem *look like* a tree is not a styling task. It is a constrained
packing problem over a non-convex region where the atoms have indivisible
widths, reading order must stay linear, and the result has to survive being read
aloud.

### The corpora

- **21 shape formulas** — implicit functions `fn(x, y) → bool` over normalised
  space, paired with an SVG outline. Not pictures: formulas, so they rasterise
  cleanly at any character resolution.
- **8 calligram exemplars** — Apollinaire's *Il Pleut*, Herbert's *Easter
  Wings* and *The Altar*, Mallarmé's *Un Coup de Dés*, Gomringer's *Silencio*,
  Carroll's *Mouse's Tale*, Finlay's *Wave/Rock*. These supply **strategies**,
  not words: `slanted-columns`, `taper-and-recover`, `scattered-field`,
  `grid-with-hole`, `serpentine-thread`, `interference-bands`.
- **12 lexical clusters** — grief, love, time, nature, sea, fire, freedom,
  night, city, hope, memory, transformation — each with nouns, verbs,
  modifiers, and hand-written fragments.

### The pipeline

1. **RETRIEVE** the shape, the exemplar (→ flow strategy), and two or three
   lexical clusters.
2. **RASTERISE** the formula into a boolean mask at the exact requested
   character resolution, 2×2 supersampled. Supersampling is not cosmetic: at one
   sample per cell a thin limb — a branch, a wing — alternates on and off and the
   silhouette breaks. Cell aspect (~0.52) is corrected so a circle reads round.
3. **ASSEMBLE VOCABULARY** — either the user's own poem, or recombination inside
   the retrieved lexical field using weighted syntactic patterns. Deliberately
   *not* free-form generation: recombination inside a retrieved field is what
   keeps a 400-word calligram thematically coherent instead of drifting.
4. **PACK SCANLINES** — each mask row is decomposed into runs. Runs are filled
   greedily with whole words under a **syllable-weighted cost**:

   ```
   want = 1 + 2 · thickness          # thickness = run_width / widest_run
   cost = |syllables(w) − want| + 0.35 · lookahead
   ```

   Polysyllabic words land where the figure is broad; the line thins to
   monosyllables where the figure narrows. That is the Herbert manoeuvre,
   automated — form performing the argument.

   When a run is narrower than every nearby word, the solver searches 40 words
   ahead, then falls back to a function word, and only hyphenates as a last
   resort. A silhouette with a hole in it is worse than a hyphenated line.
5. **EMIT** — the character grid, and an SVG with per-row monospace tspans over
   a faint tracing of the shape outline.

**Validation:** shape fidelity = glyphs placed / mask cells. Typical 0.55–0.85
depending on how aggressively the retrieved strategy thins the text.

---

## 12. The interface and the Observatory

`frontend/`, React 18 + Vite, no component library.

Every heavy endpoint has a **Server-Sent Events twin**. The synchronous,
CPU-bound pipelines run on a worker thread; a bounded queue relays stage events
to the client as they happen.

The **Observatory** panel renders that stream: which intent fired, what spatial
prior was synthesised and over which rectangle, what each of the four retrievers
returned and at what rank with semantic/spatial score split, how fusion weighted
them, which graph operations ran, and the real millisecond cost of every stage.

The **AsciiStage** component is the visual argument of the whole project. It
measures the exact cell metrics of the rendered monospace text from a hidden
probe, then places absolutely-positioned overlay boxes at `(r0,c0)→(r1,c1)`.
Hover a retrieval hit and **the region of the picture it came from lights up**.
Magenta for resolved entities, green for neighbours and matches, amber for
waypoints and hubs, dashed cyan for retrieved chunk rectangles.

The **GraphView** deliberately uses no force-directed layout. The parser
recovered each node's real bounding box, so the graph is drawn at its true
coordinates — a faithful vector tracing of the ASCII, which makes a bad parse
immediately obvious rather than cosmetically hidden.

---

## 13. Performance

Measured by `python run.py --check` on a laptop, offline encoder:

| operation | time |
|---|---|
| index a 32×79 diagram end-to-end | **~180 ms** |
| ├ canvas construction | ~3 ms |
| ├ structural parse | ~25 ms |
| ├ 2-D chunking (188 chunks) | ~45 ms |
| ├ dual-stream embedding | ~100 ms |
| └ graph construction | ~2 ms |
| retrieval + grounded answer | **1–4 ms** |
| compose an ANSI poster (104×34) | **~50 ms** |
| shape a concrete poem (74×34) | **~10 ms** |
| full cold boot, 5 diagrams, 887 vectors | **~1.0 s** |

With the transformer encoder, indexing rises to roughly 400–700 ms per diagram
and cold boot to a few seconds; query latency is unchanged, because the question
is a single short encode.

Retrieval is in the low single-digit milliseconds because exact cosine over
~200 × 456 floats is trivially fast and the graph operations are over tens of
nodes. There is no network hop anywhere in the request path.

---

## 14. Design decisions and the arguments against them

**Exact scan instead of an ANN index.**
*Against:* it does not scale to millions of chunks. *For:* at this corpus size
it is faster in wall-clock, loses no recall, and is introspectable — which is
the property the project exists to demonstrate. Swapping in FAISS behind the
`VectorStore` interface is a contained change if the corpus ever grows.

**Hand-engineered spatial features instead of a learned visual encoder.**
*Against:* a CNN over a rendered bitmap might capture more. *For:* 72
interpretable floats, no training data, no GPU, millisecond extraction, and a
score decomposition a human can read. The features encode exactly the
invariances that matter here.

**Deterministic composition instead of LLM generation.**
*Against:* the prose is more uniform than a model would write. *For:* it cannot
hallucinate, it needs no weights, and it is instant. The LLM is available as a
narration layer for whoever wants the fluency, and the facts stay authoritative
either way.

**Four retrievers instead of one.**
*Against:* more moving parts. *For:* they fail in genuinely different
directions. Dense misses exact identifiers; BM25 misses paraphrase; neither has
any notion of position; none of them can traverse a graph. RRF is the cheapest
honest way to combine rankings on incomparable scales.

**Verbalising geometry for the semantic encoder.**
*Against:* it is a translation, and translations lose information. *For:* it
lets a general-purpose pre-trained encoder participate at all, at zero training
cost — and the information it loses is precisely the information Φ carries in
parallel.

---

## 15. Limitations and honest failure modes

- **Hand-drawn irregularity.** The rectangle scanner needs ≥ 60 % edge
  solidity. Sketchy, deliberately broken or heavily stylised borders will be
  missed; the text-cluster fallback keeps the system usable but the graph will
  be thin.
- **Diagonal connectors.** `/` and `\` are classified and included in wire
  masks, but incidence is strongest on cardinal contacts. Heavily diagonal
  layouts recover fewer edges than orthogonal ones.
- **Bus semantics.** An N-way bus expands into pairwise edges. For a floor-plan
  corridor that is the right reading; for a shared data bus it slightly
  overstates direct connectivity. Both cases are flagged `bus=True` so a
  consumer can distinguish them.
- **Intent routing is regex-based.** It is fast, inspectable and easy to extend,
  but a question phrased far outside the patterns falls through to the semantic
  route — which still answers, just without the symbolic derivation.
- **Poem coherence.** Recombination inside a retrieved lexical field produces
  striking juxtaposition and genuine thematic unity, not argument. It is closer
  to Burroughs than to Herbert. Supplying your own poem gives the packing solver
  real verse to work with and is the stronger mode.
- **Shape fidelity varies by figure.** An hourglass packs to ~0.54 because its
  waist is one or two cells wide; a bird packs to ~0.84. The metric is reported
  rather than hidden.

---

## 16. Extending the system

**Add a diagram to the seed corpus** — append to `backend/corpus/diagrams.py`
with `name`, `kind`, `text` and a few sample questions.

**Add a motif, frame, palette or layout** — append to
`backend/corpus/components.py`. The asset index embeds `name + tags + geometry`
automatically on next boot; nothing else needs touching.

**Add a shape** — append to `backend/corpus/shapes.py`: an `fn(x, y) → bool`
over `[-1, 1]²`, an SVG path in a `0 0 100 100` viewBox, and tags. It
immediately appears in the picker and in retrieval.

**Add a lexical cluster or calligram strategy** — append to
`backend/corpus/lexicon.py`.

**Add a query intent** — add a pattern to `INTENT_PATTERNS`, a branch in
`derive_facts()`, and a renderer method on `GroundedComposer`. Those three
touch-points are the whole contract.

**Swap the encoder** — set `GLYPHMIND_EMBED_MODEL`. The vector store detects a
dimension change on load and triggers a clean reindex rather than silently
comparing incompatible vectors.
