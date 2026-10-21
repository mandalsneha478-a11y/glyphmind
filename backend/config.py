"""
GLYPHMIND global configuration.

Everything is local-first.  No cloud keys are required anywhere in the system;
the only optional network touch is the one-time download of the pre-built
sentence-transformers encoder, and even that degrades gracefully.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
#  Paths
# --------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

for _p in (DATA_DIR, INDEX_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
#  Embedding model
# --------------------------------------------------------------------------
# Pre-built, local, ~90 MB.  Cached under ~/.cache/huggingface after first run.
EMBED_MODEL_NAME = os.getenv("GLYPHMIND_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Set GLYPHMIND_OFFLINE=1 to skip the transformer entirely and use the
# built-in hashed character/word n-gram encoder (zero download, instant boot).
FORCE_OFFLINE_ENCODER = os.getenv("GLYPHMIND_OFFLINE", "0") == "1"

SEMANTIC_DIM_FALLBACK = 384      # dimension of the fallback hashed encoder

# --------------------------------------------------------------------------
#  Spatial feature engineering
# --------------------------------------------------------------------------
SPATIAL_DIM = 72                 # dimensionality of the hand-engineered vector
SPATIAL_GRID = 4                 # occupancy grid is SPATIAL_GRID x SPATIAL_GRID

# Fusion weights for the late-fusion embedding  [semantic | spatial]
SEMANTIC_WEIGHT = 1.00
SPATIAL_WEIGHT = 0.55

# --------------------------------------------------------------------------
#  Chunking
# --------------------------------------------------------------------------
TILE_SIZES = [(6, 24), (10, 40), (16, 64)]   # (rows, cols) multi-scale windows
TILE_OVERLAP = 0.5                            # 50 % stride overlap
MIN_TILE_DENSITY = 0.015                      # discard near-empty tiles
REGION_GAP_TOLERANCE = 1                      # chars of whitespace bridged when
                                              # growing a connected glyph region

# --------------------------------------------------------------------------
#  Retrieval
# --------------------------------------------------------------------------
TOP_K_DENSE = 14
TOP_K_LEXICAL = 14
TOP_K_SPATIAL = 10
RRF_K = 60                       # reciprocal-rank-fusion damping constant
GRAPH_HOPS = 2                   # expansion radius around seed nodes
GRAPH_BOOST = 0.35               # score bonus for graph-reachable chunks
FINAL_TOP_K = 8

# --------------------------------------------------------------------------
#  Optional local LLM (Ollama).  Purely optional — the deterministic grounded
#  composer produces complete answers on its own.
# --------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "25"))
USE_LLM = os.getenv("GLYPHMIND_USE_LLM", "auto")   # auto | always | never

# --------------------------------------------------------------------------
#  Server
# --------------------------------------------------------------------------
HOST = os.getenv("GLYPHMIND_HOST", "127.0.0.1")
PORT = int(os.getenv("GLYPHMIND_PORT", "8000"))
