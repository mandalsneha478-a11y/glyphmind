"""
app.py — GLYPHMIND HTTP surface.

Two response styles:

  * plain JSON            for catalogue and one-shot calls
  * Server-Sent Events    for every pipeline that has interesting *internals*

The SSE endpoints are the point.  A normal RAG demo shows you an answer; this
one streams the pipeline as it happens — route, prior synthesis, dense scan,
BM25, spatial scan, graph expansion, fusion, grounding — each stage arriving as
its own event with its own millisecond cost.  The front end renders that live,
so the retrieval architecture is visible rather than asserted.

Run:
    uvicorn app:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import FRONTEND_DIST, HOST, PORT
from core.architect import compose
from core.concrete import generate as generate_poem
from core.assets import get_asset_index
from core.generator import answer, get_llm
from core.index import IndexManager
from core.retriever import HybridRetriever
from corpus.diagrams import SEED_DIAGRAMS
from corpus.lexicon import CLUSTERS, EXEMPLARS, STRATEGIES
from corpus.scene_kb import scene_catalog
from corpus.shapes import shape_catalog

app = FastAPI(title="GLYPHMIND", version="1.0.0",
              description="Spatial RAG for text-native art")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

MANAGER = IndexManager()
RETRIEVER = HybridRetriever(MANAGER)
BOOT: Dict[str, Any] = {"ready": False, "started": time.time(), "seeded": []}


# ==========================================================================
#  Boot: index the seed corpus in the background so the UI paints instantly
# ==========================================================================
def _seed() -> None:
    for sd in SEED_DIAGRAMS:
        try:
            doc = MANAGER.ingest(sd["name"], sd["text"], sd["kind"])
            BOOT["seeded"].append({"id": doc.id, "name": doc.name,
                                   "blurb": sd.get("blurb", ""),
                                   "samples": sd.get("samples", [])})
        except Exception as exc:                        # pragma: no cover
            BOOT.setdefault("errors", []).append(f"{sd['name']}: {exc}")
    try:
        get_asset_index()                                # warm the art corpus
    except Exception as exc:                            # pragma: no cover
        BOOT.setdefault("errors", []).append(f"assets: {exc}")
    BOOT["ready"] = True
    BOOT["boot_ms"] = round((time.time() - BOOT["started"]) * 1000, 1)


@app.on_event("startup")
async def _startup() -> None:
    threading.Thread(target=_seed, daemon=True).start()


# ==========================================================================
#  SSE plumbing
# ==========================================================================
def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def stream_pipeline(worker) -> StreamingResponse:
    """Run `worker(emit)` on a thread and relay its emissions as SSE.

    The generative and retrieval pipelines are synchronous and CPU-bound, so
    they run off the event loop; a bounded queue carries their stage events
    back to the client as they happen.
    """
    q: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=256)

    def emit(stage: str, payload: Any) -> None:
        q.put(("stage", {"stage": stage, "payload": payload,
                         "t": round(time.time() * 1000)}))

    def run() -> None:
        t0 = time.perf_counter()
        try:
            result = worker(emit)
            q.put(("result", {"result": result,
                              "wall_ms": round((time.perf_counter() - t0) * 1000, 2)}))
        except Exception as exc:                        # pragma: no cover
            import traceback
            q.put(("error", {"message": str(exc),
                             "trace": traceback.format_exc()[-1800:]}))
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    def gen() -> Generator[str, None, None]:
        yield sse("open", {"ok": True})
        while True:
            item = q.get()
            if item is None:
                break
            kind, payload = item
            yield sse(kind, payload)
        yield sse("done", {"ok": True})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


# ==========================================================================
#  Models
# ==========================================================================
class IngestBody(BaseModel):
    name: str = Field(default="Untitled drawing", max_length=120)
    text: str = Field(min_length=1, max_length=200_000)
    kind: str = Field(default="custom", max_length=32)


class QueryBody(BaseModel):
    doc_id: str
    question: str = Field(min_length=1, max_length=600)
    use_llm: bool = True
    top_k: int = 8


class ArchitectBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=400)
    width: int = 110
    height: int = 40
    title: Optional[str] = None
    seed: Optional[int] = None
    mode: Optional[str] = None                 # "scene" | "poster" | None (auto)
    horizon: Optional[float] = None
    frame_id: Optional[str] = None
    palette_id: Optional[str] = None
    font_id: Optional[str] = None
    # pin individual depth layers; None means "let retrieval decide"
    subject_id: Optional[str] = None
    sky_id: Optional[str] = None
    far_id: Optional[str] = None
    ground_id: Optional[str] = None


class PoetryBody(BaseModel):
    theme: str = Field(min_length=1, max_length=400)
    poem: Optional[str] = Field(default=None, max_length=20_000)
    shape_id: Optional[str] = None
    strategy_id: Optional[str] = None
    width: int = 72
    height: int = 34
    seed: Optional[int] = None


# ==========================================================================
#  System
# ==========================================================================
@app.get("/api/health")
def health() -> Dict:
    idx = None
    try:
        idx = get_asset_index().kinds() if BOOT["ready"] else None
    except Exception:
        idx = None
    return {
        "ok": True,
        "ready": BOOT["ready"],
        "boot_ms": BOOT.get("boot_ms"),
        "documents": len(MANAGER.docs),
        "vectors": MANAGER.store.size(),
        "encoder": MANAGER.embedder.backend_name,
        "dims": {"semantic": MANAGER.embedder.sem_dim,
                 "spatial": MANAGER.embedder.spa_dim,
                 "fused": MANAGER.embedder.dim},
        "asset_corpus": idx,
        "llm": {"available": get_llm().available(), "model": get_llm().model},
        "errors": BOOT.get("errors", []),
    }


@app.get("/api/catalog")
def catalog() -> Dict:
    idx = get_asset_index()
    def strip(a: Dict) -> Dict:
        return {k: v for k, v in a.items() if k not in ("fn", "art")}
    return {
        "diagrams": BOOT["seeded"],
        "documents": MANAGER.list(),
        "frames": [strip(a) for a in idx.assets if a["kind"] == "frame"],
        "palettes": [strip(a) for a in idx.assets if a["kind"] == "palette"],
        "layouts": [strip(a) for a in idx.assets if a["kind"] == "layout"],
        "fonts": [strip(a) for a in idx.assets if a["kind"] == "font"],
        "textures": [strip(a) for a in idx.assets if a["kind"] == "texture"],
        "motifs": [{**strip(a), "preview": "\n".join(a["art"][:3])}
                   for a in idx.assets if a["kind"] == "motif"],
        "scenes": scene_catalog(),
        "shapes": shape_catalog(),
        "strategies": [{"key": k, **v} for k, v in STRATEGIES.items()],
        "exemplars": EXEMPLARS,
        "lexicon": [{"id": c["id"], "name": c["name"], "tags": c["tags"]}
                    for c in CLUSTERS],
    }


# ==========================================================================
#  Section 1 — Blueprint Reader
# ==========================================================================
@app.get("/api/documents")
def documents() -> Dict:
    return {"documents": MANAGER.list()}


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> Dict:
    doc = MANAGER.get(doc_id)
    if not doc:
        raise HTTPException(404, "unknown document")
    return {
        "meta": doc.meta(),
        "raw": doc.raw,
        "graph": doc.graph.to_dict(),
        "summary": doc.graph.summary(),
        "parse": doc.parse.diagnostics,
        "chunk_kinds": doc.stats["chunk_kinds"],
        "annotations": [a.to_dict() for a in doc.parse.annotations],
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> Dict:
    return {"deleted": MANAGER.delete(doc_id)}


@app.post("/api/ingest")
def ingest(body: IngestBody) -> Dict:
    doc = MANAGER.ingest(body.name, body.text, body.kind)
    return {"meta": doc.meta(), "graph": doc.graph.to_dict(),
            "summary": doc.graph.summary(), "raw": doc.raw}


@app.post("/api/ingest/stream")
def ingest_stream(body: IngestBody) -> StreamingResponse:
    def worker(emit):
        doc = MANAGER.ingest(body.name, body.text, body.kind, progress=emit)
        return {"meta": doc.meta(), "graph": doc.graph.to_dict(),
                "summary": doc.graph.summary(), "raw": doc.raw}
    return stream_pipeline(worker)


@app.post("/api/query")
def query(body: QueryBody) -> Dict:
    doc = MANAGER.get(body.doc_id)
    if not doc:
        raise HTTPException(404, "unknown document")
    ev = RETRIEVER.retrieve(doc, body.question, top_k=body.top_k)
    d = ev.to_dict()
    d["answer"] = answer(d, use_llm=body.use_llm)
    return d


@app.post("/api/query/stream")
def query_stream(body: QueryBody) -> StreamingResponse:
    doc = MANAGER.get(body.doc_id)
    if not doc:
        raise HTTPException(404, "unknown document")

    def worker(emit):
        ev = RETRIEVER.retrieve(doc, body.question, top_k=body.top_k,
                                progress=emit)
        d = ev.to_dict()
        emit("compose", {"status": "deriving grounded answer"})
        d["answer"] = answer(d, use_llm=body.use_llm)
        return d
    return stream_pipeline(worker)


# ==========================================================================
#  Section 2 — ASCII / ANSI Architect
# ==========================================================================
@app.post("/api/architect")
def architect(body: ArchitectBody) -> Dict:
    return compose(body.prompt, width=body.width, height=body.height,
                   title=body.title, seed=body.seed, mode=body.mode,
                   horizon=body.horizon, frame_id=body.frame_id,
                   palette_id=body.palette_id, font_id=body.font_id,
                   subject_id=body.subject_id, sky_id=body.sky_id,
                   far_id=body.far_id, ground_id=body.ground_id)


@app.post("/api/architect/stream")
def architect_stream(body: ArchitectBody) -> StreamingResponse:
    def worker(emit):
        return compose(body.prompt, width=body.width, height=body.height,
                       title=body.title, seed=body.seed, mode=body.mode,
                       horizon=body.horizon, frame_id=body.frame_id,
                       palette_id=body.palette_id, font_id=body.font_id,
                       subject_id=body.subject_id, sky_id=body.sky_id,
                       far_id=body.far_id, ground_id=body.ground_id,
                       progress=emit)
    return stream_pipeline(worker)


# ==========================================================================
#  Section 3 — Concrete Poetry
# ==========================================================================
@app.post("/api/poetry")
def poetry(body: PoetryBody) -> Dict:
    return generate_poem(body.theme, poem=body.poem, shape_id=body.shape_id,
                         strategy_id=body.strategy_id, width=body.width,
                         height=body.height, seed=body.seed)


@app.post("/api/poetry/stream")
def poetry_stream(body: PoetryBody) -> StreamingResponse:
    def worker(emit):
        return generate_poem(body.theme, poem=body.poem, shape_id=body.shape_id,
                             strategy_id=body.strategy_id, width=body.width,
                             height=body.height, seed=body.seed, progress=emit)
    return stream_pipeline(worker)


# ==========================================================================
#  Static front end (production build)
# ==========================================================================
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def _root() -> FileResponse:
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{path:path}", include_in_schema=False)
    def _spa(path: str) -> FileResponse:
        target = FRONTEND_DIST / path
        if target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    def _dev_root() -> HTMLResponse:
        return HTMLResponse(
            "<pre style='font:14px ui-monospace;padding:40px;line-height:1.6'>"
            "GLYPHMIND API is running.\n\n"
            "The React front end is not built yet. Either:\n"
            "  cd frontend &amp;&amp; npm run dev      (dev server on :5173)\n"
            "  cd frontend &amp;&amp; npm run build    (then reload this page)\n\n"
            "API docs: <a href='/docs'>/docs</a></pre>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)
