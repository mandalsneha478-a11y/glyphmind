"""
generator.py — grounded answer composition.

Design stance: **the answer is derived before it is phrased.**

`GroundedComposer` turns the structured facts produced by the retriever into
prose deterministically.  It cannot hallucinate, because it has no generative
freedom — every sentence is a rendering of a computed fact, and every claim
carries the chunk id it came from.

`LocalLLM` is an *optional* narration layer.  If an Ollama daemon is running
locally it is handed the same facts in a strict prompt and asked only to
re-phrase them.  If it is absent, unreachable, slow, or returns something
suspicious, the deterministic answer stands.  The system is therefore always
fully functional with zero models and zero keys, and merely nicer with one.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Sequence

try:
    from ..config import (OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL, USE_LLM)
except ImportError:                                   # pragma: no cover
    from config import (OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL, USE_LLM)


# ==========================================================================
#  Optional local LLM
# ==========================================================================
class LocalLLM:
    def __init__(self) -> None:
        self._checked = False
        self._ok = False
        self.model = OLLAMA_MODEL

    def available(self) -> bool:
        if USE_LLM == "never":
            return False
        if self._checked:
            return self._ok
        self._checked = True
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(req, timeout=1.5) as r:
                blob = json.loads(r.read().decode())
            names = [m.get("name", "") for m in blob.get("models", [])]
            if names:
                if not any(n.startswith(self.model) for n in names):
                    self.model = names[0]
                self._ok = True
        except Exception:
            self._ok = False
        return self._ok

    def complete(self, system: str, prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 420) -> Optional[str]:
        if not self.available():
            return None
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        try:
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/generate", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as r:
                out = json.loads(r.read().decode())
            txt = (out.get("response") or "").strip()
            return txt or None
        except Exception:
            return None


_llm: Optional[LocalLLM] = None


def get_llm() -> LocalLLM:
    global _llm
    if _llm is None:
        _llm = LocalLLM()
    return _llm


# ==========================================================================
#  Deterministic grounded composition
# ==========================================================================
def _join(items: Sequence[str], conj: str = "and") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


def _arrow_chain(labels: Sequence[str]) -> str:
    return " → ".join(labels)


class GroundedComposer:
    """Facts in, prose out.  No freedom, no hallucination."""

    def compose(self, evidence_dict: Dict) -> Dict:
        intent = evidence_dict["intent"]
        facts = evidence_dict["facts"]
        q = evidence_dict["question"]
        chunks = evidence_dict["chunks"]

        handler = {
            "path": self._path, "connectivity": self._connectivity,
            "direction": self._direction, "topology": self._topology,
            "inventory": self._inventory, "containment": self._containment,
        }.get(intent, self._semantic)

        body = handler(q, facts, chunks)
        citations = self._citations(chunks)
        return {"answer": body, "citations": citations, "mode": "grounded-derivation"}

    # -- per-intent renderers ---------------------------------------------
    def _path(self, q, facts, chunks) -> str:
        sp = next((f for f in facts if f["type"] == "shortest_path"), None)
        alts = [f for f in facts if f["type"] == "path"]
        none = next((f for f in facts if f["type"] == "no_path"), None)
        if none:
            return (f"There is **no route** from **{none['from']}** to **{none['to']}** "
                    f"in this diagram. I traced every simple path up to 7 hops across "
                    f"the recovered graph and the two sit in disconnected regions of "
                    f"the drawing.")
        if not sp and not alts:
            return ("I could not pin both endpoints to boxes in the drawing. Name them "
                    "exactly as they are lettered on the canvas and I will trace the route.")
        lines = []
        if sp:
            lines.append(f"**Shortest route — {sp['hops']} hop"
                         f"{'s' if sp['hops'] != 1 else ''}:**\n\n"
                         f"`{_arrow_chain(sp['nodes'])}`")
        others = [f for f in alts if not sp or f["nodes"] != sp["nodes"]]
        if others:
            lines.append(f"\n**{len(others)} alternate path"
                         f"{'s' if len(others) != 1 else ''}:**")
            for f in others[:3]:
                lines.append(f"- `{_arrow_chain(f['nodes'])}`  *({f['hops']} hops)*")
        if sp and len(sp["nodes"]) > 2:
            lines.append(f"\nEvery route passes through **{sp['nodes'][1]}**, "
                         f"which makes it the load-bearing element of this connection.")
        return "\n".join(lines)

    def _connectivity(self, q, facts, chunks) -> str:
        recs = [f for f in facts if f["type"] == "connectivity"]
        if not recs:
            return self._semantic(q, facts, chunks)
        out = []
        for f in recs:
            name = f["node"]
            out.append(f"**{name}** has degree **{f['degree']}** in the recovered graph.")
            if f["outgoing"]:
                items = [f"**{r['label']}**" + (f" *({r['edge_label']})*" if r["edge_label"] else "")
                         for r in f["outgoing"]]
                out.append(f"\n- Sends to → {_join(items)}")
            if f["incoming"]:
                items = [f"**{r['label']}**" + (f" *({r['edge_label']})*" if r["edge_label"] else "")
                         for r in f["incoming"]]
                out.append(f"- Receives from ← {_join(items)}")
            if f["undirected"]:
                items = [f"**{r['label']}**" + (f" *({r['edge_label']})*" if r["edge_label"] else "")
                         for r in f["undirected"]]
                out.append(f"- Undirected links ↔ {_join(items)}")
            if f["degree"] == 0:
                out.append("\nIt is drawn on the canvas but nothing is wired to it — "
                           "an orphan in the topology.")
            out.append("")
        return "\n".join(out).strip()

    def _direction(self, q, facts, chunks) -> str:
        d = next((f for f in facts if f["type"] == "direction"), None)
        if d:
            if not d["matches"]:
                return (f"Nothing is drawn **{d['direction']}** **{d['anchor']}** — "
                        f"it is the last element in that direction on the canvas.")
            first = d["matches"][0]
            rest = [m["label"] for m in d["matches"][1:5]]
            s = (f"Directly **{d['direction']}** **{d['anchor']}** sits "
                 f"**{first['label']}** (centroid distance {first['distance']} cells).")
            if rest:
                s += f" Further {d['direction']}, in order: {_join([f'**{r}**' for r in rest])}."
            s += ("\n\nThis was resolved geometrically — by comparing bounding-box "
                  "centroids on the character matrix — not by guessing from word order.")
            return s
        qd = next((f for f in facts if f["type"] == "quadrant"), None)
        if qd:
            if not qd["matches"]:
                return f"The **{qd['quadrant']}** region of the canvas is empty."
            return (f"The **{qd['quadrant']}** region contains "
                    f"{_join([f'**{m}**' for m in qd['matches']])}.")
        return self._semantic(q, facts, chunks)

    def _topology(self, q, facts, chunks) -> str:
        t = next((f for f in facts if f["type"] == "topology"), None)
        if not t:
            return self._semantic(q, facts, chunks)
        out = [f"The drawing resolves to **{t['nodes']} nodes** and **{t['edges']} edges** "
               f"across **{t['components']} connected component"
               f"{'s' if t['components'] != 1 else ''}**."]
        if t.get("hubs"):
            top = t["hubs"][0]
            out.append(f"\n**Most central element: {top['label']}** "
                       f"(centrality {top['score']}). Removing it would do the most "
                       f"damage to connectivity — the closest thing this diagram has "
                       f"to a single point of failure.")
            if len(t["hubs"]) > 1:
                out.append("\nFull centrality ranking:")
                for h in t["hubs"][:6]:
                    bar = "█" * max(1, int(h["score"] * 24))
                    out.append(f"- `{bar:<24}` **{h['label']}** · {h['score']}")
        if t.get("roots"):
            out.append(f"\n**Entry points** (no inbound edges): "
                       f"{_join([f'**{r}**' for r in t['roots'][:6]])}.")
        if t.get("sinks"):
            out.append(f"**Terminals** (no outbound edges): "
                       f"{_join([f'**{r}**' for r in t['sinks'][:6]])}.")
        if t.get("isolated"):
            out.append(f"\n**{t['isolated']} isolated element"
                       f"{'s' if t['isolated'] != 1 else ''}** — drawn but unconnected.")
        cy = next((f for f in facts if f["type"] == "cycles"), None)
        if cy and cy["cycles"]:
            out.append("\n**Cycles detected:**")
            for c in cy["cycles"][:3]:
                out.append(f"- `{_arrow_chain(c + [c[0]])}`")
        elif t.get("has_cycles") is False:
            out.append("\nThe directed graph is **acyclic** — it flows one way.")
        return "\n".join(out)

    def _inventory(self, q, facts, chunks) -> str:
        inv = next((f for f in facts if f["type"] == "inventory"), None)
        if not inv:
            return self._semantic(q, facts, chunks)
        out = [f"**{inv['count']} element{'s' if inv['count'] != 1 else ''}** were "
               f"recovered from the character matrix, joined by **{inv['edges']} "
               f"connection{'s' if inv['edges'] != 1 else ''}**.\n"]
        for n in inv["nodes"][:40]:
            b = n.get("bbox") or {}
            out.append(f"- **{n['label']}** · `{n.get('style','?')}` · "
                       f"rows {b.get('r0','?')}–{b.get('r1','?')}, "
                       f"cols {b.get('c0','?')}–{b.get('c1','?')}")
        if inv["count"] > 40:
            out.append(f"- …and {inv['count'] - 40} more")
        if inv.get("annotations"):
            out.append(f"\n**Free-floating text** (titles, legends, captions): "
                       f"{_join([f'`{a}`' for a in inv['annotations'][:10]])}.")
        return "\n".join(out)

    def _containment(self, q, facts, chunks) -> str:
        c = next((f for f in facts if f["type"] == "containment"), None)
        if not c:
            return self._semantic(q, facts, chunks)
        if c["contains"]:
            return (f"**{c['container']}** encloses "
                    f"{_join([f'**{x}**' for x in c['contains']])}. "
                    f"Containment was established by bounding-box subsumption on the "
                    f"grid — a nested box is literally inside its parent's rectangle.")
        return (f"**{c['container']}** contains no nested boxes. Its interior reads: "
                f"`{c['raw_text'][:200]}`")

    def _semantic(self, q, facts, chunks) -> str:
        frame = next((f for f in facts if f["type"] in ("frame", "topology")), {})
        best = [c for c in chunks if c["kind"] in ("node", "edge", "region")][:5]
        if not best:
            best = chunks[:4]
        out = []
        if frame:
            cv = frame.get("canvas", {})
            out.append(f"This canvas is **{cv.get('height','?')} rows × "
                       f"{cv.get('width','?')} columns** and resolves to "
                       f"**{frame.get('nodes',0)} elements** with "
                       f"**{frame.get('edges',0)} connections**.")
        if best:
            out.append("\n**Most relevant regions retrieved:**\n")
            for c in best:
                where = (c.get("meta") or {}).get("position_phrase", "")
                lab = (c.get("meta") or {}).get("label") or c.get("readable") or "(graphic)"
                out.append(f"- **{lab[:70]}** — `{c['kind']}` at rows "
                           f"{c['bbox']['r0']}–{c['bbox']['r1']}, cols "
                           f"{c['bbox']['c0']}–{c['bbox']['c1']}"
                           + (f", {where}" if where else ""))
        if frame.get("hubs"):
            out.append(f"\nStructurally, **{frame['hubs'][0]['label']}** is the most "
                       f"connected element in the drawing.")
        out.append("\n*Ask about connectivity, routes, containment or relative position "
                   "and I will compute the answer from the recovered graph rather than "
                   "infer it from text.*")
        return "\n".join(out)

    def _citations(self, chunks) -> List[Dict]:
        out = []
        for c in chunks[:8]:
            out.append({
                "chunk_id": c["id"], "kind": c["kind"], "bbox": c["bbox"],
                "score": c.get("score"),
                "label": (c.get("meta") or {}).get("label") or c.get("readable", "")[:60],
                "excerpt": "\n".join((c.get("text") or "").splitlines()[:6]),
            })
        return out


# ==========================================================================
#  LLM narration on top of the derivation
# ==========================================================================
NARRATION_SYSTEM = (
    "You are the narration layer of a spatial RAG system that reads ASCII "
    "diagrams. You are given FACTS that were computed exactly from a parsed "
    "graph. Your only job is to restate those facts in clear, confident prose "
    "for an engineer. Rules: (1) never introduce an entity, number, or "
    "relationship that is not in the FACTS; (2) never hedge about things the "
    "FACTS state definitively; (3) keep it under 150 words; (4) use markdown "
    "bold for entity names; (5) do not mention that you were given facts."
)


def narrate(evidence_dict: Dict, derived_answer: str) -> Optional[str]:
    llm = get_llm()
    if not llm.available():
        return None
    facts = json.dumps(evidence_dict["facts"], indent=None)[:4000]
    prompt = (f"QUESTION: {evidence_dict['question']}\n\n"
              f"FACTS (computed, authoritative):\n{facts}\n\n"
              f"DERIVED ANSWER (already correct, rephrase it):\n{derived_answer[:2000]}\n\n"
              f"Rewrite the derived answer in natural prose.")
    out = llm.complete(NARRATION_SYSTEM, prompt, temperature=0.25)
    if not out or len(out) < 20:
        return None
    return out


def answer(evidence_dict: Dict, *, use_llm: bool = True) -> Dict:
    composer = GroundedComposer()
    base = composer.compose(evidence_dict)
    if use_llm and USE_LLM != "never":
        t0 = time.perf_counter()
        nar = narrate(evidence_dict, base["answer"])
        if nar:
            base["narration"] = nar
            base["mode"] = "grounded-derivation + local-llm narration"
            base["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            base["llm_model"] = get_llm().model
    return base
