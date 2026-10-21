#!/usr/bin/env python3
"""
GLYPHMIND launcher.

    python run.py              build the UI if needed, then serve everything
                               from http://127.0.0.1:8000
    python run.py --dev        backend on :8000 + Vite dev server on :5173
                               with hot reload
    python run.py --no-build   skip the frontend build (API only)
    python run.py --port 9000  serve on a different port
    python run.py --check      run the self-test and exit

Environment flags (all optional):
    GLYPHMIND_OFFLINE=1        skip the sentence-transformers download and use
                               the built-in hashed encoder (instant boot)
    GLYPHMIND_USE_LLM=never    never attempt the optional local Ollama narration
    OLLAMA_MODEL=llama3.2      which local model to narrate with, if present
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"

C = {"c": "\033[36m", "g": "\033[32m", "y": "\033[33m", "r": "\033[31m",
     "d": "\033[2m", "b": "\033[1m", "x": "\033[0m"}


def say(msg: str, tone: str = "c") -> None:
    print(f"{C[tone]}{msg}{C['x']}", flush=True)


BANNER = r"""
   ____ _    _   _ ____  _   _ __  __ ___ _   _ ____
  / ___| |  | | | |  _ \| | | |  \/  |_ _| \ | |  _ \
 | |  _| |  | |_| | |_) | |_| | |\/| || ||  \| | | | |
 | |_| | |__|  _  |  __/|  _  | |  | || || |\  | |_| |
  \____|_____|_| |_|_|   |_| |_|_|  |_|___|_| \_|____/
        spatial RAG for text-native art
"""


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def check_python_deps() -> bool:
    missing = []
    for mod, pkg in [("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"),
                     ("numpy", "numpy"), ("networkx", "networkx"),
                     ("pyfiglet", "pyfiglet")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        say(f"  missing python packages: {' '.join(missing)}", "r")
        say(f"  install with:  pip install {' '.join(missing)}", "y")
        return False
    return True


def build_frontend() -> bool:
    if not FRONTEND.exists():
        say("  no frontend directory — serving API only", "y")
        return False
    if not have("npm"):
        say("  npm not found — serving API only (install Node to get the UI)", "y")
        return False
    if not (FRONTEND / "node_modules").exists():
        say("  installing frontend dependencies (first run only)…", "d")
        r = subprocess.run(["npm", "install", "--no-audit", "--no-fund"],
                           cwd=FRONTEND)
        if r.returncode != 0:
            say("  npm install failed", "r")
            return False
    say("  building the UI…", "d")
    r = subprocess.run(["npm", "run", "build"], cwd=FRONTEND)
    if r.returncode != 0:
        say("  build failed", "r")
        return False
    return True


def self_test() -> int:
    sys.path.insert(0, str(BACKEND))
    os.environ.setdefault("GLYPHMIND_OFFLINE", "1")
    from core.architect import compose                       # noqa: E402
    from core.concrete import generate                       # noqa: E402
    from core.generator import answer                        # noqa: E402
    from core.index import IndexManager                      # noqa: E402
    from core.retriever import HybridRetriever               # noqa: E402
    from corpus.diagrams import SEED_DIAGRAMS                # noqa: E402

    fails = 0
    say("\n[1/4] indexing the seed corpus", "c")
    m = IndexManager()
    r = HybridRetriever(m)
    docs = []
    for sd in SEED_DIAGRAMS:
        d = m.ingest(sd["name"], sd["text"], sd["kind"])
        docs.append((sd, d))
        ok = d.graph.G.number_of_nodes() > 0
        fails += 0 if ok else 1
        print(f"      {'ok ' if ok else 'FAIL'} {sd['name']:<38} "
              f"{d.graph.G.number_of_nodes():>3} nodes "
              f"{d.graph.U.number_of_edges():>3} edges "
              f"{len(d.chunks):>4} chunks "
              f"{d.stats['timings_ms']['total']:>7.1f} ms")

    say("\n[2/4] retrieval + grounded answers", "c")
    for sd, d in docs:
        for q in sd["samples"][:2]:
            t0 = time.perf_counter()
            ev = r.retrieve(d, q)
            a = answer(ev.to_dict(), use_llm=False)
            ms = (time.perf_counter() - t0) * 1000
            ok = bool(a["answer"].strip())
            fails += 0 if ok else 1
            print(f"      {'ok ' if ok else 'FAIL'} [{ev.intent:<12}] {ms:6.1f} ms  {q[:52]}")

    say("\n[3/4] ascii / ansi architect — retrieval-augmented scene synthesis", "c")
    prompts = [
        "a neon cyberpunk city skyline at night",
        "a lighthouse in a storm over the sea",
        "snowy mountains at sunrise with pine forest",
        "deep space, a ringed planet and a lone figure",
        "a rocket launch at dawn from the desert",
        "a dragon circling a burning tower",        # no recipe: fallback path
        "an espresso machine",                      # unrelated noun: fallback path
    ]
    for p in prompts:
        res = compose(p, width=100, height=32)
        v = res["validation"]
        picks = res["scene"]["picks"]
        # A scene must be rectangular, stay inside its slots, carry at least
        # three depth layers, and land in a sane ink band — too sparse reads as
        # empty, too dense reads as a solid block.
        ok = (v["rectangular"] and v["out_of_slot_writes"] == 0
              and len(v["layers_covered"]) >= 3
              and 0.18 <= v["ink_ratio"] <= 0.92
              and len(picks) >= 3)
        fails += 0 if ok else 1
        layers = "/".join(v["layers_covered"])
        print(f"      {'ok ' if ok else 'FAIL'} {p[:40]:<42} "
              f"{len(picks)} elements  ink={v['ink_ratio']:<6} "
              f"{layers:<26} {res['timings_ms']['total']:6.1f} ms")
        print(f"           └ {' · '.join(x['name'] for x in picks)}")

    say("\n[4/4] concrete poetry", "c")
    for t in ["grief for someone who is gone", "freedom and flight",
              "time running out"]:
        res = generate(t, width=64, height=28)
        ok = res["validation"]["glyphs_placed"] > 50
        fails += 0 if ok else 1
        print(f"      {'ok ' if ok else 'FAIL'} {t[:38]:<40} "
              f"shape={res['shape']['name']:<16} "
              f"fidelity={res['validation']['shape_fidelity']} "
              f"{res['timings_ms']['total']:6.1f} ms")

    if fails:
        say(f"\n{fails} check(s) failed\n", "r")
    else:
        say("\nall checks passed\n", "g")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GLYPHMIND launcher")
    ap.add_argument("--dev", action="store_true", help="hot-reload dev mode")
    ap.add_argument("--no-build", action="store_true", help="skip the UI build")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--check", action="store_true", help="run the self-test and exit")
    args = ap.parse_args()

    print(C["c"] + BANNER + C["x"])

    if not check_python_deps():
        return 1

    if args.check:
        return self_test()

    served_ui = False
    if args.dev:
        say("  dev mode — the UI will be served by Vite on :5173", "y")
    elif not args.no_build:
        if DIST.exists() and (DIST / "index.html").exists():
            say("  using the existing UI build (delete frontend/dist to rebuild)", "d")
            served_ui = True
        else:
            served_ui = build_frontend()
    else:
        served_ui = DIST.exists()

    env = dict(os.environ)
    env["GLYPHMIND_PORT"] = str(args.port)
    env["GLYPHMIND_HOST"] = args.host
    env["PYTHONUNBUFFERED"] = "1"

    vite = None
    if args.dev and have("npm"):
        if not (FRONTEND / "node_modules").exists():
            subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=FRONTEND)
        vite = subprocess.Popen(["npm", "run", "dev"], cwd=FRONTEND)

    cmd = [sys.executable, "-m", "uvicorn", "app:app",
           "--host", args.host, "--port", str(args.port)]
    if args.dev:
        cmd.append("--reload")

    url = (f"http://{args.host}:5173" if args.dev
           else f"http://{args.host}:{args.port}")
    print()
    say(f"  ▸ open  {C['b']}{url}{C['x']}{C['g']}", "g")
    say(f"  ▸ api   http://{args.host}:{args.port}/docs", "d")
    if not served_ui and not args.dev:
        say("  ▸ note: the UI was not built; the API is still fully usable", "y")
    print()

    try:
        return subprocess.call(cmd, cwd=BACKEND, env=env)
    except KeyboardInterrupt:
        return 0
    finally:
        if vite:
            vite.terminate()


if __name__ == "__main__":
    sys.exit(main())
