import React, { useEffect, useMemo, useRef, useState } from 'react'
import AsciiStage from './AsciiStage.jsx'
import RagStrip from './RagStrip.jsx'
import GraphView from './GraphView.jsx'
import Markdown from './Markdown.jsx'
import { getJSON, streamPipeline } from '../lib/api.js'

/**
 * Section 1 — ASCII & Structural Blueprint Reader.
 *
 * Index a text drawing, then interrogate it. Answers about connectivity,
 * routing, containment and relative position are *computed* from the recovered
 * graph and the character grid, then cited back to the exact rectangles of the
 * drawing they came from.
 */
export default function Reader({ catalog, onStages, onBusy, onWall, onError }) {
  const [docs, setDocs] = useState([])
  const [docId, setDocId] = useState(null)
  const [doc, setDoc] = useState(null)
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [indexing, setIndexing] = useState(false)
  const [focus, setFocus] = useState(null)
  const [view, setView] = useState('ascii')
  const [showPaste, setShowPaste] = useState(false)
  const [pasteName, setPasteName] = useState('My drawing')
  const [pasteText, setPasteText] = useState('')
  const abortRef = useRef(null)

  const seeds = catalog?.diagrams || []

  useEffect(() => {
    getJSON('/api/documents').then((d) => {
      setDocs(d.documents)
      if (!docId && d.documents.length) setDocId(d.documents[0].id)
    }).catch((e) => onError?.(String(e)))
  }, [catalog])

  useEffect(() => {
    if (!docId) return
    setResult(null); setFocus(null)
    getJSON(`/api/documents/${docId}`).then(setDoc).catch((e) => onError?.(String(e)))
  }, [docId])

  const samples = useMemo(() => {
    const seed = seeds.find((s) => s.id === docId)
    if (seed?.samples?.length) return seed.samples
    return ['What connects to each other?', 'List every element.',
            'Which element is most central?', 'Is there a cycle?']
  }, [seeds, docId])

  const ask = async (q) => {
    const text = (q ?? question).trim()
    if (!text || !docId) return
    setQuestion(text)
    setBusy(true); onBusy?.(true); onStages?.([]); onWall?.(null); onError?.(null)
    setResult(null); setFocus(null)
    const acc = []
    try {
      abortRef.current?.abort()
      abortRef.current = new AbortController()
      await streamPipeline('/api/query/stream',
        { doc_id: docId, question: text, use_llm: true, top_k: 8 },
        {
          signal: abortRef.current.signal,
          onStage: (s) => { acc.push(s); onStages?.([...acc]) },
          onResult: (r, wall) => { setResult(r); onWall?.(wall) },
          onError: (e) => onError?.(e.message || 'pipeline error'),
        })
    } catch (e) {
      if (e.name !== 'AbortError') onError?.(String(e))
    } finally {
      setBusy(false); onBusy?.(false)
    }
  }

  const indexPasted = async () => {
    if (!pasteText.trim()) return
    setIndexing(true); onBusy?.(true); onStages?.([]); onWall?.(null); onError?.(null)
    const acc = []
    try {
      await streamPipeline('/api/ingest/stream',
        { name: pasteName || 'My drawing', text: pasteText, kind: 'custom' },
        {
          onStage: (s) => { acc.push(s); onStages?.([...acc]) },
          onResult: (r, wall) => {
            onWall?.(wall)
            getJSON('/api/documents').then((d) => setDocs(d.documents))
            setDocId(r.meta.id)
            setShowPaste(false)
          },
          onError: (e) => onError?.(e.message || 'index error'),
        })
    } catch (e) {
      onError?.(String(e))
    } finally {
      setIndexing(false); onBusy?.(false)
    }
  }

  // Overlay boxes: grounded highlights + the retrieved chunk rectangles.
  const boxes = useMemo(() => {
    if (!result) return []
    const out = []
    ;(result.highlights || []).forEach((h, i) =>
      out.push({ id: `hl-${i}`, bbox: h.bbox, role: h.role, label: h.label }))
    ;(result.chunks || []).forEach((c) =>
      out.push({ id: c.id, bbox: c.bbox, role: 'chunk',
                 label: c.kind === 'node' || c.kind === 'edge' ? '' : '' }))
    return out
  }, [result])

  const highlightIds = useMemo(() => {
    const ids = new Set()
    ;(result?.entities || []).forEach((e) => ids.add(e.id))
    ;(result?.facts || []).forEach((f) => (f.node_ids || []).forEach((n) => ids.add(n)))
    return [...ids]
  }, [result])

  const ans = result?.answer

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div className="hero">
        <h1>① Blueprint Reader — retrieval that never flattens the page</h1>
        <p>
          Pick a drawing (or paste your own), then ask it anything. The canvas is chunked in
          two dimensions, parsed into a real graph, and searched by four retrievers at once —
          so questions about <strong>what connects to what</strong>, <strong>what route
          exists</strong>, <strong>what sits below something</strong> and <strong>what is
          inside something</strong> are answered by computation, then cited back to the exact
          rectangles of the picture.
        </p>
        <div className="steps">
          <span className="step"><b>1</b> choose a drawing</span>
          <span className="step"><b>2</b> click a sample question or type one</span>
          <span className="step"><b>3</b> watch the pipeline on the right</span>
          <span className="step"><b>4</b> hover a retrieved chunk to light up its region</span>
        </div>
      </div>

      <RagStrip
        corpus={`${docs.length} indexed drawings · ${doc?.meta?.chunks || 0} spatial chunks in the active document`}
        items={[
          { query: 'Question → dense index of 2-D chunks',
            decides: 'which rectangles of the drawing are semantically relevant' },
          { query: 'Question → BM25 over chunk captions',
            decides: 'exact identifiers the dense pass smooths away' },
          { query: 'Locative phrase → synthesised spatial prior',
            decides: 'retrieval by position on the grid, not by phrasing' },
          { query: 'Resolved entities → k-hop graph expansion',
            decides: 'relational context no text index can supply' },
        ]}
      />

      <div className="grid2" style={{ alignItems: 'start' }}>
        <div className="panel">
          <div className="panel-head">
            <h3>Corpus</h3>
            <span className="spacer" />
            <button className="btn ghost" onClick={() => setShowPaste((v) => !v)}>
              {showPaste ? 'close' : '+ index my own'}
            </button>
          </div>
          <div className="panel-body">
            {showPaste && (
              <div className="fadein" style={{ marginBottom: 14 }}>
                <label className="field">
                  <span>name</span>
                  <input type="text" value={pasteName} onChange={(e) => setPasteName(e.target.value)} />
                </label>
                <label className="field">
                  <span>paste any ASCII diagram, topology, floor plan or box drawing</span>
                  <textarea
                    rows={9}
                    spellCheck={false}
                    value={pasteText}
                    onChange={(e) => setPasteText(e.target.value)}
                    placeholder={'+--------+      +--------+\n|  web   +----->+  api   |\n+--------+      +---+----+\n                    |\n                    v\n                +---+----+\n                |   db   |\n                +--------+'}
                  />
                </label>
                <button className="btn" disabled={indexing || !pasteText.trim()} onClick={indexPasted}>
                  {indexing ? 'indexing…' : 'index this drawing'}
                </button>
              </div>
            )}

            <div className="doccards">
              {docs.map((d) => {
                const seed = seeds.find((s) => s.id === d.id)
                return (
                  <button
                    key={d.id}
                    className={`doccard${d.id === docId ? ' on' : ''}`}
                    onClick={() => setDocId(d.id)}
                  >
                    <div className="dc-name">{d.name}</div>
                    {seed?.blurb && <div className="dc-blurb">{seed.blurb}</div>}
                    <div className="dc-meta">
                      {d.nodes} nodes · {d.edges} edges · {d.chunks} chunks · {d.height}×{d.width}
                    </div>
                  </button>
                )
              })}
              {!docs.length && <div className="empty">indexing seed corpus…</div>}
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3>Ask the drawing</h3>
            <span className="spacer" />
            {doc && (
              <span className="pill">
                <b>{doc.summary?.nodes || 0}</b> nodes · <b>{doc.summary?.edges || 0}</b> edges
              </span>
            )}
          </div>
          <div className="panel-body">
            <label className="field">
              <span>your question</span>
              <input
                type="text"
                value={question}
                placeholder="e.g. is there a path from the internet to the audit archive?"
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') ask() }}
              />
            </label>
            <div className="chiprow" style={{ marginBottom: 12 }}>
              {samples.map((s) => (
                <button key={s} className="chip" onClick={() => ask(s)}>{s}</button>
              ))}
            </div>
            <button className="btn" disabled={busy || !question.trim()} onClick={() => ask()}>
              {busy ? 'retrieving…' : 'run retrieval'}
            </button>
            {doc?.summary && (
              <div className="statline" style={{ marginTop: 13 }}>
                <span>components <b>{doc.summary.components}</b></span>
                <span>cycles <b>{doc.summary.has_cycles ? 'yes' : 'no'}</b></span>
                {doc.summary.hubs?.[0] && <span>hub <b>{doc.summary.hubs[0].label}</b></span>}
                <span>chunks <b>{doc.meta?.chunks}</b></span>
              </div>
            )}
          </div>
        </div>
      </div>

      {ans && (
        <div className="panel fadein">
          <div className="panel-head">
            <h3>Answer</h3>
            <span className="spacer" />
            <span className="badge-mode">{ans.mode}</span>
            {result.intent && <span className="pill">intent <b>{result.intent}</b></span>}
          </div>
          <div className="panel-body">
            <Markdown text={ans.answer} />
            {ans.narration && (
              <div className="note">
                <b style={{ color: 'var(--magenta)' }}>local LLM narration ({ans.llm_model})</b>
                <div style={{ marginTop: 5 }}><Markdown text={ans.narration} /></div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="panel">
        <div className="panel-head">
          <h3>{doc?.meta?.name || 'Canvas'}</h3>
          <span className="spacer" />
          <button className={`chip${view === 'ascii' ? ' on' : ''}`} onClick={() => setView('ascii')}>
            character matrix
          </button>
          <button className={`chip${view === 'graph' ? ' on' : ''}`} onClick={() => setView('graph')}>
            recovered graph
          </button>
        </div>
        <div className="panel-body">
          {view === 'ascii' ? (
            <AsciiStage
              text={doc?.raw || ''}
              boxes={boxes}
              focusId={focus}
              scanning={busy}
            />
          ) : (
            <GraphView
              graph={doc?.graph}
              canvas={{ width: doc?.meta?.width, height: doc?.meta?.height }}
              highlight={highlightIds}
            />
          )}
          {result && (
            <p style={{ fontSize: 11, color: 'var(--ink-faint)', marginTop: 10, marginBottom: 0 }}>
              magenta = resolved entity · green = neighbour / match · amber = waypoint or hub ·
              dashed cyan = retrieved chunk rectangle
            </p>
          )}
        </div>
      </div>

      {result?.chunks?.length > 0 && (
        <div className="panel fadein">
          <div className="panel-head">
            <h3>Retrieved chunks — {result.chunks.length} of {result.retriever_stats.candidates} candidates</h3>
            <span className="spacer" />
            <span className="pill">
              encoder <b>{result.retriever_stats.encoder}</b>
            </span>
            <span className="pill">
              fused dim <b>{result.retriever_stats.dims.fused}</b>
            </span>
          </div>
          <div className="panel-body">
            <div className="hitlist">
              {result.chunks.map((c) => (
                <div
                  key={c.id}
                  className={`hit${focus === c.id ? ' on' : ''}`}
                  onMouseEnter={() => { setFocus(c.id); setView('ascii') }}
                  onMouseLeave={() => setFocus(null)}
                  onClick={() => setFocus(focus === c.id ? null : c.id)}
                >
                  <div className="hit-head">
                    <span className={`tagk ${c.kind}`}>{c.kind}</span>
                    {c.graph_hit && <span className="tagk edge">graph</span>}
                    <span className="hit-score">rrf {c.score}</span>
                  </div>
                  <div className="hit-label">
                    {c.meta?.label || c.readable || '(pure graphic region)'}
                  </div>
                  <div className="hit-geo">
                    rows {c.bbox.r0}–{c.bbox.r1} · cols {c.bbox.c0}–{c.bbox.c1} ·
                    {' '}{c.bbox.h}×{c.bbox.w} · density {c.density}
                    {c.meta?.position_phrase ? ` · ${c.meta.position_phrase}` : ''}
                  </div>
                  <div className="hit-streams">
                    {c.semantic !== null && c.semantic !== undefined && (
                      <span>semantic <b>{c.semantic}</b></span>
                    )}
                    {c.spatial !== null && c.spatial !== undefined && (
                      <span>spatial <b>{c.spatial}</b></span>
                    )}
                    {c.lexical !== null && c.lexical !== undefined && (
                      <span>bm25 <b>{c.lexical}</b></span>
                    )}
                    {Object.keys(c.ranks || {}).length > 0 && (
                      <span>ranks <b>{Object.entries(c.ranks).map(([k, v]) => `${k[0]}${v}`).join(' ')}</b></span>
                    )}
                  </div>
                  {focus === c.id && c.text && (
                    <div className="hit-excerpt">{c.text}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
