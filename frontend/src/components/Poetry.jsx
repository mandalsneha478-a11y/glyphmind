import React, { useRef, useState } from 'react'
import RagStrip from './RagStrip.jsx'
import { copyText, download, streamPipeline } from '../lib/api.js'

/**
 * Section 3 — Interactive Generative Concrete Poetry.
 *
 * A theme is embedded and matched against structural formulas (implicit
 * functions, not pictures), against calligram exemplars that supply a *flow
 * strategy*, and against lexical clusters that supply the vocabulary. The text
 * is then packed into the rasterised silhouette scanline by scanline, with
 * syllable weight tracking the local thickness of the figure.
 */

const THEMES = [
  'grief for someone who is not coming back',
  'freedom and flight',
  'the sea, and everything it takes',
  'falling in love in a city of strangers',
  'time running out',
  'a forest remembering every season',
]

export default function Poetry({ catalog, onStages, onBusy, onWall, onError }) {
  const [theme, setTheme] = useState(THEMES[0])
  const [poem, setPoem] = useState('')
  const [shapeId, setShapeId] = useState(null)
  const [strategyId, setStrategyId] = useState(null)
  const [width, setWidth] = useState(74)
  const [height, setHeight] = useState(34)
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [view, setView] = useState('ascii')
  const [usePoem, setUsePoem] = useState(false)
  const [copied, setCopied] = useState(false)
  const abortRef = useRef(null)

  const shapes = catalog?.shapes || []
  const strategies = catalog?.strategies || []

  const run = async (t) => {
    const text = (t ?? theme).trim()
    if (!text) return
    setTheme(text)
    setBusy(true); onBusy?.(true); onStages?.([]); onWall?.(null); onError?.(null); setRes(null)
    const acc = []
    try {
      abortRef.current?.abort()
      abortRef.current = new AbortController()
      await streamPipeline('/api/poetry/stream',
        { theme: text, poem: usePoem && poem.trim() ? poem : null,
          shape_id: shapeId, strategy_id: strategyId, width, height },
        {
          signal: abortRef.current.signal,
          onStage: (s) => { acc.push(s); onStages?.([...acc]) },
          onResult: (r, wall) => { setRes(r); onWall?.(wall) },
          onError: (e) => onError?.(e.message || 'pipeline error'),
        })
    } catch (e) {
      if (e.name !== 'AbortError') onError?.(String(e))
    } finally {
      setBusy(false); onBusy?.(false)
    }
  }

  const reroll = () => {
    setBusy(true); onBusy?.(true); onStages?.([]); onError?.(null)
    const acc = []
    streamPipeline('/api/poetry/stream',
      { theme, poem: usePoem && poem.trim() ? poem : null, shape_id: shapeId,
        strategy_id: strategyId, width, height, seed: Math.floor(Math.random() * 1e9) },
      {
        onStage: (s) => { acc.push(s); onStages?.([...acc]) },
        onResult: (r, wall) => { setRes(r); onWall?.(wall) },
        onError: (e) => onError?.(e.message || 'pipeline error'),
      }).finally(() => { setBusy(false); onBusy?.(false) })
  }

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div className="hero">
        <h1>③ Concrete Poetry — text packed into a retrieved silhouette</h1>
        <p>
          Give it an emotion, a theme, or your own poem. It retrieves a{' '}
          <strong>structural formula</strong> for the figure, a{' '}
          <strong>flow strategy</strong> from a real calligram (Apollinaire's falling rain,
          Herbert's tapering wings, Mallarmé's scattered field), and{' '}
          <strong>lexical clusters</strong> for the vocabulary — then solves the packing so
          heavy, polysyllabic words land where the figure is broad and the line thins to
          monosyllables where the figure narrows.
        </p>
        <div className="steps">
          <span className="step"><b>1</b> type a theme — or paste your own poem</span>
          <span className="step"><b>2</b> let retrieval pick the shape, or pin one</span>
          <span className="step"><b>3</b> shape it, then view as text or SVG</span>
        </div>
      </div>

      <RagStrip
        corpus={`${catalog?.shapes?.length || 0} shape formulas · ${catalog?.exemplars?.length || 0} calligram exemplars · ${catalog?.lexicon?.length || 0} lexical clusters`}
        items={[
          { query: 'Theme → shape-formula corpus',
            decides: 'the silhouette the text will be packed into' },
          { query: 'Theme → calligram exemplars',
            decides: 'the flow strategy — Apollinaire, Herbert, Mallarmé' },
          { query: 'Theme → lexical clusters',
            decides: 'the vocabulary the poem is allowed to draw on' },
          { query: 'Retrieved strategy → scanline solver',
            decides: 'density, justification and syllable weighting' },
        ]}
      />

      <div className="grid2" style={{ alignItems: 'start' }}>
        <div className="panel">
          <div className="panel-head"><h3>Source</h3></div>
          <div className="panel-body">
            <label className="field">
              <span>theme or emotion</span>
              <input type="text" value={theme} onChange={(e) => setTheme(e.target.value)}
                     onKeyDown={(e) => { if (e.key === 'Enter') run() }}
                     placeholder="grief for someone who is not coming back" />
            </label>
            <div className="chiprow" style={{ marginBottom: 13 }}>
              {THEMES.map((t) => (
                <button key={t} className="chip" onClick={() => run(t)}>{t.slice(0, 34)}</button>
              ))}
            </div>

            <div className="chiprow" style={{ marginBottom: 10 }}>
              <button className={`chip${usePoem ? ' on' : ''}`} onClick={() => setUsePoem((v) => !v)}>
                {usePoem ? '✓ using my own words' : 'use my own poem instead'}
              </button>
            </div>
            {usePoem && (
              <label className="field fadein">
                <span>your poem — its words will be the only vocabulary</span>
                <textarea rows={6} value={poem} spellCheck={false}
                          onChange={(e) => setPoem(e.target.value)}
                          placeholder={'Paste any verse. Line breaks are discarded —\nthe silhouette decides where the lines fall.'} />
              </label>
            )}

            <div className="grid2">
              <label className="field">
                <span>width — {width} cols</span>
                <input type="range" min="30" max="140" value={width}
                       onChange={(e) => setWidth(+e.target.value)} />
              </label>
              <label className="field">
                <span>height — {height} rows</span>
                <input type="range" min="14" max="60" value={height}
                       onChange={(e) => setHeight(+e.target.value)} />
              </label>
            </div>

            <label className="field">
              <span>flow strategy</span>
              <select value={strategyId || ''} onChange={(e) => setStrategyId(e.target.value || null)}>
                <option value="">auto (retrieved from a matching calligram)</option>
                {strategies.map((s) => (
                  <option key={s.key} value={s.key}>{s.label}</option>
                ))}
              </select>
            </label>

            <div style={{ display: 'flex', gap: 9, flexWrap: 'wrap' }}>
              <button className="btn" disabled={busy || !theme.trim()} onClick={() => run()}>
                {busy ? 'shaping…' : 'shape the poem'}
              </button>
              <button className="btn magenta" disabled={busy || !res} onClick={reroll}>
                reroll
              </button>
            </div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h3>Shape</h3>
            <span className="spacer" />
            <button className={`chip${shapeId === null ? ' on' : ''}`} onClick={() => setShapeId(null)}>
              auto · let retrieval choose
            </button>
          </div>
          <div className="panel-body">
            <div className="shapegrid">
              {shapes.map((s) => (
                <button key={s.id} className={`shapebtn${shapeId === s.id ? ' on' : ''}`}
                        onClick={() => setShapeId(shapeId === s.id ? null : s.id)}
                        title={s.tags}>
                  <svg width="42" height="42" viewBox="0 0 100 100">
                    <path d={s.svg} fill="none" stroke="currentColor" strokeWidth="3.2"
                          strokeLinejoin="round" />
                  </svg>
                  <span>{s.name}</span>
                </button>
              ))}
              {!shapes.length && <div className="empty">loading shape formulas…</div>}
            </div>
          </div>
        </div>
      </div>

      {res && (
        <div className="panel fadein">
          <div className="panel-head">
            <h3>{res.shape.name} — “{res.theme}”</h3>
            <span className="spacer" />
            <button className={`chip${view === 'ascii' ? ' on' : ''}`} onClick={() => setView('ascii')}>
              character grid
            </button>
            <button className={`chip${view === 'svg' ? ' on' : ''}`} onClick={() => setView('svg')}>
              svg typography
            </button>
            <button className="btn ghost" onClick={async () => {
              setCopied(await copyText(view === 'svg' ? res.svg : res.ascii))
              setTimeout(() => setCopied(false), 1500)
            }}>{copied ? 'copied ✓' : 'copy'}</button>
            <button className="btn ghost" onClick={() =>
              download(`glyphmind-poem-${res.seed}.${view === 'svg' ? 'svg' : 'txt'}`,
                       view === 'svg' ? res.svg : res.ascii,
                       view === 'svg' ? 'image/svg+xml' : 'text/plain')}>download</button>
          </div>
          <div className="panel-body">
            {view === 'ascii' ? (
              <pre className="poem">{res.ascii}</pre>
            ) : (
              <div style={{ background: '#04070b', border: '1px solid var(--line)',
                            borderRadius: 9, padding: 14, overflow: 'auto', maxHeight: '62vh' }}
                   dangerouslySetInnerHTML={{ __html: res.svg }} />
            )}
            <div className="statline" style={{ marginTop: 12 }}>
              <span>shape fidelity <b>{res.validation.shape_fidelity}</b></span>
              <span>glyphs placed <b>{res.validation.glyphs_placed}</b></span>
              <span>mask cells <b>{res.validation.mask_cells}</b></span>
              <span>runs filled <b>{res.flow.runs}/{res.flow.total_runs}</b></span>
              <span>syllables <b>{res.flow.syllables}</b></span>
              <span>total <b>{res.timings_ms.total} ms</b></span>
            </div>
          </div>
        </div>
      )}

      {res?.retrieval?.exemplar && (
        <div className="panel fadein">
          <div className="panel-head"><h3>Why it looks like this</h3></div>
          <div className="panel-body">
            <div className="grid2">
              <div>
                <div style={{ fontSize: 10.5, letterSpacing: '.13em', textTransform: 'uppercase',
                              color: 'var(--ink-faint)', marginBottom: 5 }}>
                  retrieved exemplar
                </div>
                <div style={{ fontSize: 13, color: 'var(--cyan)', marginBottom: 4 }}>
                  {res.retrieval.exemplar.name}
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--ink-dim)' }}>
                  {res.retrieval.exemplar.note}
                </div>
                <div className="note">
                  Flow strategy applied: <b>{res.retrieval.strategy.label}</b> — density{' '}
                  {res.retrieval.strategy.density}, justification {res.retrieval.strategy.justify}.
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10.5, letterSpacing: '.13em', textTransform: 'uppercase',
                              color: 'var(--ink-faint)', marginBottom: 5 }}>
                  shape candidates
                </div>
                {res.retrieval.shape.candidates?.map((c, i) => (
                  <div key={c.id} style={{ marginBottom: 5 }}>
                    <div style={{ display: 'flex', fontSize: 11.5,
                                  color: i === 0 ? 'var(--ink)' : 'var(--ink-faint)' }}>
                      <span>{i === 0 ? '▸' : ' '} {c.name}</span>
                      <span style={{ marginLeft: 'auto', color: 'var(--amber)' }}>
                        {c.score.toFixed(3)}
                      </span>
                    </div>
                    <div className="bar"><i style={{ width: `${Math.max(2, Math.min(100, c.score * 100))}%` }} /></div>
                  </div>
                ))}
                <div style={{ fontSize: 10.5, letterSpacing: '.13em', textTransform: 'uppercase',
                              color: 'var(--ink-faint)', margin: '12px 0 5px' }}>
                  lexical clusters in play
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--ink-dim)' }}>
                  {res.vocabulary.clusters.map((c) => c.name).join(' · ')}
                  <div style={{ color: 'var(--ink-faint)', marginTop: 3 }}>
                    source: {res.vocabulary.source} · {res.vocabulary.tokens} tokens
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
