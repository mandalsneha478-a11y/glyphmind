import React, { useMemo, useRef, useState } from 'react'
import RagStrip from './RagStrip.jsx'
import { copyText, download, parseAnsi, streamPipeline, xterm256 } from '../lib/api.js'

/**
 * Section 2 — the Typographic ASCII / ANSI Architect.
 *
 * The same retrieval engine as the Reader, pointed the other way: instead of
 * retrieving regions out of a drawing, it retrieves *recipes* to build one.
 * This component's job is to make that visible — the prompt decomposition, the
 * per-query hits, the depth layer each winning recipe was assigned to, and the
 * typography solver's accept/reject trace.
 */

const PRESETS = [
  'a neon cyberpunk city skyline at night',
  'a lighthouse in a storm over the sea',
  'snowy mountains at sunrise with pine forest',
  'a rocket launch at dawn from the desert',
  'an ancient castle above a forest in fog',
  'a sailing ship crossing an empty ocean',
  'deep space, a ringed planet and a lone figure',
  'a dragon circling a burning tower',
]

const LAYER_ORDER = ['sky', 'far', 'mid', 'near', 'fg', 'overlay']

function AnsiArt({ ansi }) {
  const lines = useMemo(() => parseAnsi(ansi), [ansi])
  return (
    <pre className="ansi">
      {lines.map((runs, i) => (
        <div key={i}>
          {runs.map((r, j) => (
            <span key={j} style={{ color: r.color || '#7f98ad' }}>{r.text}</span>
          ))}
          {!runs.length ? ' ' : null}
        </div>
      ))}
    </pre>
  )
}

function LayerSelect({ label, value, onChange, scenes, layer }) {
  const opts = (scenes || []).filter((s) => s.layer === layer)
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value || ''} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">auto — retrieved from the prompt</option>
        {opts.map((o) => (
          <option key={o.id} value={o.id}>{o.name}</option>
        ))}
      </select>
    </label>
  )
}

function Select({ label, value, onChange, options }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value || ''} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">auto — retrieved from the prompt</option>
        {(options || []).map((o) => (
          <option key={o.id} value={o.id}>{o.name}</option>
        ))}
      </select>
    </label>
  )
}

export default function Architect({ catalog, onStages, onBusy, onWall, onError }) {
  const [prompt, setPrompt] = useState(PRESETS[0])
  const [title, setTitle] = useState('')
  const [width, setWidth] = useState(118)
  const [height, setHeight] = useState(40)
  const [mode, setMode] = useState('')
  const [res, setRes] = useState(null)
  const [busy, setBusy] = useState(false)
  const [view, setView] = useState('ansi')
  const [open, setOpen] = useState(false)
  const [ov, setOv] = useState({
    frame_id: null, palette_id: null, font_id: null,
    sky_id: null, far_id: null, subject_id: null, ground_id: null,
  })
  const [copied, setCopied] = useState(false)
  const abortRef = useRef(null)

  const go = (body) => {
    setBusy(true); onBusy?.(true); onStages?.([]); onWall?.(null); onError?.(null)
    const acc = []
    abortRef.current?.abort()
    abortRef.current = new AbortController()
    return streamPipeline('/api/architect/stream', body, {
      signal: abortRef.current.signal,
      onStage: (s) => { acc.push(s); onStages?.([...acc]) },
      onResult: (r, wall) => { setRes(r); onWall?.(wall) },
      onError: (e) => onError?.(e.message || 'pipeline error'),
    }).catch((e) => { if (e.name !== 'AbortError') onError?.(String(e)) })
      .finally(() => { setBusy(false); onBusy?.(false) })
  }

  const run = (p) => {
    const text = (p ?? prompt).trim()
    if (!text) return
    setPrompt(text); setRes(null)
    return go({ prompt: text, width, height, title: title || null,
                mode: mode || null, ...ov })
  }

  const reroll = () => go({ prompt, width, height, title: title || null,
                            mode: mode || null, ...ov,
                            seed: Math.floor(Math.random() * 1e9) })

  const v = res?.validation
  const picks = res?.scene?.picks || []
  const byLayer = LAYER_ORDER.flatMap((L) => picks.filter((p) => p.layer === L))

  return (
    <div style={{ display: 'grid', gap: 14 }}>
      <div className="hero">
        <h1>② ASCII / ANSI Architect — retrieval-augmented scene synthesis</h1>
        <p>
          Describe anything. The prompt is <strong>decomposed into separate
          queries</strong>, each searched against a knowledge base of scene recipes.
          Winning recipes are assigned to depth layers — sky, horizon, subject,
          ground, foreground, weather — and executed by <strong>procedural
          generators</strong> that synthesise detail at whatever size you ask for.
          Nothing is a stored picture; a fractal ridge is computed fresh every run.
        </p>
        <div className="steps">
          <span className="step"><b>1</b> describe a scene, or pick a preset</span>
          <span className="step"><b>2</b> compose</span>
          <span className="step"><b>3</b> read the scene graph below — every element has a score</span>
        </div>
      </div>

      <RagStrip
        corpus={`${catalog?.scenes?.length || 0} scene recipes · ${catalog?.shapes?.length || 0} shape formulas · ${catalog?.palettes?.length || 0} palettes · ${catalog?.fonts?.length || 0} typefaces`}
        items={[
          { query: 'Each prompt fragment → scene knowledge base',
            decides: 'which elements exist, and in which depth layer' },
          { query: 'Prompt → palette, frame and typeface corpora',
            decides: 'the colour ramp, border glyph set and font ranking' },
          { query: 'Unmatched subject → shape-formula corpus',
            decides: 'the fallback silhouette, so any noun still renders' },
          { query: 'Retrieved geometry → typography solver',
            decides: 'the title face, measured against an integer slot' },
        ]}
      />

      <div className="panel">
        <div className="panel-head">
          <h3>Prompt</h3>
          <span className="spacer" />
          <button className="btn ghost" onClick={() => setOpen((o) => !o)}>
            {open ? 'hide layer pinning' : 'pin layers manually'}
          </button>
        </div>
        <div className="panel-body">
          <label className="field">
            <span>what should it depict?</span>
            <input type="text" value={prompt}
                   onChange={(e) => setPrompt(e.target.value)}
                   onKeyDown={(e) => { if (e.key === 'Enter') run() }}
                   placeholder="a lighthouse in a storm over the sea" />
          </label>

          <div className="chiprow" style={{ marginBottom: 13 }}>
            {PRESETS.map((p) => (
              <button key={p} className="chip" onClick={() => run(p)}>{p}</button>
            ))}
          </div>

          <div className="grid3">
            <label className="field">
              <span>composition</span>
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="">auto</option>
                <option value="scene">scene — full-bleed art</option>
                <option value="poster">poster — big title band</option>
              </select>
            </label>
            <label className="field">
              <span>width — {width} cols</span>
              <input type="range" min="60" max="200" value={width}
                     onChange={(e) => setWidth(+e.target.value)} />
            </label>
            <label className="field">
              <span>height — {height} rows</span>
              <input type="range" min="20" max="80" value={height}
                     onChange={(e) => setHeight(+e.target.value)} />
            </label>
          </div>

          {open && (
            <div className="fadein">
              <p style={{ margin: '0 0 9px', fontSize: 11, color: 'var(--ink-faint)' }}>
                Leave these on <b>auto</b> to watch retrieval choose. Pinning one
                overrides that layer only; every other layer is still retrieved.
              </p>
              <div className="grid3">
                <LayerSelect label="sky" layer="sky" scenes={catalog?.scenes}
                             value={ov.sky_id} onChange={(x) => setOv({ ...ov, sky_id: x })} />
                <LayerSelect label="horizon" layer="far" scenes={catalog?.scenes}
                             value={ov.far_id} onChange={(x) => setOv({ ...ov, far_id: x })} />
                <LayerSelect label="subject" layer="mid" scenes={catalog?.scenes}
                             value={ov.subject_id} onChange={(x) => setOv({ ...ov, subject_id: x })} />
                <LayerSelect label="ground" layer="near" scenes={catalog?.scenes}
                             value={ov.ground_id} onChange={(x) => setOv({ ...ov, ground_id: x })} />
                <Select label="palette" options={catalog?.palettes}
                        value={ov.palette_id} onChange={(x) => setOv({ ...ov, palette_id: x })} />
                <Select label="frame" options={catalog?.frames}
                        value={ov.frame_id} onChange={(x) => setOv({ ...ov, frame_id: x })} />
                <Select label="typeface" options={catalog?.fonts}
                        value={ov.font_id} onChange={(x) => setOv({ ...ov, font_id: x })} />
                <label className="field">
                  <span>title override (poster mode)</span>
                  <input type="text" value={title} placeholder="auto from prompt"
                         onChange={(e) => setTitle(e.target.value)} />
                </label>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', gap: 9, flexWrap: 'wrap', marginTop: 4 }}>
            <button className="btn" disabled={busy || !prompt.trim()} onClick={() => run()}>
              {busy ? 'composing…' : 'compose'}
            </button>
            <button className="btn magenta" disabled={busy || !res} onClick={reroll}>
              reroll seed
            </button>
          </div>
        </div>
      </div>

      {res && (
        <div className="panel fadein">
          <div className="panel-head">
            <h3>{res.title}</h3>
            <span className="spacer" />
            <span className="pill">{res.mode}</span>
            <span className="pill">{res.mood?.night ? 'night' : 'day'}</span>
            <button className={`chip${view === 'ansi' ? ' on' : ''}`} onClick={() => setView('ansi')}>
              ansi colour
            </button>
            <button className={`chip${view === 'plain' ? ' on' : ''}`} onClick={() => setView('plain')}>
              plain text
            </button>
            <button className="btn ghost" onClick={async () => {
              setCopied(await copyText(view === 'ansi' ? res.ansi : res.plain))
              setTimeout(() => setCopied(false), 1500)
            }}>{copied ? 'copied ✓' : 'copy'}</button>
            <button className="btn ghost" onClick={() =>
              download(`glyphmind-${res.seed}.${view === 'ansi' ? 'ans' : 'txt'}`,
                       view === 'ansi' ? res.ansi : res.plain)}>download</button>
          </div>
          <div className="panel-body">
            {view === 'ansi' ? <AnsiArt ansi={res.ansi} />
                             : <pre className="poem">{res.plain}</pre>}
            <div className="statline" style={{ marginTop: 12 }}>
              <span>scene elements <b>{v.scene_elements}</b></span>
              <span>layers <b>{(v.layers_covered || []).join(' · ')}</b></span>
              <span>rectangular <b style={{ color: v.rectangular ? 'var(--green)' : 'var(--red)' }}>
                {v.rectangular ? 'yes' : 'no'}</b></span>
              <span>out-of-slot writes <b style={{ color: v.out_of_slot_writes ? 'var(--amber)' : 'var(--green)' }}>
                {v.out_of_slot_writes}</b></span>
              <span>ink ratio <b>{v.ink_ratio}</b></span>
              <span>seed <b>{res.seed}</b></span>
              <span>total <b>{res.timings_ms.total} ms</b></span>
            </div>
          </div>
        </div>
      )}

      {res && (
        <div className="panel fadein">
          <div className="panel-head">
            <h3>Scene graph — what retrieval assembled</h3>
            <span className="spacer" />
            <span className="pill">pool <b>{res.scene.candidates}</b> candidates</span>
            <span className="pill">top score <b>{res.scene.top_score}</b></span>
          </div>
          <div className="panel-body">
            <div className="scenegraph">
              {byLayer.map((p, i) => (
                <div className={`sgrow${p.score === 0 ? ' fallback' : ''}`} key={i}>
                  <span className={`sg-layer ${p.layer}`}>{p.layer}</span>
                  <div>
                    <div className="sg-name">
                      {p.name} <span className="sg-gen">· {p.gen}()</span>
                    </div>
                    <div className="sg-meta">
                      {p.role} — {p.reason}
                      {p.matched_query ? <> · matched on “{p.matched_query}”</> : null}
                    </div>
                  </div>
                  <div className="sg-score">
                    {p.score > 0 ? p.score : '—'}
                    <div className="bar" style={{ width: 70 }}>
                      <i style={{ width: `${Math.max(2, Math.min(100,
                        (p.score / Math.max(0.0001, res.scene.top_score)) * 100))}%` }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
            {res.scene.subject_fallback?.mode && (
              <div className="note">
                Subject path: <b>{res.scene.subject_fallback.mode}</b>
                {res.scene.subject_fallback.reason
                  ? ` — ${res.scene.subject_fallback.reason}`
                  : ' — the scene knowledge base had no recipe for this noun, so a '
                    + 'lower-tier corpus supplied the figure.'}
              </div>
            )}
          </div>
        </div>
      )}

      {res && (
        <div className="grid2 fadein" style={{ alignItems: 'start' }}>
          <div className="panel">
            <div className="panel-head">
              <h3>Prompt decomposition — {res.scene.queries.length} queries</h3>
            </div>
            <div className="panel-body">
              <p style={{ margin: '0 0 10px', fontSize: 11.5, color: 'var(--ink-faint)' }}>
                One embedding of a whole sentence is an average, and averages lose the
                minority concept. Each fragment is searched independently and scores
                are max-pooled per element.
              </p>
              <div className="querylist">
                {res.scene.per_query.map((q, i) => (
                  <div className="qrow" key={i}>
                    <div className="q">“{q.query}”</div>
                    <div className="h">
                      {q.hits.length
                        ? q.hits.map((h) => (
                            <span key={h.id} style={{ marginRight: 10 }}>
                              <b>{h.name}</b> <span style={{ color: 'var(--amber)' }}>{h.score}</span>
                              <span style={{ opacity: 0.6 }}> ({h.layer})</span>
                            </span>
                          ))
                        : 'no hits'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><h3>Style retrieval & typography</h3></div>
            <div className="panel-body">
              {['palette', 'frame'].map((k) => {
                const slot = res.retrieval[k]
                if (!slot) return null
                return (
                  <div key={k} style={{ marginBottom: 12 }}>
                    <div style={{ fontSize: 10.5, letterSpacing: '.13em',
                                  textTransform: 'uppercase', color: 'var(--ink-faint)',
                                  marginBottom: 4 }}>{k}</div>
                    {slot.candidates?.slice(0, 3).map((c, i) => (
                      <div key={c.id} style={{ marginBottom: 4 }}>
                        <div style={{ display: 'flex', gap: 8, fontSize: 11.5,
                                      color: i === 0 ? 'var(--ink)' : 'var(--ink-faint)' }}>
                          <span>{i === 0 ? '▸' : ' '} {c.name}</span>
                          <span style={{ marginLeft: 'auto', color: 'var(--amber)' }}>
                            {c.score.toFixed(3)}
                          </span>
                        </div>
                        <div className="bar"><i style={{ width: `${Math.max(2, Math.min(100, c.score * 100))}%` }} /></div>
                      </div>
                    ))}
                    {k === 'palette' && res.assets.palette?.colors && (
                      <div className="swatches">
                        {res.assets.palette.colors.map((n) => (
                          <i key={n} style={{ background: xterm256(n) }} />
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
              {res.typography.attempts?.length > 0 ? (
                <>
                  <div style={{ fontSize: 10.5, letterSpacing: '.13em',
                                textTransform: 'uppercase', color: 'var(--ink-faint)',
                                margin: '12px 0 5px' }}>typography solver</div>
                  {res.typography.attempts.map((a, i) => {
                    const ok = a.status?.startsWith('accepted')
                    return (
                      <div key={i} style={{
                        display: 'flex', gap: 8, fontSize: 11,
                        padding: '3px 0', borderBottom: '1px solid var(--line)',
                        color: ok ? 'var(--green)' : 'var(--ink-faint)',
                      }}>
                        <span style={{ minWidth: 88 }}>{a.font}{a.lines ? ` ×${a.lines}` : ''}</span>
                        <span>{a.measured ? `${a.measured.w}×${a.measured.h}` : '—'}
                          {a.slot ? ` ≤ ${a.slot.w}×${a.slot.h}` : ''}</span>
                        <span style={{ marginLeft: 'auto' }}>{ok ? '✓' : '✕'}</span>
                      </div>
                    )
                  })}
                </>
              ) : (
                <div className="note">
                  Scene mode sets the title as a cleared plate rather than a figlet
                  band — switch composition to <b>poster</b> to watch the typography
                  solver measure and reject candidate typefaces.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
