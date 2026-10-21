import React from 'react'

/**
 * Observatory — the live view into the retrieval pipeline.
 *
 * Most RAG demos hand you a confident paragraph and ask you to take the
 * retrieval on faith. This panel streams every stage as it completes, with its
 * real cost in milliseconds and its real intermediate results: which intent
 * fired, what spatial prior was synthesised, what each of the four retrievers
 * returned, how fusion weighted them, and which graph operations produced the
 * grounded facts.
 */

const LABELS = {
  route: 'Intent router',
  prior: 'Spatial prior',
  dense: 'R1 · dense (semantic ⊕ spatial)',
  lexical: 'R2 · lexical BM25',
  spatial: 'R3 · spatial-only scan',
  graph: 'R4 · graph expansion',
  fuse: 'Reciprocal rank fusion',
  ground: 'Symbolic grounding',
  compose: 'Answer composition',
  canvas: 'Canvas build',
  parse: 'Structural parse',
  chunk: '2-D chunking',
  embed: 'Dual-stream embedding',
  done: 'Indexed',
  retrieve: 'Asset retrieval',
  scene: 'Scene retrieval & planning',
  style: 'Style retrieval',
  render: 'Procedural render',
  plan: 'Layout planning',
  typography: 'Typography solver',
  rasterise: 'Shape rasterisation',
  vocabulary: 'Lexicon assembly',
  flow: 'Scanline packing',
  validate: 'Validation',
}

function num(v) {
  if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(3)
  return v
}

function Row({ stage, payload, live }) {
  const ms = payload?.ms
  const name = LABELS[stage] || stage

  const kvs = []
  const push = (k, v) => {
    if (v === undefined || v === null || v === '' || (Array.isArray(v) && !v.length)) return
    kvs.push(
      <span className="kv" key={k}>
        {k} <b>{Array.isArray(v) ? v.length : num(v)}</b>
      </span>,
    )
  }

  let detail = null

  switch (stage) {
    case 'route':
      push('intent', payload.intent)
      detail = payload.entities?.length ? (
        <div className="sr-body">
          resolved:{' '}
          {payload.entities.map((e) => (
            <code key={e.id} style={{ marginRight: 6 }}>
              {e.label} · {e.confidence}
            </code>
          ))}
        </div>
      ) : (
        <div className="sr-body">no named entity in the question — pure semantic route</div>
      )
      break
    case 'prior':
      if (payload.active) {
        detail = (
          <div className="sr-body">
            {payload.mode === 'relative'
              ? `"${payload.direction}" of ${payload.anchor} → region rows ${payload.box.r0}–${payload.box.r1}, cols ${payload.box.c0}–${payload.box.c1}`
              : payload.mode === 'absolute'
                ? `${payload.quadrant} quadrant → rows ${payload.box.r0}–${payload.box.r1}, cols ${payload.box.c0}–${payload.box.c1}`
                : `anchored on ${payload.anchor}`}
          </div>
        )
      } else {
        detail = <div className="sr-body">no locative language — spatial half left unbiased</div>
      }
      break
    case 'dense':
    case 'spatial':
    case 'lexical':
      push('hits', payload.hits)
      detail = payload.top?.length ? (
        <div className="sr-body">
          {payload.top.slice(0, 3).map((t, i) => (
            <div key={i}>
              <span className={`tagk ${t.kind || 'tile'}`}>{t.kind}</span>{' '}
              {t.score}
              {t.semantic !== undefined && (
                <span style={{ color: 'var(--ink-faint)' }}>
                  {' '}· sem {t.semantic} · spa {t.spatial}
                </span>
              )}
            </div>
          ))}
        </div>
      ) : null
      break
    case 'graph':
      push('seeds', payload.seeds)
      push('expanded', payload.expanded)
      push('chunk hits', payload.chunk_hits)
      detail = payload.seeds?.length ? (
        <div className="sr-body">{payload.hops}-hop neighbourhood of {payload.seeds.join(', ')}</div>
      ) : null
      break
    case 'fuse':
      push('candidates', payload.candidates)
      push('selected', payload.selected)
      detail = payload.weights ? (
        <div className="sr-body">
          weights · {Object.entries(payload.weights).map(([k, v]) => `${k} ${v}`).join('  ')}
        </div>
      ) : null
      break
    case 'ground':
      push('facts', payload.facts)
      detail = payload.operations?.length ? (
        <div className="sr-body">
          {payload.operations.map((o, i) => (
            <div key={i}><code>{o}</code></div>
          ))}
        </div>
      ) : <div className="sr-body">no symbolic operation required</div>
      break
    case 'canvas':
      push('rows', payload.height); push('cols', payload.width); push('ink', payload.ink)
      break
    case 'parse':
      push('nodes', payload.nodes); push('edges', payload.edges)
      push('wires', payload.wire_components); push('notes', payload.annotations)
      break
    case 'chunk':
      push('total', payload.total); push('tiles', payload.tile)
      push('regions', payload.region); push('nodes', payload.node); push('edges', payload.edge)
      break
    case 'embed':
      push('vectors', payload.vectors); push('dim', payload.dim)
      detail = <div className="sr-body">{payload.backend}</div>
      break
    case 'scene': {
      push('queries', payload.queries)
      push('candidates', payload.candidates)
      push('elements', payload.picks)
      push('horizon', payload.horizon)
      detail = (
        <div className="sr-body">
          {(payload.picks || []).map((p, i) => (
            <div key={i}>
              <span className={`tagk ${p.layer === 'mid' ? 'node' : 'tile'}`}>{p.layer}</span>{' '}
              {p.name} <span style={{ color: 'var(--amber)' }}>{p.score || '—'}</span>
            </div>
          ))}
          <div style={{ marginTop: 4, opacity: 0.7 }}>
            {payload.mood?.night ? 'night palette' : 'day palette'} · horizon row{' '}
            {Math.round((payload.horizon || 0) * 100)}%
          </div>
        </div>
      )
      break
    }
    case 'style':
      detail = (
        <div className="sr-body">
          {['palette', 'frame'].map((k) =>
            payload[k]?.chosen ? (
              <div key={k}>{k} <code>{payload[k].chosen}</code> · {payload[k].score}</div>
            ) : null,
          )}
        </div>
      )
      break
    case 'render':
      push('elements', payload.elements)
      push('ink', payload.ink_cells)
      push('ink ratio', payload.ink_ratio)
      break
    case 'retrieve':
      if (payload.shape) {
        detail = (
          <div className="sr-body">
            shape <code>{payload.shape.name}</code> · {payload.shape.score}
            {payload.exemplar && (
              <div style={{ marginTop: 4 }}>
                exemplar <code>{payload.exemplar.name}</code> → strategy{' '}
                <code>{payload.strategy?.key}</code>
              </div>
            )}
            {payload.lexicon?.length > 0 && (
              <div style={{ marginTop: 4 }}>
                lexicon: {payload.lexicon.map((l) => l.name).join(' · ')}
              </div>
            )}
          </div>
        )
      } else {
        detail = (
          <div className="sr-body">
            {['layout', 'frame', 'palette', 'texture'].map((k) =>
              payload[k]?.chosen ? (
                <div key={k}>
                  {k} <code>{payload[k].chosen}</code> · {payload[k].score}
                </div>
              ) : null,
            )}
            {payload.motifs?.length > 0 && (
              <div>motifs: {payload.motifs.map((m) => `${m.name} (${m.w}×${m.h})`).join(', ')}</div>
            )}
          </div>
        )
      }
      break
    case 'typography':
      detail = (
        <div className="sr-body">
          title <code>{payload.title}</code>
          {payload.attempts?.map((a, i) => (
            <div key={i} style={{ color: a.status?.startsWith('accepted') ? 'var(--green)' : 'var(--ink-faint)' }}>
              {a.font}
              {a.lines ? ` ×${a.lines}` : ''}{' '}
              {a.measured ? `${a.measured.w}×${a.measured.h}` : ''}
              {a.slot ? ` ≤ ${a.slot.w}×${a.slot.h}` : ''} — {a.status}
            </div>
          ))}
        </div>
      )
      break
    case 'rasterise':
      push('inside', payload.inside_cells); push('coverage', payload.coverage)
      break
    case 'vocabulary':
      push('tokens', payload.tokens)
      detail = <div className="sr-body">{payload.source}<br />“{payload.sample}…”</div>
      break
    case 'flow':
      push('runs filled', payload.runs); push('of', payload.total_runs)
      push('syllables', payload.syllables)
      break
    case 'validate':
      Object.entries(payload).forEach(([k, v]) => {
        if (k === 'ms') return
        push(k.replace(/_/g, ' '), typeof v === 'boolean' ? (v ? 'yes' : 'no') : v)
      })
      break
    case 'compose':
      if (payload.gradient) push('gradient', payload.gradient)
      if (payload.overflow !== undefined) push('overflow', payload.overflow)
      if (payload.status) detail = <div className="sr-body">{payload.status}</div>
      break
    case 'done':
      push('chunks', payload.chunks); push('nodes', payload.nodes); push('edges', payload.edges)
      break
    default:
      break
  }

  return (
    <div className={`stagerow${live ? ' live' : ''}`}>
      <div className="sr-head">
        <span className="sr-name">{name}</span>
        {ms !== undefined && <span className="sr-ms">{ms} ms</span>}
      </div>
      {detail}
      {kvs.length > 0 && <div className="sr-kv">{kvs}</div>}
    </div>
  )
}

export default function Observatory({ stages, busy, wallMs, title, subtitle, error }) {
  return (
    <div className="panel observatory">
      <div className="panel-head">
        <h2>{title || 'RAG Observatory'}</h2>
        <span className="spacer" />
        <span className={`dot ${busy ? 'busy' : stages.length ? '' : 'idle'}`} />
        {wallMs !== null && wallMs !== undefined && (
          <span className="pill"><b>{wallMs}</b> ms wall</span>
        )}
      </div>
      <div className="panel-body">
        {subtitle && (
          <p style={{ margin: '0 0 12px', fontSize: 11.5, color: 'var(--ink-faint)' }}>{subtitle}</p>
        )}
        {error && <div className="err" style={{ marginBottom: 10 }}>{error}</div>}
        {!stages.length && !busy && (
          <div className="empty">
            Nothing running.<br />
            Every stage of the pipeline will appear here with its real cost.
          </div>
        )}
        <div className="stagelist">
          {stages.map((s, i) => (
            <Row key={`${s.stage}-${i}`} stage={s.stage} payload={s.payload} live={busy && i === stages.length - 1} />
          ))}
        </div>
      </div>
    </div>
  )
}
