import React, { useEffect, useState } from 'react'
import Architect from './components/Architect.jsx'
import Observatory from './components/Observatory.jsx'
import Poetry from './components/Poetry.jsx'
import Reader from './components/Reader.jsx'
import { getJSON } from './lib/api.js'

const TABS = [
  { key: 'reader', no: '01', name: 'Blueprint Reader',
    sub: 'Ask an ASCII diagram questions it can prove' },
  { key: 'architect', no: '02', name: 'ASCII / ANSI Architect',
    sub: 'Compose art from retrieved geometry' },
  { key: 'poetry', no: '03', name: 'Concrete Poetry',
    sub: 'Pack verse into a retrieved silhouette' },
]

const OBS_SUB = {
  reader: 'Four retrievers run against every question. Each stage below is real, timed work — not a progress animation.',
  architect: 'Retrieve geometry → plan slots → measure typefaces → compose → validate.',
  poetry: 'Retrieve formula + strategy + lexicon → rasterise → pack scanlines → emit.',
}

export default function App() {
  const [tab, setTab] = useState('reader')
  const [health, setHealth] = useState(null)
  const [catalog, setCatalog] = useState(null)
  const [stages, setStages] = useState([])
  const [busy, setBusy] = useState(false)
  const [wall, setWall] = useState(null)
  const [error, setError] = useState(null)

  // Poll health until the seed corpus has finished indexing, then fetch catalog.
  useEffect(() => {
    let alive = true
    let timer = null
    const tick = async () => {
      try {
        const h = await getJSON('/api/health')
        if (!alive) return
        setHealth(h)
        if (h.ready) {
          const c = await getJSON('/api/catalog')
          if (alive) setCatalog(c)
          return
        }
      } catch (e) {
        if (alive) setHealth({ ok: false, ready: false, error: String(e) })
      }
      timer = setTimeout(tick, 700)
    }
    tick()
    return () => { alive = false; if (timer) clearTimeout(timer) }
  }, [])

  const reset = () => { setStages([]); setWall(null); setError(null) }

  const switchTab = (k) => { setTab(k); reset() }

  const bridge = {
    catalog,
    onStages: setStages,
    onBusy: setBusy,
    onWall: setWall,
    onError: setError,
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">GLYPHMIND</span>
          <span className="brand-sub">spatial RAG for text-native art</span>
        </div>
        <div className="pillrow">
          <span className="pill">
            <span className={`dot ${health?.ready ? '' : 'busy'}`} />
            {health?.ready ? 'ready' : 'indexing corpus…'}
          </span>
          {health?.ready && (
            <>
              <span className="pill">docs <b>{health.documents}</b></span>
              <span className="pill">vectors <b>{health.vectors}</b></span>
              <span className="pill">
                dim <b>{health.dims.semantic}</b>+<b>{health.dims.spatial}</b>
              </span>
              <span className="pill" title={health.encoder}>
                encoder <b>{health.encoder.split(' ')[0].split('/').pop()}</b>
              </span>
              <span className={`pill ${health.llm?.available ? 'ok' : ''}`}>
                llm <b>{health.llm?.available ? health.llm.model : 'off · grounded only'}</b>
              </span>
              <span className="pill">boot <b>{health.boot_ms} ms</b></span>
            </>
          )}
        </div>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.key} className={`tab${tab === t.key ? ' active' : ''}`}
                  onClick={() => switchTab(t.key)}>
            <span className="tab-no">{t.no}</span>{t.name}
            <span className="tab-sub">{t.sub}</span>
          </button>
        ))}
      </nav>

      <main className="workspace">
        <section style={{ minWidth: 0 }}>
          {!health?.ready ? (
            <div className="panel">
              <div className="panel-body">
                <div className="empty">
                  Booting GLYPHMIND — parsing the seed corpus, building the 2-D chunk index,
                  and embedding both streams.<br />
                  This takes a second or two on first run.
                </div>
              </div>
            </div>
          ) : (
            <>
              {tab === 'reader' && <Reader {...bridge} />}
              {tab === 'architect' && <Architect {...bridge} />}
              {tab === 'poetry' && <Poetry {...bridge} />}
            </>
          )}
        </section>

        <aside style={{ minWidth: 0 }}>
          <Observatory
            stages={stages}
            busy={busy}
            wallMs={wall}
            error={error}
            title={tab === 'reader' ? 'RAG Observatory' : 'Pipeline Observatory'}
            subtitle={OBS_SUB[tab]}
          />
        </aside>
      </main>
    </div>
  )
}
