import React from 'react'

/**
 * RagStrip — an unmissable statement of where retrieval actually happens.
 *
 * A reviewer looking at generated output has no way to tell retrieval from
 * generation unless you show them. This strip names, per section, exactly
 * which corpus is being searched, what the query is, and what the retrieved
 * records decide. The live traces elsewhere in the page then back it up with
 * real scores.
 */
export default function RagStrip({ items, corpus }) {
  return (
    <div className="ragstrip">
      <div className="rs-head">
        <span className="rs-badge">RAG</span>
        <span className="rs-title">Where retrieval is doing the work</span>
        {corpus ? <span className="rs-corpus">{corpus}</span> : null}
      </div>
      <div className="rs-grid">
        {items.map((it, i) => (
          <div className="rs-cell" key={i}>
            <div className="rs-step">{String(i + 1).padStart(2, '0')}</div>
            <div>
              <div className="rs-q">{it.query}</div>
              <div className="rs-a">{it.decides}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
