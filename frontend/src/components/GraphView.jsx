import React, { useMemo } from 'react'

/**
 * GraphView — the recovered topology, drawn where it actually lives.
 *
 * No force-directed layout here, and that is the point: the parser recovered
 * each node's real bounding box on the character grid, so the graph is laid
 * out at its true coordinates. The SVG is a faithful vector tracing of the
 * ASCII drawing, which makes it immediately obvious whether the structural
 * parse was correct.
 */
export default function GraphView({ graph, canvas, highlight = [], onPick }) {
  const { nodes, edges, W, H, cw, chh } = useMemo(() => {
    const cw = 9
    const chh = 17
    const ns = (graph?.nodes || []).filter((n) => n.bbox)
    const W = Math.max(320, (canvas?.width || 80) * cw)
    const H = Math.max(200, (canvas?.height || 30) * chh)
    const seen = new Set()
    const es = (graph?.edges || []).filter((e) => {
      if (e.alias) return false
      const k = [e.source, e.target].sort().join('|') + (e.label || '')
      if (seen.has(k)) return false
      seen.add(k)
      return true
    })
    return { nodes: ns, edges: es, W, H, cw, chh }
  }, [graph, canvas])

  const pos = useMemo(() => {
    const m = {}
    nodes.forEach((n) => {
      const b = n.bbox
      m[n.id] = {
        x: b.c0 * cw,
        y: b.r0 * chh,
        w: Math.max(28, (b.c1 - b.c0) * cw),
        h: Math.max(18, (b.r1 - b.r0) * chh),
        cx: ((b.c0 + b.c1) / 2) * cw,
        cy: ((b.r0 + b.r1) / 2) * chh,
        label: n.label,
      }
    })
    return m
  }, [nodes, cw, chh])

  const hot = new Set(highlight)

  if (!nodes.length) {
    return <div className="empty">No structural nodes were recovered from this drawing.</div>
  }

  return (
    <div className="graphwrap">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        <defs>
          <marker id="gm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
                  markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(120,200,240,0.65)" />
          </marker>
          <marker id="gm-arrow-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8"
                  markerHeight="8" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff5fd2" />
          </marker>
        </defs>

        {edges.map((e, i) => {
          const a = pos[e.source]
          const b = pos[e.target]
          if (!a || !b) return null
          const on = hot.has(e.source) && hot.has(e.target)
          const mx = (a.cx + b.cx) / 2
          const my = (a.cy + b.cy) / 2
          const d = `M ${a.cx} ${a.cy} Q ${mx} ${my} ${b.cx} ${b.cy}`
          return (
            <g key={`e-${i}`}>
              <path
                className={`gedge${on ? ' on' : ''}`}
                d={d}
                markerEnd={e.directed ? (on ? 'url(#gm-arrow-on)' : 'url(#gm-arrow)') : undefined}
              />
              {e.label ? (
                <text className="gelabel" x={mx} y={my - 3} textAnchor="middle">
                  {e.label.slice(0, 18)}
                </text>
              ) : null}
            </g>
          )
        })}

        {nodes.map((n) => {
          const p = pos[n.id]
          if (!p) return null
          const on = hot.has(n.id)
          return (
            <g
              key={n.id}
              className={`gnode${on ? ' on' : ''}`}
              onClick={() => onPick?.(n)}
              style={{ cursor: onPick ? 'pointer' : 'default' }}
            >
              <rect x={p.x} y={p.y} width={p.w} height={p.h} rx="4" />
              <text x={p.cx} y={p.cy + 3.5} textAnchor="middle">
                {String(p.label || '').slice(0, Math.max(4, Math.floor(p.w / 6)))}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
