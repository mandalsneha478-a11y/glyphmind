import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'

/**
 * AsciiStage — renders a character matrix and draws real geometry on top of it.
 *
 * This component is the visual proof of the whole thesis. A retrieved chunk is
 * not a paragraph; it is a rectangle of the drawing. So we measure the exact
 * cell metrics of the rendered monospace text and place absolutely-positioned
 * boxes at (r0,c0)→(r1,c1). When you hover a retrieval hit in the Observatory,
 * the region it came from lights up *in the picture*.
 */
export default function AsciiStage({
  text,
  boxes = [],          // [{bbox:{r0,c0,r1,c1}, kind, label, role}]
  focusId = null,
  scanning = false,
  fontSize = 12.5,
}) {
  const preRef = useRef(null)
  const probeRef = useRef(null)
  const [cell, setCell] = useState({ w: 7.5, h: 16.5 })

  // Measure one character box from a hidden probe so overlays land exactly.
  useLayoutEffect(() => {
    const probe = probeRef.current
    if (!probe) return
    const rect = probe.getBoundingClientRect()
    const w = rect.width / 10
    const h = rect.height
    if (w > 0 && h > 0 && (Math.abs(w - cell.w) > 0.01 || Math.abs(h - cell.h) > 0.01)) {
      setCell({ w, h })
    }
  }, [fontSize, text])

  const lines = String(text || '').split('\n')
  const cols = lines.reduce((m, l) => Math.max(m, l.length), 0)

  const roleClass = (b) => {
    if (b.id && b.id === focusId) return 'focus'
    switch (b.role) {
      case 'entity':
      case 'focus':
      case 'anchor':
      case 'source':
      case 'target':
      case 'container':
        return 'entity'
      case 'neighbour':
      case 'contained':
      case 'match':
      case 'item':
        return 'neighbour'
      case 'waypoint':
        return 'waypoint'
      case 'hub':
        return 'hub'
      default:
        return 'chunk'
    }
  }

  const labelClass = (b) => {
    const c = roleClass(b)
    if (c === 'entity') return 'ov-label magenta'
    if (c === 'neighbour') return 'ov-label green'
    if (c === 'waypoint' || c === 'hub') return 'ov-label amber'
    return 'ov-label'
  }

  return (
    <div className="stage">
      <div className="stage-inner" style={{ width: Math.max(cols * cell.w + 4, 0) }}>
        <span
          ref={probeRef}
          aria-hidden="true"
          style={{
            position: 'absolute',
            visibility: 'hidden',
            whiteSpace: 'pre',
            fontFamily: 'var(--mono)',
            fontSize: `${fontSize}px`,
            lineHeight: 1.32,
            top: 0,
            left: 0,
          }}
        >
          MMMMMMMMMM
        </span>

        <div className="overlay">
          {boxes.map((b, i) => {
            if (!b?.bbox) return null
            const { r0, c0, r1, c1 } = b.bbox
            const style = {
              top: r0 * cell.h,
              left: c0 * cell.w,
              height: Math.max(cell.h * (r1 - r0), 3),
              width: Math.max(cell.w * (c1 - c0), 3),
            }
            const cls = roleClass(b)
            return (
              <div key={`${b.id || b.label || 'b'}-${i}`} className={`ov-box ${cls}`} style={style}>
                {b.label && (r1 - r0) * cell.h > 10 && (
                  <span className={labelClass(b)} style={{ top: 0, left: 0 }}>
                    {b.label.slice(0, 28)}
                  </span>
                )}
              </div>
            )
          })}
        </div>

        <pre className="glyphs" ref={preRef} style={{ fontSize: `${fontSize}px` }}>
          {text}
        </pre>

        {scanning && <div className="scanline" />}
      </div>
    </div>
  )
}
