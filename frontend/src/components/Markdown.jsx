import React from 'react'

/**
 * A deliberately tiny markdown renderer.
 *
 * The grounded composer emits a constrained subset — bold, inline code,
 * bullet lists, italics — and nothing else. Pulling a full markdown library in
 * for that would be heavier than the feature. This covers exactly what the
 * backend can produce, and ignores anything it cannot.
 */
function inline(text, keyBase) {
  const out = []
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g
  let last = 0
  let m
  let i = 0
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const tok = m[0]
    const k = `${keyBase}-${i++}`
    if (tok.startsWith('**')) out.push(<strong key={k}>{tok.slice(2, -2)}</strong>)
    else if (tok.startsWith('`')) out.push(<code key={k}>{tok.slice(1, -1)}</code>)
    else out.push(<em key={k}>{tok.slice(1, -1)}</em>)
    last = m.index + tok.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

export default function Markdown({ text }) {
  const lines = String(text || '').split('\n')
  const blocks = []
  let list = null
  let key = 0

  const flush = () => {
    if (list && list.length) {
      blocks.push(<ul key={`ul-${key++}`}>{list}</ul>)
    }
    list = null
  }

  lines.forEach((raw, i) => {
    const line = raw.replace(/\s+$/, '')
    if (/^\s*[-•]\s+/.test(line)) {
      if (!list) list = []
      list.push(<li key={`li-${i}`}>{inline(line.replace(/^\s*[-•]\s+/, ''), `li${i}`)}</li>)
      return
    }
    flush()
    if (!line.trim()) return
    blocks.push(<div className="para" key={`p-${i}`}>{inline(line, `p${i}`)}</div>)
  })
  flush()

  return <div className="answer">{blocks}</div>
}
