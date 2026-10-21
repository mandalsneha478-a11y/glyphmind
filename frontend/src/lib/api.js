/**
 * api.js — thin client for the GLYPHMIND backend.
 *
 * The interesting part is `streamPipeline`: every heavy endpoint has a
 * Server-Sent-Events twin that reports each pipeline stage as it completes.
 * We use fetch + a manual SSE parser rather than EventSource because the
 * endpoints are POSTs carrying a JSON body, which EventSource cannot do.
 */

const BASE = ''

export async function getJSON(path) {
  const r = await fetch(BASE + path)
  if (!r.ok) throw new Error(`${path} → ${r.status}`)
  return r.json()
}

export async function postJSON(path, body) {
  const r = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    let detail = ''
    try { detail = JSON.stringify(await r.json()) } catch (_) { /* noop */ }
    throw new Error(`${path} → ${r.status} ${detail}`)
  }
  return r.json()
}

/**
 * POST a body and consume the SSE response.
 * @param {string} path
 * @param {object} body
 * @param {{onStage?:Function,onResult?:Function,onError?:Function,signal?:AbortSignal}} handlers
 */
export async function streamPipeline(path, body, handlers = {}) {
  const { onStage, onResult, onError, signal } = handlers
  const res = await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    const msg = `${path} → ${res.status}`
    onError?.({ message: msg })
    throw new Error(msg)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })

    let idx
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const raw = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      let event = 'message'
      const dataLines = []
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
      }
      if (!dataLines.length) continue
      let payload
      try { payload = JSON.parse(dataLines.join('\n')) } catch (_) { continue }

      if (event === 'stage') onStage?.(payload)
      else if (event === 'result') onResult?.(payload.result, payload.wall_ms)
      else if (event === 'error') onError?.(payload)
    }
  }
}

/* ------------------------------------------------------------------ */
/* xterm-256 → CSS, so ANSI art renders in colour in the browser.      */
/* ------------------------------------------------------------------ */
const BASE16 = [
  [0, 0, 0], [205, 49, 49], [13, 188, 121], [229, 229, 16],
  [36, 114, 200], [188, 63, 188], [17, 168, 205], [229, 229, 229],
  [102, 102, 102], [241, 76, 76], [35, 209, 139], [245, 245, 67],
  [59, 142, 234], [214, 112, 214], [41, 184, 219], [255, 255, 255],
]
const LEVELS = [0, 95, 135, 175, 215, 255]

const CACHE = new Map()
export function xterm256(n) {
  if (CACHE.has(n)) return CACHE.get(n)
  let rgb
  if (n < 16) rgb = BASE16[n]
  else if (n < 232) {
    const i = n - 16
    rgb = [LEVELS[Math.floor(i / 36) % 6], LEVELS[Math.floor(i / 6) % 6], LEVELS[i % 6]]
  } else {
    const v = 8 + (n - 232) * 10
    rgb = [v, v, v]
  }
  const css = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
  CACHE.set(n, css)
  return css
}

/**
 * Parse an ANSI string with SGR 38;5;N foreground codes into coloured runs
 * grouped per line. Returns [[{text, color}], ...].
 */
export function parseAnsi(ansi) {
  const lines = (ansi || '').split('\n')
  let color = null
  return lines.map((line) => {
    const runs = []
    let i = 0
    let buf = ''
    while (i < line.length) {
      if (line[i] === '\x1b' && line[i + 1] === '[') {
        const end = line.indexOf('m', i)
        if (end === -1) { buf += line[i]; i += 1; continue }
        if (buf) { runs.push({ text: buf, color }); buf = '' }
        const code = line.slice(i + 2, end)
        if (code === '0' || code === '') color = null
        else {
          const m = code.match(/38;5;(\d+)/)
          if (m) color = xterm256(parseInt(m[1], 10))
        }
        i = end + 1
        continue
      }
      buf += line[i]
      i += 1
    }
    if (buf) runs.push({ text: buf, color })
    return runs
  })
}

export function download(filename, text, type = 'text/plain') {
  const blob = new Blob([text], { type: `${type};charset=utf-8` })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch (_) {
    const ta = document.createElement('textarea')
    ta.value = text
    document.body.appendChild(ta)
    ta.select()
    let ok = false
    try { ok = document.execCommand('copy') } catch (_) { ok = false }
    ta.remove()
    return ok
  }
}
