/**
 * SSE client for real-time NEXUS events via POST fetch + ReadableStream.
 *
 * Usage:
 *   import { connectStream } from './lib/stream'
 *   const close = connectStream('/v1/execute/stream', { task: '...' }, {
 *     onGateResult: (data) => console.log(data),
 *     onSealCreated: (data) => console.log(data),
 *   })
 */

import { getToken } from './auth'

function snakeToCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase())
}

export function connectStream(url, body, handlers = {}) {
  const controller = new AbortController()
  let retryDelay = 500
  let closed = false

  async function connect() {
    if (closed) return

    try {
      const token = getToken()
      const headers = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`

      const response = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Stream request failed' }))
        if (handlers.onError) handlers.onError(new Error((err.detail || `HTTP ${response.status}`).split('\n')[0].slice(0, 200)))
        scheduleReconnect()
        return
      }

      // Reset backoff on successful connection
      retryDelay = 500

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Split on double newline to extract complete SSE blocks
        const blocks = buffer.split('\n\n')
        // Last element is incomplete — keep it in the buffer
        buffer = blocks.pop()

        for (const block of blocks) {
          if (!block.trim()) continue

          let eventName = null
          let dataLine = null

          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) {
              eventName = line.slice(6).trim()
            } else if (line.startsWith('data:')) {
              dataLine = line.slice(5).trim()
            }
          }

          if (!dataLine) continue

          let data
          try {
            data = JSON.parse(dataLine)
          } catch {
            continue
          }

          if (eventName) {
            const handlerName = 'on' + snakeToCamel('_' + eventName)
            if (handlers[handlerName]) {
              handlers[handlerName](data)
            }
          }
        }
      }
    } catch (err) {
      if (closed || err.name === 'AbortError') return
      if (handlers.onError) handlers.onError(err)
      scheduleReconnect()
    }
  }

  function scheduleReconnect() {
    if (closed) return
    setTimeout(() => connect(), retryDelay)
    retryDelay = Math.min(retryDelay * 2, 30000)
  }

  connect()

  return function close() {
    closed = true
    controller.abort()
  }
}
