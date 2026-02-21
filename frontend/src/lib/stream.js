/**
 * SSE client for real-time NEXUS events.
 *
 * Usage:
 *   import { connectStream } from './lib/stream'
 *   const close = connectStream('/v1/execute/stream', { task: '...' }, {
 *     onGateResult: (data) => console.log(data),
 *     onSealCreated: (data) => console.log(data),
 *   })
 */

export function connectStream(url, body, handlers = {}) {
  // TODO: Implement SSE connection with EventSource or fetch + ReadableStream
  // For POST requests, use fetch with ReadableStream since EventSource only supports GET
  // Reconnect with exponential backoff on disconnect
  console.log('SSE streaming not yet implemented')
  return () => {} // close function
}
