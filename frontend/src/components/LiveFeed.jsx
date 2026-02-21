import React, { useEffect, useRef } from 'react'

const MAX_EVENTS = 200

const TYPE_STYLES = {
  chain_started: 'bg-blue-900 text-blue-300',
  step_started: 'bg-indigo-900 text-indigo-300',
  step_completed: 'bg-emerald-950 text-emerald-400',
  chain_completed: 'bg-green-900 text-green-300',
  error: 'bg-red-900 text-red-300',
}

function getGateResultStyle(event) {
  const verdict = event.data?.verdict ?? event.data?.gate_verdict
  if (verdict === 'pass' || verdict === true) return 'bg-green-900 text-green-300'
  if (verdict === 'fail' || verdict === false) return 'bg-red-900 text-red-300'
  return 'bg-yellow-900 text-yellow-300'
}

function getSealCreatedStyle(event) {
  const status = event.data?.status
  if (status === 'executed') return 'bg-green-900 text-green-300'
  if (status === 'blocked') return 'bg-red-900 text-red-300'
  return 'bg-gray-700 text-gray-300'
}

function getTypeStyle(event) {
  if (event.type === 'gate_result') return getGateResultStyle(event)
  if (event.type === 'seal_created') return getSealCreatedStyle(event)
  return TYPE_STYLES[event.type] || 'bg-gray-700 text-gray-300'
}

function formatTimestamp(ts) {
  if (!ts) return ''
  const date = ts instanceof Date ? ts : new Date(ts)
  if (isNaN(date.getTime())) return ''
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function summarize(event) {
  const d = event.data || {}
  switch (event.type) {
    case 'chain_started':
      return `Chain ${d.chain_id ? d.chain_id.slice(0, 8) : ''} started${d.task ? ': ' + d.task : ''}`
    case 'step_started': {
      const step = d.step_index ?? d.step ?? '?'
      const persona = d.persona_name || d.persona || ''
      const tool = d.tool_name || d.tool || ''
      return `Step ${step}${persona ? ': ' + persona : ''}${tool ? ' / ' + tool : ''}`
    }
    case 'gate_result': {
      const gate = d.gate_name || d.gate || '?'
      const verdict = d.verdict ?? d.gate_verdict ?? '?'
      const score = d.score != null ? ` ${Number(d.score).toFixed(2)}` : ''
      return `Gate: ${gate} ${String(verdict).toUpperCase()}${score}`
    }
    case 'seal_created': {
      const status = d.status || 'sealed'
      const id = d.seal_id || d.id || ''
      return `Seal ${status}${id ? ' ' + id.slice(0, 8) : ''}`
    }
    case 'step_completed': {
      const step = d.step_index ?? d.step ?? '?'
      return `Step ${step} completed`
    }
    case 'chain_completed': {
      const dur = d.duration_seconds || d.duration
      return `Chain completed${dur ? ' in ' + Number(dur).toFixed(1) + 's' : ''}`
    }
    case 'error':
      return d.message || d.error || 'An error occurred'
    default:
      return event.type
  }
}

export default function LiveFeed({ events = [] }) {
  const scrollRef = useRef(null)
  const displayEvents = events.length > MAX_EVENTS ? events.slice(-MAX_EVENTS) : events

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [displayEvents.length])

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-sm font-medium text-gray-200">Live Feed</h3>
        <span className="bg-gray-700 text-gray-300 rounded-full px-2 py-0.5 text-xs font-medium">
          {displayEvents.length}
        </span>
      </div>

      <div className="overflow-y-auto max-h-96 p-2 space-y-1">
        {displayEvents.length === 0 ? (
          <p className="text-gray-500 text-sm text-center py-6">Waiting for events...</p>
        ) : (
          displayEvents.map((event, i) => {
            const typeStyle = getTypeStyle(event)
            return (
              <div key={i} className="flex items-start gap-2 px-2 py-1.5 rounded hover:bg-gray-800/50">
                <span className="text-xs text-gray-500 flex-shrink-0 mt-0.5 w-16 text-right font-mono">
                  {formatTimestamp(event.timestamp)}
                </span>
                <span className={`rounded-full px-2 py-0.5 text-xs font-medium flex-shrink-0 ${typeStyle}`}>
                  {event.type}
                </span>
                <span className="text-xs text-gray-300 leading-relaxed">
                  {summarize(event)}
                </span>
              </div>
            )
          })
        )}
        <div ref={scrollRef} />
      </div>
    </div>
  )
}
