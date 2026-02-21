import React, { useState } from 'react'

const GATE_ORDER = ['scope', 'intent', 'ttl', 'drift']

const STATUS_STYLES = {
  executed: 'bg-green-900 text-green-300',
  blocked: 'bg-red-900 text-red-300',
  failed: 'bg-orange-900 text-orange-300',
  pending: 'bg-gray-700 text-gray-300',
}

const PERSONA_COLORS = {
  researcher: 'bg-blue-900 text-blue-300',
  analyst: 'bg-purple-900 text-purple-300',
  executor: 'bg-orange-900 text-orange-300',
}

function getGateDotColor(verdict) {
  if (verdict === 'pass' || verdict === true) return 'bg-green-500'
  if (verdict === 'fail' || verdict === false) return 'bg-red-500'
  if (verdict === 'skip') return 'bg-yellow-400'
  return 'bg-gray-600'
}

function normalizeGates(gates) {
  if (!gates) return {}
  if (Array.isArray(gates)) {
    const obj = {}
    for (const g of gates) {
      obj[g.gate_name] = { verdict: g.verdict, score: g.score, threshold: g.threshold }
    }
    return obj
  }
  return gates
}

export default function SealCard({ seal }) {
  const [expanded, setExpanded] = useState(false)

  if (!seal) return null

  const stepIndex = seal.step_index ?? 0
  const persona = seal.persona_name || seal.persona || 'unknown'
  const tool = seal.tool_name || seal.tool || 'unknown'
  const status = seal.status || 'pending'
  const sealId = seal.seal_id || seal.id || ''
  const reasoning = seal.reasoning || seal.cot_reasoning || []
  const gates = normalizeGates(seal.gates)

  const statusStyle = STATUS_STYLES[status] || STATUS_STYLES.pending
  const personaStyle = PERSONA_COLORS[persona] || 'bg-gray-700 text-gray-300'

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="bg-gray-700 text-gray-200 rounded-full px-2 py-0.5 text-xs font-medium">
            #{stepIndex}
          </span>
          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${personaStyle}`}>
            {persona}
          </span>
          <span className="text-sm text-gray-300">{tool}</span>
          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${statusStyle}`}>
            {status}
          </span>
        </div>
        {reasoning.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-gray-400 hover:text-gray-200 transition-colors ml-2 flex-shrink-0"
          >
            <svg
              className={`w-5 h-5 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        )}
      </div>

      <div className="flex items-center gap-1.5 mt-3">
        {GATE_ORDER.map(name => {
          const gate = gates[name]
          const verdict = gate ? gate.verdict : null
          const dotColor = getGateDotColor(verdict)
          return (
            <div
              key={name}
              title={`${name}: ${verdict ?? 'pending'}`}
              className={`w-3 h-3 rounded-full ${dotColor}`}
            />
          )
        })}
        {sealId && (
          <span className="text-xs text-gray-600 ml-auto font-mono truncate max-w-[120px]" title={sealId}>
            {sealId.slice(0, 8)}
          </span>
        )}
      </div>

      {expanded && reasoning.length > 0 && (
        <div className="mt-3 bg-gray-950 border border-gray-800 rounded p-3 space-y-1">
          {reasoning.map((line, i) => (
            <p key={i} className="text-xs text-gray-400 font-mono leading-relaxed">
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
