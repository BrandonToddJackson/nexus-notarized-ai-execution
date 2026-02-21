import React, { useState, useEffect, useRef } from 'react'

const GATE_ORDER = ['scope', 'intent', 'ttl', 'drift']
const GATE_LABELS = { scope: 'SCOPE', intent: 'INTENT', ttl: 'TTL', drift: 'DRIFT' }

function getGateStyle(gate) {
  if (!gate || gate.verdict === undefined || gate.verdict === null) {
    return { bg: 'bg-gray-700', glow: '' }
  }
  switch (gate.verdict) {
    case 'pass':
    case true:
      return { bg: 'bg-green-500', glow: 'shadow-[0_0_12px_rgba(34,197,94,0.6)]' }
    case 'fail':
    case false:
      return { bg: 'bg-red-500', glow: 'shadow-[0_0_12px_rgba(239,68,68,0.6)]' }
    case 'skip':
      return { bg: 'bg-yellow-400', glow: 'shadow-[0_0_12px_rgba(250,204,21,0.6)]' }
    default:
      return { bg: 'bg-gray-700', glow: '' }
  }
}

function getConnectorColor(leftGate, rightGate) {
  const leftPass = leftGate && (leftGate.verdict === 'pass' || leftGate.verdict === true)
  const rightPass = rightGate && (rightGate.verdict === 'pass' || rightGate.verdict === true)
  return leftPass && rightPass ? 'bg-green-500' : 'bg-gray-600'
}

function getOverallStatus(gates) {
  const results = GATE_ORDER.map(name => gates[name])
  const hasAny = results.some(g => g && g.verdict !== undefined && g.verdict !== null)
  if (!hasAny) return { text: 'Checking...', color: 'text-gray-400' }

  const allResolved = results.every(g => g && g.verdict !== undefined && g.verdict !== null)
  const hasFail = results.some(g => g && (g.verdict === 'fail' || g.verdict === false))

  if (hasFail) return { text: 'Gate blocked', color: 'text-red-400' }
  if (allResolved) return { text: 'All gates passed', color: 'text-green-400' }
  return { text: 'Checking...', color: 'text-gray-400' }
}

export default function GateVisualizer({ gates = {} }) {
  const [pulsing, setPulsing] = useState({})
  const prevGatesRef = useRef({})

  useEffect(() => {
    const newPulses = {}
    for (const name of GATE_ORDER) {
      const prev = prevGatesRef.current[name]
      const curr = gates[name]
      const prevVerdict = prev ? prev.verdict : undefined
      const currVerdict = curr ? curr.verdict : undefined
      if (currVerdict !== undefined && currVerdict !== prevVerdict) {
        newPulses[name] = true
      }
    }

    if (Object.keys(newPulses).length > 0) {
      setPulsing(prev => ({ ...prev, ...newPulses }))
      const timer = setTimeout(() => {
        setPulsing(prev => {
          const next = { ...prev }
          for (const name of Object.keys(newPulses)) {
            delete next[name]
          }
          return next
        })
      }, 300)
      prevGatesRef.current = { ...gates }
      return () => clearTimeout(timer)
    }
    prevGatesRef.current = { ...gates }
  }, [gates])

  const overall = getOverallStatus(gates)

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-center gap-0">
        {GATE_ORDER.map((name, i) => {
          const gate = gates[name] || null
          const style = getGateStyle(gate)
          const isPulsing = pulsing[name]
          const score = gate && gate.score != null ? gate.score.toFixed(2) : '\u2014'

          return (
            <React.Fragment key={name}>
              <div className="flex flex-col items-center">
                <div
                  className={[
                    'w-12 h-12 rounded-full',
                    'transition-all duration-500',
                    style.bg,
                    style.glow,
                    isPulsing ? 'scale-110' : 'scale-100',
                  ].join(' ')}
                />
                <span className="text-xs text-gray-400 mt-1.5">{GATE_LABELS[name]}</span>
                <span className="text-xs text-gray-500 mt-0.5">{score}</span>
              </div>
              {i < GATE_ORDER.length - 1 && (
                <div
                  className={[
                    'h-0.5 w-10 -mt-6',
                    getConnectorColor(gates[GATE_ORDER[i]], gates[GATE_ORDER[i + 1]]),
                    'transition-colors duration-500',
                  ].join(' ')}
                />
              )}
            </React.Fragment>
          )
        })}
      </div>
      <div className={`text-center mt-4 text-sm font-medium ${overall.color}`}>
        {overall.text}
      </div>
    </div>
  )
}
