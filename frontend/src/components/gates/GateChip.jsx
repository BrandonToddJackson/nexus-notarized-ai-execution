const GATE_NAMES = { scope: 'G1 Scope', intent: 'G2 Intent', ttl: 'G3 TTL', drift: 'G4 Drift' }

export function GateChip({ gate, result, value, threshold }) {
  const label = GATE_NAMES[gate] ?? gate
  const pass = result === 'pass' || result === true
  const icon = pass ? '\u2713' : '\u2717'
  const color = pass ? 'text-green-600 bg-green-50 border-green-200' : 'text-red-600 bg-red-50 border-red-200'
  const title = value != null
    ? `${label}: ${value}${threshold != null ? ` / ${threshold}` : ''} (${pass ? 'pass' : 'fail'})`
    : `${label}: ${pass ? 'pass' : 'fail'}`
  return (
    <span title={title} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-xs font-mono ${color}`}>
      {label} {icon} {value != null ? `${value}${threshold ? `/${threshold}` : ''}` : ''}
    </span>
  )
}
