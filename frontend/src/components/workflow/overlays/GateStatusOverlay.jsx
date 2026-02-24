const GATES = [
  { key: 'g1', label: 'Scope' },
  { key: 'g2', label: 'Intent' },
  { key: 'g3', label: 'TTL' },
  { key: 'g4', label: 'Drift' },
]

const STATUS_COLORS = {
  idle: 'bg-gray-600',
  pass: 'bg-green-500',
  fail: 'bg-red-500',
}

export default function GateStatusOverlay({ gateStatus = {} }) {
  return (
    <div className="absolute bottom-4 left-4 bg-gray-900/90 border border-gray-700 rounded-lg p-2 flex gap-2 z-10">
      {GATES.map(({ key, label }) => (
        <div key={key} className="flex flex-col items-center gap-1">
          <div className={`w-3 h-3 rounded-full ${STATUS_COLORS[gateStatus[key] || 'idle']}`} />
          <span className="text-xs text-gray-400">{label}</span>
        </div>
      ))}
    </div>
  )
}
