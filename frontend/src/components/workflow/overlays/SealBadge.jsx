const STATUS_COLORS = {
  EXECUTED: 'bg-green-500',
  BLOCKED: 'bg-red-500',
  ERROR: 'bg-yellow-500',
}

export default function SealBadge({ seal }) {
  if (!seal) return null
  const shortId = (seal.seal_id || '').slice(0, 8)
  const color = STATUS_COLORS[seal.status] || 'bg-gray-500'
  return (
    <div className="absolute top-0 right-0 flex items-center gap-1 bg-gray-900/90 border border-gray-700 rounded-bl-lg rounded-tr-lg px-1.5 py-0.5">
      <div className={`w-1.5 h-1.5 rounded-full ${color}`} />
      <span className="text-xs font-mono text-gray-400">{shortId}</span>
    </div>
  )
}
