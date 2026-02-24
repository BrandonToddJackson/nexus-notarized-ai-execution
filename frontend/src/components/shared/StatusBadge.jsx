const COLORS = {
  running: 'bg-blue-100 text-blue-800',
  success: 'bg-green-100 text-green-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  blocked: 'bg-orange-100 text-orange-800',
  cancelled: 'bg-gray-100 text-gray-700',
  pending: 'bg-yellow-100 text-yellow-800',
}

export function StatusBadge({ status }) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${COLORS[status] ?? COLORS.cancelled}`}>
      {status}
    </span>
  )
}
