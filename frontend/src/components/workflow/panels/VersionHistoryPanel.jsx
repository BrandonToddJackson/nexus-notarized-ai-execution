export default function VersionHistoryPanel({ workflow, onClose }) {
  const formatDate = (d) => d ? new Date(d).toLocaleString() : '—'
  const statusColors = { active: 'text-green-400', paused: 'text-yellow-400', draft: 'text-gray-400', archived: 'text-gray-500' }

  return (
    <div className="fixed right-0 top-0 h-full w-80 bg-gray-900 border-l border-gray-800 shadow-2xl z-40 flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-sm font-semibold text-white">Version History</h3>
        <button onClick={onClose} className="text-gray-400 hover:text-white transition-colors">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="p-4 flex flex-col gap-3">
        <div className="bg-gray-800 rounded-lg p-3 flex flex-col gap-2">
          <div className="text-xs text-gray-400">Current Version</div>
          <div className="text-white font-semibold">{workflow?.name || 'Untitled Workflow'}</div>
          <div className="flex items-center gap-2">
            <span className="text-xs bg-indigo-900 text-indigo-300 px-2 py-0.5 rounded">v{workflow?.version || 1}</span>
            <span className={`text-xs font-medium ${statusColors[workflow?.status] || 'text-gray-400'}`}>
              {workflow?.status || 'draft'}
            </span>
          </div>
          <div className="text-xs text-gray-500">{formatDate(workflow?.updated_at)}</div>
        </div>
        <div className="text-xs text-gray-500 text-center pt-2">
          Full version history coming soon
        </div>
        <button
          disabled
          className="w-full py-2 px-3 bg-gray-700 text-gray-500 rounded-lg text-sm cursor-not-allowed"
          title="Full version list not yet implemented"
        >
          Restore Previous Version
        </button>
      </div>
    </div>
  )
}
