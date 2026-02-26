export default function ValidationModal({ nodes, edges, onSaveAnyway, onClose }) {
  const errors = []
  const warnings = []

  if (nodes.length === 0) {
    errors.push('Workflow must have at least one node')
  }

  const nodeIds = new Set(nodes.map(n => n.id))
  const nodesWithIncoming = new Set(edges.map(e => e.target))

  nodes.forEach(n => {
    const stepType = n.data?.step?.step_type || n.type?.replace('Node', '')
    if (stepType !== 'trigger' && n.type !== 'triggerNode' && !nodesWithIncoming.has(n.id)) {
      warnings.push(`Node "${n.data?.label || n.id}" has no incoming edges`)
    }
    if (n.type === 'branchNode' || stepType === 'branch') {
      const outgoing = edges.filter(e => e.source === n.id)
      if (outgoing.length < 2) errors.push(`Branch node "${n.data?.label || n.id}" needs at least 2 outgoing edges`)
    }
    if ((n.type === 'actionNode' || stepType === 'action') && !n.data?.step?.tool_name) {
      errors.push(`Action node "${n.data?.label || n.id}" must have a tool selected`)
    }
    if ((n.type === 'loopNode' || stepType === 'loop') && !n.data?.step?.config?.iterator) {
      warnings.push(`Loop node "${n.data?.label || n.id}" has no iterator set`)
    }
  })

  const hasErrors = errors.length > 0

  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-md p-6 flex flex-col gap-4">
        <h2 className="text-lg font-semibold text-white">Workflow Validation</h2>

        {errors.length > 0 && (
          <div className="flex flex-col gap-2">
            <div className="text-xs font-semibold text-red-400 uppercase tracking-wider">Errors (must fix)</div>
            {errors.map((e, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-red-300 bg-red-900/30 border border-red-800 rounded-lg px-3 py-2">
                <span className="mt-0.5">✗</span>
                <span>{e}</span>
              </div>
            ))}
          </div>
        )}

        {warnings.length > 0 && (
          <div className="flex flex-col gap-2">
            <div className="text-xs font-semibold text-yellow-400 uppercase tracking-wider">Warnings</div>
            {warnings.map((w, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-yellow-300 bg-yellow-900/30 border border-yellow-800 rounded-lg px-3 py-2">
                <span className="mt-0.5">⚠</span>
                <span>{w}</span>
              </div>
            ))}
          </div>
        )}

        {errors.length === 0 && warnings.length === 0 && (
          <div className="text-sm text-green-400 bg-green-900/30 border border-green-800 rounded-lg px-3 py-2">
            ✓ Workflow looks good!
          </div>
        )}

        <div className="flex gap-3 justify-end pt-2">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
            {hasErrors ? 'Fix Issues' : 'Close'}
          </button>
          {!hasErrors && (
            <button
              onClick={onSaveAnyway}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-500 transition-colors"
            >
              Save Anyway
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
