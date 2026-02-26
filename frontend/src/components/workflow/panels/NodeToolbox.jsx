const NODE_TYPES = [
  { type: 'action', label: 'Action', icon: '▶', description: 'Execute a tool' },
  { type: 'branch', label: 'Branch', icon: '⑂', description: 'Conditional branch' },
  { type: 'loop', label: 'Loop', icon: '↺', description: 'Iterate over items' },
  { type: 'parallel', label: 'Parallel', icon: '⫴', description: 'Run steps in parallel' },
  { type: 'sub_workflow', label: 'Sub-Workflow', icon: '⊞', description: 'Nested workflow' },
  { type: 'wait', label: 'Wait', icon: '⏳', description: 'Wait / delay' },
  { type: 'human_approval', label: 'Approval', icon: '🛡', description: 'Require approval' },
]

export default function NodeToolbox() {
  const onDragStart = (event, nodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType)
    event.dataTransfer.effectAllowed = 'move'
  }

  return (
    <div className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col overflow-y-auto">
      <div className="px-3 py-2 text-xs font-semibold text-gray-400 uppercase tracking-wider border-b border-gray-800">
        Add Node
      </div>
      <div className="p-2 flex flex-col gap-1">
        {NODE_TYPES.map(({ type, label, icon, description }) => (
          <div
            key={type}
            className="flex items-center gap-2 p-2 rounded-lg border border-gray-700 bg-gray-800 cursor-grab hover:border-indigo-500 hover:bg-gray-700 transition-colors active:cursor-grabbing"
            draggable
            onDragStart={(e) => onDragStart(e, type)}
            title={description}
          >
            <span className="text-lg w-6 text-center">{icon}</span>
            <span className="text-xs text-gray-200 font-medium">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
