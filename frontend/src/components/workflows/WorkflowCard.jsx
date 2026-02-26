import { useNavigate } from 'react-router-dom'
import { StatusBadge } from '../shared/StatusBadge.jsx'
import { Toggle } from '../shared/Toggle.jsx'

export function WorkflowCard({ workflow, onToggle }) {
  const navigate = useNavigate()
  const stepCount = workflow.steps?.length || 0

  return (
    <div className="bg-white border rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-900 truncate pr-2">{workflow.name || 'Untitled'}</h3>
        <StatusBadge status={workflow.status || 'draft'} />
      </div>
      {workflow.description && (
        <p className="text-xs text-gray-600 mb-3 line-clamp-2">{workflow.description}</p>
      )}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-3">
          {workflow.trigger_type && (
            <span className="bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded">{workflow.trigger_type}</span>
          )}
          <span>{stepCount} steps</span>
          <span>v{workflow.version || 1}</span>
        </div>
        <div className="flex items-center gap-2">
          {workflow.last_execution_status && (
            <span className={`w-2 h-2 rounded-full ${
              workflow.last_execution_status === 'success' ? 'bg-green-500' :
              workflow.last_execution_status === 'failed' ? 'bg-red-500' : 'bg-gray-400'
            }`} />
          )}
          <Toggle
            checked={workflow.status === 'active'}
            onChange={(v) => onToggle?.(workflow.id, v ? 'active' : 'paused')}
          />
        </div>
      </div>
      <button
        onClick={() => navigate(`/workflows/${workflow.id}/edit`)}
        className="mt-3 w-full py-1.5 text-xs text-indigo-600 bg-indigo-50 hover:bg-indigo-100 rounded transition-colors"
      >
        Edit
      </button>
    </div>
  )
}
