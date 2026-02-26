import { Handle, Position } from 'reactflow'

export default function SubWorkflowNode({ data, selected, isConnectable }) {
  return (
    <div className={`min-w-[160px] rounded-lg border-2 p-3 bg-gray-900 text-white text-sm ${selected ? 'border-orange-400 shadow-lg shadow-orange-500/20' : 'border-orange-700'}`}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="flex items-center gap-2 font-medium">
        <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
        <span>{data.label}</span>
      </div>
      {data.step?.config?.workflow_id && (
        <div className="mt-1 text-xs text-orange-300">{data.step.config.workflow_id}</div>
      )}
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  )
}
