import { Handle, Position } from 'reactflow'

export default function TriggerNode({ data, selected, isConnectable }) {
  return (
    <div className={`min-w-[160px] rounded-lg border-2 p-3 bg-indigo-900 text-white text-sm ${selected ? 'border-indigo-400 shadow-lg shadow-indigo-500/20' : 'border-indigo-700'}`}>
      <div className="flex items-center gap-2 font-medium">
        <svg className="w-4 h-4 text-indigo-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        <span>{data.label}</span>
      </div>
      {data.step?.config?.trigger_type && (
        <div className="mt-1 text-xs text-indigo-300">{data.step.config.trigger_type}</div>
      )}
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
    </div>
  )
}
