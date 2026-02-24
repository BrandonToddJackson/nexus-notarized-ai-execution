import { Handle, Position } from 'reactflow'

export default function LoopNode({ data, selected, isConnectable }) {
  return (
    <div className={`min-w-[160px] rounded-lg border-2 p-3 bg-gray-900 text-white text-sm ${selected ? 'border-cyan-400 shadow-lg shadow-cyan-500/20' : 'border-cyan-700'}`}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="flex items-center gap-2 font-medium">
        <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        <span>{data.label}</span>
      </div>
      {data.step?.config?.max_iterations && (
        <div className="mt-1 text-xs text-cyan-300">max: {data.step.config.max_iterations}</div>
      )}
      <Handle type="source" position={Position.Bottom} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Top} id="loop_back" style={{ right: '-8px', top: '50%', left: 'auto' }} isConnectable={isConnectable} />
    </div>
  )
}
