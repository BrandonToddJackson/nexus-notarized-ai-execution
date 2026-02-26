import { Handle, Position } from 'reactflow'

export default function BranchNode({ data, selected, isConnectable }) {
  return (
    <div className={`min-w-[160px] rounded-lg border-2 p-3 bg-gray-900 text-white text-sm ${selected ? 'border-yellow-400 shadow-lg shadow-yellow-500/20' : 'border-yellow-700'}`}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="flex items-center gap-2 font-medium">
        <svg className="w-4 h-4 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
        </svg>
        <span>{data.label}</span>
      </div>
      <div className="flex justify-between mt-2 text-xs text-gray-400">
        <span>Yes</span>
        <span>No</span>
      </div>
      <Handle type="source" position={Position.Bottom} id="true" style={{ left: '25%' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Bottom} id="false" style={{ left: '75%' }} isConnectable={isConnectable} />
    </div>
  )
}
