import { Handle, Position } from 'reactflow'

export default function ApprovalNode({ data, selected, isConnectable }) {
  return (
    <div className={`min-w-[160px] rounded-lg border-2 p-3 bg-gradient-to-b from-red-900 to-orange-900 text-white text-sm ${selected ? 'border-orange-400 shadow-lg shadow-orange-500/20' : 'border-orange-800'}`}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      <div className="flex items-center gap-2 font-medium">
        <svg className="w-4 h-4 text-orange-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
        </svg>
        <span>{data.label}</span>
      </div>
      <div className="flex justify-between mt-2 text-xs">
        <span className="text-green-300">Approved</span>
        <span className="text-red-300">Rejected</span>
      </div>
      <Handle type="source" position={Position.Bottom} id="approved" style={{ left: '25%' }} isConnectable={isConnectable} />
      <Handle type="source" position={Position.Bottom} id="rejected" style={{ left: '75%' }} isConnectable={isConnectable} />
    </div>
  )
}
