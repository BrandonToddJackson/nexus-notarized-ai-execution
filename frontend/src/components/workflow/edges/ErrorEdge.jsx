import { getBezierPath, BaseEdge, EdgeLabelRenderer } from 'reactflow'

export default function ErrorEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd }) {
  const [edgePath, labelX, labelY] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={{ stroke: '#ef4444', strokeDasharray: '5,3', strokeWidth: 2 }} />
      <EdgeLabelRenderer>
        <div
          className="nopan absolute pointer-events-all px-2 py-0.5 bg-red-900 border border-red-600 rounded text-xs text-red-200 font-mono"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}
        >
          ERROR
        </div>
      </EdgeLabelRenderer>
    </>
  )
}
