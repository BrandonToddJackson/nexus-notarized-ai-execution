import { getBezierPath, BaseEdge, EdgeLabelRenderer } from 'reactflow'

export default function ConditionalEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd }) {
  const [edgePath, labelX, labelY] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={{ stroke: '#f97316', strokeDasharray: '5,3', strokeWidth: 2 }} />
      {data?.edge?.condition && (
        <EdgeLabelRenderer>
          <div
            className="nopan absolute pointer-events-all px-2 py-0.5 bg-orange-900 border border-orange-600 rounded text-xs text-orange-200 font-mono"
            style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}
          >
            {data.edge.condition}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
