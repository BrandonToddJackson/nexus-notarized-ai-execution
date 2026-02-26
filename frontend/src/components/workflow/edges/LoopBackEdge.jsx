import { getSmoothStepPath, BaseEdge, EdgeLabelRenderer } from 'reactflow'

export default function LoopBackEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd }) {
  const [edgePath, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={{ stroke: '#06b6d4', strokeWidth: 2 }} />
      <EdgeLabelRenderer>
        <div
          className="nopan absolute pointer-events-all px-1.5 py-0.5 bg-cyan-900 border border-cyan-600 rounded text-xs text-cyan-200"
          style={{ transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)` }}
        >
          ↺
        </div>
      </EdgeLabelRenderer>
    </>
  )
}
