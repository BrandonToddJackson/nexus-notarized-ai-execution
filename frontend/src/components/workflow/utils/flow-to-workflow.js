// Converts ReactFlow { nodes, edges } -> backend DTO

const NODE_TYPE_TO_STEP_TYPE = {
  actionNode: 'action',
  triggerNode: 'trigger',
  branchNode: 'branch',
  loopNode: 'loop',
  parallelNode: 'parallel',
  subWorkflowNode: 'sub_workflow',
  waitNode: 'wait',
  approvalNode: 'human_approval',
}

const RF_TYPE_TO_EDGE_TYPE = {
  conditionalEdge: 'conditional',
  errorEdge: 'error',
  loopBackEdge: 'loop_back',
  default: 'sequence',
}

export function flowToWorkflow(nodes, edges, existingWorkflow = {}) {
  const steps = nodes.map((n) => ({
    ...(n.data?.step || {}),
    id: n.id,
    name: n.data?.label || n.data?.step?.name || n.type,
    step_type: NODE_TYPE_TO_STEP_TYPE[n.type] || 'action',
    position: n.position,
  }))

  const workflowEdges = edges.map((e) => ({
    id: e.id,
    source_step_id: e.source,
    target_step_id: e.target,
    source_handle: e.sourceHandle || null,
    edge_type: RF_TYPE_TO_EDGE_TYPE[e.type] || 'sequence',
    condition: e.label || e.data?.edge?.condition || null,
  }))

  return {
    ...existingWorkflow,
    steps,
    edges: workflowEdges,
  }
}
