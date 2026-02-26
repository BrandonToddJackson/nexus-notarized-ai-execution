// Converts WorkflowDefinition (backend DTO) -> ReactFlow { nodes, edges }

const STEP_TYPE_TO_NODE_TYPE = {
  action: 'actionNode',
  trigger: 'triggerNode',
  branch: 'branchNode',
  loop: 'loopNode',
  parallel: 'parallelNode',
  sub_workflow: 'subWorkflowNode',
  wait: 'waitNode',
  human_approval: 'approvalNode',
}

const EDGE_TYPE_TO_RF_TYPE = {
  conditional: 'conditionalEdge',
  error: 'errorEdge',
  loop_back: 'loopBackEdge',
}

export function workflowToFlow(workflow) {
  if (!workflow) return { nodes: [], edges: [] }

  const steps = workflow.steps || []
  const workflowEdges = workflow.edges || []

  const nodes = steps.map((step) => ({
    id: step.id,
    type: STEP_TYPE_TO_NODE_TYPE[step.step_type] || 'actionNode',
    position: step.position || { x: 0, y: 0 },
    data: {
      label: step.name || step.step_type,
      step,
    },
  }))

  const edges = workflowEdges.map((edge) => ({
    id: edge.id || `${edge.source_step_id}-${edge.target_step_id}`,
    source: edge.source_step_id,
    target: edge.target_step_id,
    sourceHandle: edge.source_handle || null,
    type: EDGE_TYPE_TO_RF_TYPE[edge.edge_type] || 'default',
    label: edge.condition || undefined,
    data: { edge },
  }))

  return { nodes, edges }
}
