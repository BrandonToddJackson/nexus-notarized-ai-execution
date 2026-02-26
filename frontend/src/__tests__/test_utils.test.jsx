import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import '@testing-library/jest-dom'

vi.mock('reactflow', () => ({
  default: ({ children, onDrop, onDragOver, onNodeClick, onEdgeClick, onPaneClick }) => (
    <div data-testid="react-flow" onDrop={onDrop} onDragOver={onDragOver}>{children}</div>
  ),
  Background: () => null,
  Controls: () => null,
  MiniMap: () => null,
  Handle: ({ type, id, position }) => <div data-testid={`handle-${type}-${id || position}`} />,
  Position: { Top: 'top', Bottom: 'bottom', Left: 'left', Right: 'right' },
  MarkerType: { ArrowClosed: 'arrowclosed' },
  useNodesState: () => [[], vi.fn(), vi.fn()],
  useEdgesState: () => [[], vi.fn(), vi.fn()],
  addEdge: vi.fn((params, edges) => [...edges, params]),
  getBezierPath: () => ['M0 0', 50, 50],
  getSmoothStepPath: () => ['M0 0', 50, 50],
  EdgeLabelRenderer: ({ children }) => <div>{children}</div>,
  BaseEdge: () => <svg><path /></svg>,
}))

import { workflowToFlow } from '../components/workflow/utils/workflow-to-flow'
import { flowToWorkflow } from '../components/workflow/utils/flow-to-workflow'
import { applyDagreLayout } from '../components/workflow/utils/dagre-layout'

describe('workflowToFlow', () => {
  it('converts workflow with steps and edges to nodes and edges', () => {
    const workflow = {
      steps: [
        { id: 's1', name: 'Start', step_type: 'trigger', config: {} },
        { id: 's2', name: 'Do Thing', step_type: 'action', tool_name: 'web_search', config: {} },
      ],
      edges: [
        { id: 'e1', source_step_id: 's1', target_step_id: 's2', edge_type: 'sequence' },
      ],
    }
    const { nodes, edges } = workflowToFlow(workflow)
    expect(nodes).toHaveLength(2)
    expect(edges).toHaveLength(1)
    expect(edges[0].source).toBe('s1')
    expect(edges[0].target).toBe('s2')
  })

  it('maps step_type to correct node type', () => {
    const workflow = {
      steps: [
        { id: 's1', name: 'A', step_type: 'action', config: {} },
        { id: 's2', name: 'B', step_type: 'human_approval', config: {} },
        { id: 's3', name: 'C', step_type: 'branch', config: {} },
      ],
      edges: [],
    }
    const { nodes } = workflowToFlow(workflow)
    expect(nodes[0].type).toBe('actionNode')
    expect(nodes[1].type).toBe('approvalNode')
    expect(nodes[2].type).toBe('branchNode')
  })

  it('preserves step in node.data.step', () => {
    const step = { id: 's1', name: 'My Action', step_type: 'action', tool_name: 'web_search', config: { key: 'val' } }
    const { nodes } = workflowToFlow({ steps: [step], edges: [] })
    expect(nodes[0].data.step).toEqual(step)
    expect(nodes[0].data.label).toBe('My Action')
  })
})

describe('flowToWorkflow', () => {
  it('reverse maps node type to step_type', () => {
    const nodes = [
      { id: 'n1', type: 'actionNode', position: { x: 0, y: 0 }, data: { label: 'Action', step: { step_type: 'action' } } },
      { id: 'n2', type: 'approvalNode', position: { x: 100, y: 100 }, data: { label: 'Approve', step: { step_type: 'human_approval' } } },
    ]
    const result = flowToWorkflow(nodes, [])
    expect(result.steps[0].step_type).toBe('action')
    expect(result.steps[1].step_type).toBe('human_approval')
  })

  it('preserves position in step', () => {
    const nodes = [
      { id: 'n1', type: 'actionNode', position: { x: 42, y: 99 }, data: { label: 'Test', step: {} } },
    ]
    const result = flowToWorkflow(nodes, [])
    expect(result.steps[0].position).toEqual({ x: 42, y: 99 })
  })
})

describe('applyDagreLayout', () => {
  it('returns same number of nodes as input', () => {
    const nodes = [
      { id: 'a', position: { x: 0, y: 0 } },
      { id: 'b', position: { x: 0, y: 0 } },
      { id: 'c', position: { x: 0, y: 0 } },
    ]
    const edges = [{ source: 'a', target: 'b' }, { source: 'b', target: 'c' }]
    const result = applyDagreLayout(nodes, edges)
    expect(result).toHaveLength(3)
  })

  it('all returned nodes have position with x and y', () => {
    const nodes = [
      { id: 'a', position: { x: 0, y: 0 } },
      { id: 'b', position: { x: 0, y: 0 } },
    ]
    const edges = [{ source: 'a', target: 'b' }]
    const result = applyDagreLayout(nodes, edges)
    result.forEach((node) => {
      expect(node.position).toBeDefined()
      expect(typeof node.position.x).toBe('number')
      expect(typeof node.position.y).toBe('number')
    })
  })
})
