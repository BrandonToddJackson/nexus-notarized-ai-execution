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

import ActionNode from '../components/workflow/nodes/ActionNode'
import BranchNode from '../components/workflow/nodes/BranchNode'
import TriggerNode from '../components/workflow/nodes/TriggerNode'
import ApprovalNode from '../components/workflow/nodes/ApprovalNode'

describe('ActionNode', () => {
  it('renders with label', () => {
    render(<ActionNode data={{ label: 'My Action' }} selected={false} isConnectable={true} />)
    expect(screen.getByText('My Action')).toBeInTheDocument()
  })

  it('shows tool_name when set in data.step', () => {
    render(
      <ActionNode
        data={{ label: 'Action', step: { tool_name: 'web_search' } }}
        selected={false}
        isConnectable={true}
      />,
    )
    expect(screen.getByText('web_search')).toBeInTheDocument()
  })

  it('selected state shows indigo ring', () => {
    const { container } = render(
      <ActionNode data={{ label: 'Test' }} selected={true} isConnectable={true} />,
    )
    const node = container.firstChild
    expect(node.className).toContain('border-indigo-500')
    expect(node.className).toContain('shadow-lg')
  })
})

describe('BranchNode', () => {
  it('renders with true/false handles', () => {
    render(<BranchNode data={{ label: 'Branch' }} selected={false} isConnectable={true} />)
    expect(screen.getByText('Branch')).toBeInTheDocument()
    expect(screen.getByText('Yes')).toBeInTheDocument()
    expect(screen.getByText('No')).toBeInTheDocument()
    expect(screen.getByTestId('handle-source-true')).toBeInTheDocument()
    expect(screen.getByTestId('handle-source-false')).toBeInTheDocument()
  })
})

describe('TriggerNode', () => {
  it('renders without target handle (only source)', () => {
    render(<TriggerNode data={{ label: 'Start' }} selected={false} isConnectable={true} />)
    expect(screen.getByText('Start')).toBeInTheDocument()
    expect(screen.getByTestId('handle-source-bottom')).toBeInTheDocument()
    expect(screen.queryByTestId('handle-target-top')).not.toBeInTheDocument()
  })
})

describe('ApprovalNode', () => {
  it('renders with approved/rejected handles', () => {
    render(<ApprovalNode data={{ label: 'Approve' }} selected={false} isConnectable={true} />)
    expect(screen.getByText('Approve')).toBeInTheDocument()
    expect(screen.getByText('Approved')).toBeInTheDocument()
    expect(screen.getByText('Rejected')).toBeInTheDocument()
    expect(screen.getByTestId('handle-source-approved')).toBeInTheDocument()
    expect(screen.getByTestId('handle-source-rejected')).toBeInTheDocument()
  })
})
