import { vi, describe, it, expect } from 'vitest'
import { render, screen, fireEvent, act } from '@testing-library/react'
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

vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn(() => Promise.resolve([])),
    post: vi.fn(() => Promise.resolve({})),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
  },
}))

import NodeToolbox from '../components/workflow/panels/NodeToolbox'
import PropertiesPanel from '../components/workflow/panels/PropertiesPanel'
import VersionHistoryPanel from '../components/workflow/panels/VersionHistoryPanel'

describe('NodeToolbox', () => {
  it('renders all 7 node types', () => {
    render(<NodeToolbox />)
    expect(screen.getByText('Action')).toBeInTheDocument()
    expect(screen.getByText('Branch')).toBeInTheDocument()
    expect(screen.getByText('Loop')).toBeInTheDocument()
    expect(screen.getByText('Parallel')).toBeInTheDocument()
    expect(screen.getByText('Sub-Workflow')).toBeInTheDocument()
    expect(screen.getByText('Wait')).toBeInTheDocument()
    expect(screen.getByText('Approval')).toBeInTheDocument()
  })

  it('drag start sets correct dataTransfer data', () => {
    render(<NodeToolbox />)
    const actionNode = screen.getByText('Action').closest('[draggable]')
    const mockDataTransfer = { setData: vi.fn(), effectAllowed: '' }
    fireEvent.dragStart(actionNode, { dataTransfer: mockDataTransfer })
    expect(mockDataTransfer.setData).toHaveBeenCalledWith('application/reactflow', 'action')
  })
})

describe('PropertiesPanel', () => {
  it('shows empty state when no node selected', async () => {
    await act(async () => { render(<PropertiesPanel selectedNode={null} selectedEdge={null} onChange={vi.fn()} />) })
    expect(screen.getByText(/select a node or edge/i)).toBeInTheDocument()
  })

  it('shows action form when action node selected', async () => {
    const node = {
      id: 'n1',
      type: 'actionNode',
      data: {
        label: 'My Action',
        step: { name: 'My Action', step_type: 'action', tool_name: '' },
      },
    }
    await act(async () => { render(<PropertiesPanel selectedNode={node} selectedEdge={null} onChange={vi.fn()} />) })
    expect(screen.getByText('Label')).toBeInTheDocument()
    expect(screen.getByText('Tool')).toBeInTheDocument()
  })
})

describe('VersionHistoryPanel', () => {
  it('shows workflow name and version', () => {
    const workflow = { name: 'Test Workflow', version: 3, status: 'active' }
    render(<VersionHistoryPanel workflow={workflow} onClose={vi.fn()} />)
    expect(screen.getByText('Test Workflow')).toBeInTheDocument()
    expect(screen.getByText('v3')).toBeInTheDocument()
  })
})
