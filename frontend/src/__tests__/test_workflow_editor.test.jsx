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
  useNodesState: (init) => [init || [], vi.fn(), vi.fn()],
  useEdgesState: (init) => [init || [], vi.fn(), vi.fn()],
  addEdge: vi.fn((params, edges) => [...edges, params]),
  getBezierPath: () => ['M0 0', 50, 50],
  getSmoothStepPath: () => ['M0 0', 50, 50],
  EdgeLabelRenderer: ({ children }) => <div>{children}</div>,
  BaseEdge: () => <svg><path /></svg>,
}))

vi.mock('react-router-dom', () => ({
  useParams: () => ({ workflowId: 'new' }),
  useNavigate: () => vi.fn(),
}))

vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn(() => Promise.resolve([])),
    post: vi.fn(() => Promise.resolve({ id: 'wf-123' })),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
  },
}))

vi.mock('@dagrejs/dagre', () => {
  const Graph = vi.fn()
  Graph.prototype.setDefaultEdgeLabel = vi.fn()
  Graph.prototype.setGraph = vi.fn()
  Graph.prototype.setNode = vi.fn()
  Graph.prototype.setEdge = vi.fn()
  Graph.prototype.node = vi.fn(() => ({ x: 100, y: 100 }))
  return {
    default: {
      graphlib: { Graph },
      layout: vi.fn(),
    },
  }
})

import WorkflowEditor from '../pages/WorkflowEditor'

describe('WorkflowEditor', () => {
  it('renders without crashing (workflowId=new)', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Untitled Workflow')).toBeInTheDocument()
  })

  it('shows toolbar with Save button', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Save')).toBeInTheDocument()
  })

  it('shows NodeToolbox', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Add Node')).toBeInTheDocument()
  })

  it('NL Generate modal opens when Generate button clicked', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.queryByText('Describe what this workflow should do')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Generate'))
    expect(screen.getByText('Describe what this workflow should do')).toBeInTheDocument()
  })
})
