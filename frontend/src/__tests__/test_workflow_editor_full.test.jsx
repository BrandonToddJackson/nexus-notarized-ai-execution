import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../mocks/server.js'
import { fixtures } from '../mocks/data.js'

// Mock reactflow
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

// Mock react-router-dom — start with workflowId=new
let mockParams = { workflowId: 'new' }
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useParams: () => mockParams,
  useNavigate: () => mockNavigate,
}))

vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn(() => Promise.resolve(fixtures.workflow)),
    post: vi.fn(() => Promise.resolve({ id: 'wf-123' })),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
    delete: vi.fn(() => Promise.resolve({})),
  },
}))

import WorkflowEditor from '../pages/WorkflowEditor.jsx'

beforeEach(() => {
  mockParams = { workflowId: 'new' }
  mockNavigate.mockReset()
})

describe('WorkflowEditor full tests', () => {
  it('WE1: New workflow renders with Untitled Workflow title', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Untitled Workflow')).toBeInTheDocument()
  })

  it('WE2: Save button is present', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Save')).toBeInTheDocument()
  })

  it('WE3: Generate button opens NL Generate modal', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.queryByText('Describe what this workflow should do')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('Generate'))
    expect(screen.getByText('Describe what this workflow should do')).toBeInTheDocument()
  })

  it('WE4: NodeToolbox Add Node panel is visible', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Add Node')).toBeInTheDocument()
  })

  it('WE5: Validate button is present', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Validate')).toBeInTheDocument()
  })

  it('WE6: Auto Layout button is present', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('Auto Layout')).toBeInTheDocument()
  })

  it('WE7: Version badge shows v1 for new workflow', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    expect(screen.getByText('v1')).toBeInTheDocument()
  })

  it('WE8: Back button is present', async () => {
    await act(async () => { render(<WorkflowEditor />) })
    // Back button is the first button with an SVG chevron
    const buttons = screen.getAllByRole('button')
    // First button has title "Back to workflows"
    const backBtn = buttons.find(b => b.getAttribute('title') === 'Back to workflows')
    expect(backBtn).toBeDefined()
  })

  it('WE9: Save on new workflow calls POST and navigates', async () => {
    const api = (await import('../lib/api')).default
    api.post.mockResolvedValue({ id: 'wf-new-123' })

    await act(async () => { render(<WorkflowEditor />) })

    const saveBtn = screen.getByText('Save')
    await act(async () => { fireEvent.click(saveBtn) })

    expect(api.post).toHaveBeenCalledWith('/v2/workflows', expect.any(Object))
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/workflows/wf-new-123')
    })
  })

  it('WE10: Existing workflow loads and shows its name', async () => {
    mockParams = { workflowId: 'wf-1' }
    const api = (await import('../lib/api')).default
    api.get.mockResolvedValue({ ...fixtures.workflow, name: 'My Loaded Workflow' })

    await act(async () => { render(<WorkflowEditor />) })

    await waitFor(() => {
      expect(screen.getByText('My Loaded Workflow')).toBeInTheDocument()
    })
  })

  it('WE11: Save on existing workflow calls PUT', async () => {
    mockParams = { workflowId: 'wf-1' }
    const api = (await import('../lib/api')).default
    api.get.mockResolvedValue(fixtures.workflow)
    api.put.mockResolvedValue({})

    await act(async () => { render(<WorkflowEditor />) })

    await waitFor(() => {
      expect(screen.getByText('Test Workflow')).toBeInTheDocument()
    })

    const saveBtn = screen.getByText('Save')
    await act(async () => { fireEvent.click(saveBtn) })

    expect(api.put).toHaveBeenCalledWith('/v2/workflows/wf-1', expect.any(Object))
  })

  it('WE12: Existing workflow shows Activate/Pause button', async () => {
    mockParams = { workflowId: 'wf-1' }
    const api = (await import('../lib/api')).default
    api.get.mockResolvedValue({ ...fixtures.workflow, status: 'draft' })

    await act(async () => { render(<WorkflowEditor />) })

    await waitFor(() => {
      expect(screen.getByText('Activate')).toBeInTheDocument()
    })
  })

  it('WE13: Active workflow shows Pause button', async () => {
    mockParams = { workflowId: 'wf-active' }
    const api = (await import('../lib/api')).default
    api.get.mockResolvedValue({ ...fixtures.activeWorkflow })

    await act(async () => { render(<WorkflowEditor />) })

    await waitFor(() => {
      expect(screen.getByText('Pause')).toBeInTheDocument()
    })
  })

  it('WE14: API error shows error message', async () => {
    mockParams = { workflowId: 'wf-bad' }
    const api = (await import('../lib/api')).default
    api.get.mockRejectedValue(new Error('Workflow not found'))

    await act(async () => { render(<WorkflowEditor />) })

    await waitFor(() => {
      expect(screen.getByText('Workflow not found')).toBeInTheDocument()
    })
  })
})
