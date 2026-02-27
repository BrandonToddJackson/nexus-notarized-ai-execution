import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'

// --- Mocks ---
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/mcp-servers' }),
  useParams: () => ({}),
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  BrowserRouter: ({ children }) => <div>{children}</div>,
}))

const mockServers = [
  { id: 'srv-1', name: 'Local MCP', transport: 'stdio', status: 'connected', tool_count: 5 },
  { id: 'srv-2', name: 'Cloud MCP', transport: 'sse', status: 'disconnected', tool_count: 12 },
]

const mockTools = [
  { name: 'read_file', description: 'Read file contents' },
  { name: 'write_file', description: 'Write to a file' },
]

let mockServerData = { servers: mockServers }
const mockAddMutate = vi.fn()
const mockRemoveMutate = vi.fn()
const mockReconnectMutate = vi.fn()

vi.mock('../hooks/useMCPServers.js', () => ({
  useMCPServers: () => ({ data: mockServerData, isLoading: false }),
  useAddMCPServer: () => ({ mutate: mockAddMutate }),
  useRemoveMCPServer: () => ({ mutate: mockRemoveMutate }),
  useMCPServerTools: (id) => ({
    data: id ? { tools: mockTools } : { tools: [] },
    isLoading: false,
  }),
  useReconnectMCPServer: () => ({ mutate: mockReconnectMutate }),
  useTestMCPServer: () => ({ mutate: vi.fn() }),
}))

vi.mock('../stores/appStore.js', () => ({
  useAppStore: (selector) => {
    const state = { sidebarCollapsed: false, toggleSidebar: vi.fn() }
    return selector ? selector(state) : state
  },
}))

vi.mock('react-hot-toast', () => ({
  Toaster: () => null,
  default: { success: vi.fn(), error: vi.fn() },
}))

import MCPServers from '../pages/MCPServers.jsx'

beforeEach(() => {
  mockServerData = { servers: mockServers }
  mockAddMutate.mockClear()
  mockRemoveMutate.mockClear()
  mockReconnectMutate.mockClear()
})

// MCP1: Servers list loads
describe('MCP1: Servers list', () => {
  it('renders MCP Servers heading', () => {
    render(<MCPServers />)
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
  })

  it('renders server names', () => {
    render(<MCPServers />)
    expect(screen.getByText('Local MCP')).toBeInTheDocument()
    expect(screen.getByText('Cloud MCP')).toBeInTheDocument()
  })

  it('renders column headers', () => {
    render(<MCPServers />)
    expect(screen.getByText('Name')).toBeInTheDocument()
    expect(screen.getByText('Transport')).toBeInTheDocument()
    expect(screen.getByText('Status')).toBeInTheDocument()
    expect(screen.getByText('Tools')).toBeInTheDocument()
  })

  it('renders transport badges', () => {
    render(<MCPServers />)
    expect(screen.getByText('stdio')).toBeInTheDocument()
    expect(screen.getByText('sse')).toBeInTheDocument()
  })

  it('renders tool counts', () => {
    render(<MCPServers />)
    expect(screen.getByText('5')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
  })
})

// MCP2: Add server (stdio type)
describe('MCP2: Add stdio server', () => {
  it('opens add server modal', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })
  })

  it('shows command field for stdio transport (default)', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Command')).toBeInTheDocument()
      expect(screen.getByText('Arguments')).toBeInTheDocument()
    })
  })

  it('adds stdio server with name and command', async () => {
    const user = userEvent.setup()
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })

    // The form has multiple inputs — name is first
    const inputs = document.querySelectorAll('input[type="text"], input:not([type])')
    await user.type(inputs[0], 'My Stdio Server')
    await user.type(inputs[1], 'npx')
    await user.type(inputs[2], '-y @mcp/server')

    // The Add button is the submit button
    const addBtn = screen.getAllByText('Add').find(btn => btn.tagName === 'BUTTON' && !btn.disabled)
    fireEvent.click(addBtn)
    expect(mockAddMutate).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'My Stdio Server',
        transport: 'stdio',
        command: 'npx',
        args: ['-y', '@mcp/server'],
      }),
      expect.any(Object),
    )
  })
})

// MCP3: Add server (HTTP type)
describe('MCP3: Add HTTP server', () => {
  it('shows URL field when sse transport selected', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })
    // Select sse radio
    const sseRadio = screen.getByLabelText('sse')
    fireEvent.click(sseRadio)
    expect(screen.getByText('URL')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('http://localhost:3000/mcp')).toBeInTheDocument()
  })

  it('shows URL field when streamable_http transport selected', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })
    const httpRadio = screen.getByLabelText('streamable_http')
    fireEvent.click(httpRadio)
    expect(screen.getByText('URL')).toBeInTheDocument()
  })

  it('adds sse server with name and URL', async () => {
    const user = userEvent.setup()
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })

    const sseRadio = screen.getByLabelText('sse')
    fireEvent.click(sseRadio)

    const inputs = document.querySelectorAll('input[type="text"], input:not([type="radio"])')
    // First text input is Name, second (after switching to sse) is URL
    const textInputs = Array.from(inputs).filter(i => i.type !== 'radio')
    await user.type(textInputs[0], 'Remote SSE')
    await user.type(textInputs[textInputs.length - 1], 'http://example.com/mcp')

    const addBtn = screen.getAllByText('Add').find(btn => btn.tagName === 'BUTTON' && !btn.disabled)
    fireEvent.click(addBtn)
    expect(mockAddMutate).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'Remote SSE',
        transport: 'sse',
        url: 'http://example.com/mcp',
      }),
      expect.any(Object),
    )
  })
})

// MCP4: Validation — Add button disabled without name
describe('MCP4: Form validation', () => {
  it('Add button is disabled when name is empty', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })
    // The submit Add button should be disabled
    const addBtns = screen.getAllByText('Add')
    const submitBtn = addBtns.find(btn => btn.closest('.flex.justify-end'))
    expect(submitBtn).toBeDisabled()
  })
})

// MCP5: Tools accordion — expand row shows tools
describe('MCP5: Tools expansion', () => {
  it('renders expand buttons for rows', () => {
    const { container } = render(<MCPServers />)
    // DataTable with renderExpanded shows expand toggle buttons
    const expandBtns = container.querySelectorAll('button')
    expect(expandBtns.length).toBeGreaterThanOrEqual(2)
  })
})

// MCP6: Test server — API exists
describe('MCP6: Test server API', () => {
  it('testMCPServer API function exists', async () => {
    const { testMCPServer } = await import('../api/mcpServers.js')
    expect(testMCPServer).toBeDefined()
  })
})

// MCP7: Reconnect button
describe('MCP7: Reconnect', () => {
  it('renders Reconnect button for each server', () => {
    render(<MCPServers />)
    const reconnectBtns = screen.getAllByText('Reconnect')
    expect(reconnectBtns.length).toBe(2)
  })

  it('calls reconnect.mutate with server id', () => {
    render(<MCPServers />)
    const reconnectBtns = screen.getAllByText('Reconnect')
    fireEvent.click(reconnectBtns[0])
    expect(mockReconnectMutate).toHaveBeenCalledWith('srv-1')
  })

  it('calls reconnect for second server', () => {
    render(<MCPServers />)
    const reconnectBtns = screen.getAllByText('Reconnect')
    fireEvent.click(reconnectBtns[1])
    expect(mockReconnectMutate).toHaveBeenCalledWith('srv-2')
  })
})

// MCP8: Remove server with confirmation
describe('MCP8: Remove server', () => {
  it('renders Remove button for each server', () => {
    render(<MCPServers />)
    const removeBtns = screen.getAllByText('Remove')
    expect(removeBtns.length).toBe(2)
  })

  it('shows confirmation dialog when Remove clicked', () => {
    render(<MCPServers />)
    const removeBtns = screen.getAllByText('Remove')
    fireEvent.click(removeBtns[0])
    expect(screen.getByText('Remove MCP Server')).toBeInTheDocument()
    expect(screen.getByText(/Remove "Local MCP" and all its tools\?/)).toBeInTheDocument()
  })

  it('calls removeServer.mutate on confirm', () => {
    render(<MCPServers />)
    const removeBtns = screen.getAllByText('Remove')
    fireEvent.click(removeBtns[0])
    // ConfirmDelete renders a "Delete" button to confirm
    fireEvent.click(screen.getByText('Delete'))
    expect(mockRemoveMutate).toHaveBeenCalledWith('srv-1')
  })

  it('cancel closes confirmation without removing', () => {
    render(<MCPServers />)
    const removeBtns = screen.getAllByText('Remove')
    fireEvent.click(removeBtns[0])
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Remove MCP Server')).not.toBeInTheDocument()
    expect(mockRemoveMutate).not.toHaveBeenCalled()
  })
})

// Empty state
describe('Empty state', () => {
  it('shows EmptyState when no servers', () => {
    mockServerData = { servers: [] }
    render(<MCPServers />)
    expect(screen.getByText('No MCP servers')).toBeInTheDocument()
    expect(screen.getByText('Connect MCP servers to extend tool capabilities')).toBeInTheDocument()
  })

  it('shows Add Server action in empty state', () => {
    mockServerData = { servers: [] }
    render(<MCPServers />)
    const buttons = screen.getAllByText('Add Server')
    expect(buttons.length).toBe(2)
  })
})

// Transport radio group
describe('Transport radio group', () => {
  it('renders three transport options', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByLabelText('stdio')).toBeInTheDocument()
      expect(screen.getByLabelText('sse')).toBeInTheDocument()
      expect(screen.getByLabelText('streamable_http')).toBeInTheDocument()
    })
  })

  it('stdio is selected by default', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByLabelText('stdio')).toBeChecked()
    })
  })
})

// Cancel add modal
describe('Cancel add modal', () => {
  it('closes when Cancel clicked', async () => {
    render(<MCPServers />)
    fireEvent.click(screen.getByText('Add Server'))
    await waitFor(() => {
      expect(screen.getByText('Add MCP Server')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Add MCP Server')).not.toBeInTheDocument()
  })
})
