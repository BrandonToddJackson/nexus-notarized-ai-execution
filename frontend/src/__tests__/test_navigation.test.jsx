import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

// Mock react-router-dom
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/' }),
  useParams: () => ({}),
  useSearchParams: () => [new URLSearchParams(), vi.fn()],
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  Navigate: ({ to }) => <div data-testid="navigate" data-to={to} />,
  BrowserRouter: ({ children }) => <div>{children}</div>,
  Routes: ({ children }) => <div>{children}</div>,
  Route: () => null,
}))

vi.mock('@tanstack/react-query', () => ({
  QueryClient: vi.fn(() => ({})),
  QueryClientProvider: ({ children }) => <div>{children}</div>,
  useQuery: () => ({ data: null, isLoading: false }),
  useMutation: () => ({ mutate: vi.fn() }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}))

vi.mock('../lib/auth', () => ({
  isAuthenticated: () => true,
  getToken: () => 'fake-token',
  clearAuth: vi.fn(),
}))

vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn(() => Promise.resolve([])),
    post: vi.fn(() => Promise.resolve({})),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
    delete: vi.fn(() => Promise.resolve({})),
    setToken: vi.fn(),
    clearToken: vi.fn(),
  },
}))

vi.mock('../hooks/useSkills.js', () => ({
  useSkills: () => ({ data: { skills: [] }, isLoading: false }),
  useSkill: () => ({ data: null, isLoading: false }),
  useUpdateSkill: () => ({ mutate: vi.fn() }),
  useDeleteSkill: () => ({ mutate: vi.fn() }),
  useDuplicateSkill: () => ({ mutate: vi.fn() }),
  useCreateSkill: () => ({ mutate: vi.fn() }),
  useSkillInvocations: () => ({ data: [], isLoading: false }),
}))

vi.mock('../hooks/useCredentials.js', () => ({
  useCredentials: () => ({ data: { credentials: [] }, isLoading: false }),
  useCreateCredential: () => ({ mutate: vi.fn() }),
  useDeleteCredential: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useMCPServers.js', () => ({
  useMCPServers: () => ({ data: { servers: [] }, isLoading: false }),
  useAddMCPServer: () => ({ mutate: vi.fn() }),
  useRemoveMCPServer: () => ({ mutate: vi.fn() }),
  useMCPServerTools: () => ({ data: [], isLoading: false }),
  useReconnectMCPServer: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useExecutions.js', () => ({
  useExecutions: () => ({ data: { executions: [] }, isLoading: false }),
}))

vi.mock('../hooks/useExecutionStream.js', () => ({
  useExecutionStream: () => {},
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

import { Sidebar } from '../components/layout/Sidebar.jsx'

beforeEach(() => {
  mockNavigate.mockReset()
})

describe('Navigation', () => {
  it('NAV1: Sidebar renders all primary nav items', () => {
    render(<Sidebar />)
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Workflows')).toBeInTheDocument()
    expect(screen.getByText('Executions')).toBeInTheDocument()
  })

  it('NAV2: Sidebar renders connection nav items', () => {
    render(<Sidebar />)
    expect(screen.getByText('Credentials')).toBeInTheDocument()
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
  })

  it('NAV3: Sidebar renders build section items', () => {
    render(<Sidebar />)
    expect(screen.getByText('Skills')).toBeInTheDocument()
    expect(screen.getByText('Tools')).toBeInTheDocument()
    expect(screen.getByText('Personas')).toBeInTheDocument()
    expect(screen.getByText('Knowledge')).toBeInTheDocument()
  })

  it('NAV4: Sidebar renders section headers', () => {
    render(<Sidebar />)
    expect(screen.getByText('Automation')).toBeInTheDocument()
    expect(screen.getByText('Connections')).toBeInTheDocument()
    expect(screen.getByText('Build')).toBeInTheDocument()
  })

  it('NAV5: Sidebar renders NEXUS brand', () => {
    render(<Sidebar />)
    expect(screen.getByText('NEXUS')).toBeInTheDocument()
  })

  it('NAV6: Sidebar nav items are rendered as links', () => {
    render(<Sidebar />)
    // With our Link mock, nav items render as <a> tags
    const links = screen.getAllByRole('link')
    expect(links.length).toBeGreaterThan(0)

    // Check some expected href values
    const hrefs = links.map(l => l.getAttribute('href'))
    expect(hrefs).toContain('/')
    expect(hrefs).toContain('/workflows')
    expect(hrefs).toContain('/executions')
  })

  it('NAV7: Sidebar credentials link points to /credentials', () => {
    render(<Sidebar />)
    const credLink = screen.getByText('Credentials').closest('a')
    expect(credLink).toHaveAttribute('href', '/credentials')
  })

  it('NAV8: Sidebar MCP Servers link points to /mcp-servers', () => {
    render(<Sidebar />)
    const mcpLink = screen.getByText('MCP Servers').closest('a')
    expect(mcpLink).toHaveAttribute('href', '/mcp-servers')
  })
})
