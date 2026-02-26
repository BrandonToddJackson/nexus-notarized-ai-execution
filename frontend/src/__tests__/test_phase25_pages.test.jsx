import { vi, describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'

// Mock react-router-dom
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/' }),
  useParams: () => ({}),
  useSearchParams: () => [new URLSearchParams(), vi.fn()],
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  Navigate: () => null,
  BrowserRouter: ({ children }) => <div>{children}</div>,
  Routes: ({ children }) => <div>{children}</div>,
  Route: () => null,
}))

// Mock tanstack/react-query
vi.mock('@tanstack/react-query', () => ({
  QueryClient: vi.fn(() => ({})),
  QueryClientProvider: ({ children }) => <div>{children}</div>,
  useQuery: () => ({ data: null, isLoading: false }),
  useMutation: () => ({ mutate: vi.fn() }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}))

// Mock hooks
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
    clearToken: vi.fn(),
  },
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

import Skills from '../pages/Skills.jsx'
import Credentials from '../pages/Credentials.jsx'
import MCPServers from '../pages/MCPServers.jsx'
import Executions from '../pages/Executions.jsx'
import { GateBar } from '../components/gates/GateBar.jsx'
import { GateChip } from '../components/gates/GateChip.jsx'
import { Sidebar } from '../components/layout/Sidebar.jsx'

describe('Skills page', () => {
  it('renders Skills heading', () => {
    render(<Skills />)
    expect(screen.getByText('Skills')).toBeInTheDocument()
  })

  it('renders New Skill button', () => {
    render(<Skills />)
    // Header button + empty state action both show "New Skill"
    const buttons = screen.getAllByText('New Skill')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('renders SearchInput', () => {
    render(<Skills />)
    expect(screen.getByPlaceholderText('Search skills...')).toBeInTheDocument()
  })

  it('shows EmptyState when skills array is empty', () => {
    render(<Skills />)
    expect(screen.getByText('No skills yet')).toBeInTheDocument()
  })
})

describe('Credentials page', () => {
  it('renders Credentials heading', () => {
    render(<Credentials />)
    expect(screen.getByText('Credentials')).toBeInTheDocument()
  })

  it('renders Add Credential button', () => {
    render(<Credentials />)
    // Header button + empty state action both show "Add Credential"
    const buttons = screen.getAllByText('Add Credential')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('shows EmptyState when no credentials', () => {
    render(<Credentials />)
    expect(screen.getByText('No credentials')).toBeInTheDocument()
  })
})

describe('MCPServers page', () => {
  it('renders MCP Servers heading', () => {
    render(<MCPServers />)
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
  })

  it('renders Add Server button in header', () => {
    render(<MCPServers />)
    // There are two "Add Server" texts — header button and empty state action
    const buttons = screen.getAllByText('Add Server')
    expect(buttons.length).toBeGreaterThanOrEqual(1)
  })

  it('shows EmptyState when no servers', () => {
    render(<MCPServers />)
    expect(screen.getByText('No MCP servers')).toBeInTheDocument()
  })
})

describe('Executions page', () => {
  it('renders Executions heading', () => {
    render(<Executions />)
    expect(screen.getByText('Executions')).toBeInTheDocument()
  })

  it('renders status filter', () => {
    render(<Executions />)
    expect(screen.getByText('All statuses')).toBeInTheDocument()
  })

  it('renders Has gate failures checkbox', () => {
    render(<Executions />)
    expect(screen.getByText('Has gate failures')).toBeInTheDocument()
  })

  it('shows EmptyState when no executions', () => {
    render(<Executions />)
    expect(screen.getByText('No executions')).toBeInTheDocument()
  })
})

describe('GateBar', () => {
  it('renders 4 GateChip elements for 4 gates', () => {
    const gates = [
      { gate: 'scope', result: 'pass' },
      { gate: 'intent', result: 'pass' },
      { gate: 'ttl', result: 'pass' },
      { gate: 'drift', result: 'fail' },
    ]
    const { container } = render(<GateBar gates={gates} />)
    const chips = container.querySelectorAll('span')
    expect(chips.length).toBe(4)
  })

  it('returns null for empty gates', () => {
    const { container } = render(<GateBar gates={[]} />)
    expect(container.innerHTML).toBe('')
  })
})

describe('GateChip', () => {
  it('renders pass indicator for passing gate', () => {
    render(<GateChip gate="scope" result="pass" />)
    expect(screen.getByText(/G1 Scope/)).toBeInTheDocument()
    expect(screen.getByText(/\u2713/)).toBeInTheDocument()
  })

  it('renders fail indicator for failing gate', () => {
    render(<GateChip gate="scope" result="fail" />)
    expect(screen.getByText(/G1 Scope/)).toBeInTheDocument()
    expect(screen.getByText(/\u2717/)).toBeInTheDocument()
  })

  it('shows value and threshold when provided', () => {
    render(<GateChip gate="intent" result="pass" value={0.85} threshold={0.75} />)
    expect(screen.getByText(/0.85/)).toBeInTheDocument()
    expect(screen.getByText(/0.75/)).toBeInTheDocument()
  })
})

describe('Sidebar', () => {
  it('renders NEXUS brand', () => {
    render(<Sidebar />)
    expect(screen.getByText('NEXUS')).toBeInTheDocument()
  })

  it('renders all nav section items', () => {
    render(<Sidebar />)
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Workflows')).toBeInTheDocument()
    expect(screen.getByText('Executions')).toBeInTheDocument()
    expect(screen.getByText('Credentials')).toBeInTheDocument()
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
    expect(screen.getByText('Skills')).toBeInTheDocument()
    expect(screen.getByText('Tools')).toBeInTheDocument()
    expect(screen.getByText('Personas')).toBeInTheDocument()
    expect(screen.getByText('Knowledge')).toBeInTheDocument()
  })

  it('renders section headers', () => {
    render(<Sidebar />)
    expect(screen.getByText('Automation')).toBeInTheDocument()
    expect(screen.getByText('Connections')).toBeInTheDocument()
    expect(screen.getByText('Build')).toBeInTheDocument()
  })
})
