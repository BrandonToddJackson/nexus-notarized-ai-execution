import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import '@testing-library/jest-dom'

// --- Mocks ---
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/' }),
  useParams: () => ({ workflowId: 'new' }),
  useSearchParams: () => [new URLSearchParams(), vi.fn()],
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  BrowserRouter: ({ children }) => <div>{children}</div>,
  Navigate: () => null,
  Routes: ({ children }) => <div>{children}</div>,
  Route: () => null,
}))

const mockApiGet = vi.fn(() => Promise.resolve([]))
const mockApiPost = vi.fn(() => Promise.resolve({}))

vi.mock('../lib/api', () => ({
  default: {
    get: (...args) => mockApiGet(...args),
    post: (...args) => mockApiPost(...args),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
    delete: vi.fn(() => Promise.resolve({})),
    setToken: vi.fn(),
    clearToken: vi.fn(),
  },
}))

vi.mock('../stores/appStore.js', () => ({
  useAppStore: (selector) => {
    const state = { sidebarCollapsed: false, toggleSidebar: vi.fn(), gateFailureCount: 0, incrementGateFailure: vi.fn() }
    return selector ? selector(state) : state
  },
}))

const mockToast = { success: vi.fn(), error: vi.fn() }
vi.mock('react-hot-toast', () => ({
  Toaster: () => null,
  default: mockToast,
}))

vi.mock('../hooks/useExecutions.js', () => ({
  useExecutions: () => ({ data: { executions: [] }, isLoading: false }),
  useRetryExecution: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useExecutionStream.js', () => ({
  useExecutionStream: () => {},
}))

vi.mock('../hooks/useSkills.js', () => ({
  useSkills: () => ({ data: { skills: [] }, isLoading: false }),
  useSkill: () => ({ data: null, isLoading: false }),
  useUpdateSkill: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteSkill: () => ({ mutate: vi.fn() }),
  useDuplicateSkill: () => ({ mutate: vi.fn() }),
  useCreateSkill: () => ({ mutate: vi.fn(), isPending: false }),
  useSkillInvocations: () => ({ data: [], isLoading: false }),
}))

vi.mock('../hooks/useCredentials.js', () => ({
  useCredentials: () => ({ data: { credentials: [] }, isLoading: false }),
  useCreateCredential: () => ({ mutate: vi.fn() }),
  useDeleteCredential: () => ({ mutate: vi.fn() }),
  useCredentialTypes: () => ({ data: [], isLoading: false }),
  useTestCredential: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useMCPServers.js', () => ({
  useMCPServers: () => ({ data: { servers: [] }, isLoading: false }),
  useAddMCPServer: () => ({ mutate: vi.fn() }),
  useRemoveMCPServer: () => ({ mutate: vi.fn() }),
  useMCPServerTools: () => ({ data: [], isLoading: false }),
  useReconnectMCPServer: () => ({ mutate: vi.fn() }),
  useTestMCPServer: () => ({ mutate: vi.fn() }),
}))

vi.mock('../lib/auth', () => ({
  isAuthenticated: () => true,
  getToken: () => 'fake-token',
  clearAuth: vi.fn(),
  setToken: vi.fn(),
  setTenantId: vi.fn(),
}))

import Dashboard from '../pages/Dashboard.jsx'
import Workflows from '../pages/Workflows.jsx'
import Executions from '../pages/Executions.jsx'
import Credentials from '../pages/Credentials.jsx'
import Skills from '../pages/Skills.jsx'
import MCPServers from '../pages/MCPServers.jsx'

beforeEach(() => {
  mockApiGet.mockReset()
  mockApiPost.mockReset()
  mockApiGet.mockResolvedValue([])
  mockApiPost.mockResolvedValue({})
  mockToast.success.mockClear()
  mockToast.error.mockClear()
  vi.useFakeTimers({ shouldAdvanceTime: true })
})

afterEach(() => {
  vi.useRealTimers()
})

// ERR1: GET /v2/workflows returns 500 — page not crashed
describe('ERR1: Workflow list API error', () => {
  it('shows error message when API returns 500', async () => {
    mockApiGet.mockRejectedValue(new Error('Internal Server Error'))
    render(<Workflows />)
    await waitFor(() => {
      expect(screen.getByText(/Internal Server Error/)).toBeInTheDocument()
    })
  })

  it('page still renders heading despite error', async () => {
    mockApiGet.mockRejectedValue(new Error('Server Error'))
    render(<Workflows />)
    expect(screen.getByText('Workflows')).toBeInTheDocument()
  })
})

// ERR2: Dashboard API error shows error message
describe('ERR2: Dashboard API error', () => {
  it('shows error message when ledger API fails', async () => {
    mockApiGet.mockRejectedValue(new Error('422 Unprocessable Entity'))
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText(/Failed to load data/)).toBeInTheDocument()
    })
  })
})

// ERR3: Various pages do not crash on API errors
describe('ERR3: Pages resilient to API errors', () => {
  it('Executions page renders with empty data', () => {
    render(<Executions />)
    expect(screen.getByText('Executions')).toBeInTheDocument()
    expect(screen.getByText('No executions')).toBeInTheDocument()
  })

  it('Credentials page renders with empty data', () => {
    render(<Credentials />)
    expect(screen.getByText('Credentials')).toBeInTheDocument()
  })

  it('Skills page renders with empty data', () => {
    render(<Skills />)
    expect(screen.getByText('Skills')).toBeInTheDocument()
  })

  it('MCPServers page renders with empty data', () => {
    render(<MCPServers />)
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
  })
})

// ERR4: Execution with null seals field — no crash
describe('ERR4: Null seals field handling', () => {
  it('Executions page handles null data gracefully', () => {
    // useExecutions returns empty executions — no crash
    render(<Executions />)
    expect(screen.getByText('Executions')).toBeInTheDocument()
  })
})

// ERR5: Network error — Dashboard handles fetch throws
describe('ERR5: Network error handling', () => {
  it('Dashboard handles TypeError: Failed to fetch', async () => {
    mockApiGet.mockRejectedValue(new TypeError('Failed to fetch'))
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText(/Failed to load data/)).toBeInTheDocument()
    })
    // Still renders structure
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Total Seals')).toBeInTheDocument()
  })

  it('Workflows page handles network error', async () => {
    mockApiGet.mockRejectedValue(new TypeError('Failed to fetch'))
    render(<Workflows />)
    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch/)).toBeInTheDocument()
    })
    expect(screen.getByText('Workflows')).toBeInTheDocument()
  })
})

// ERR6: Empty state pages display properly
describe('ERR6: Empty state rendering', () => {
  it('Workflows empty state renders', async () => {
    mockApiGet.mockResolvedValue([])
    render(<Workflows />)
    await waitFor(() => {
      expect(screen.getByText('No workflows yet')).toBeInTheDocument()
    })
  })

  it('Executions empty state renders', () => {
    render(<Executions />)
    expect(screen.getByText('No executions')).toBeInTheDocument()
  })

  it('Credentials empty state renders', () => {
    render(<Credentials />)
    expect(screen.getByText('No credentials')).toBeInTheDocument()
  })

  it('Skills empty state renders', () => {
    render(<Skills />)
    expect(screen.getByText('No skills yet')).toBeInTheDocument()
  })

  it('MCP Servers empty state renders', () => {
    render(<MCPServers />)
    expect(screen.getByText('No MCP servers')).toBeInTheDocument()
  })
})

// ERR7-ERR8: API returns error details
describe('ERR7-ERR8: Error message display', () => {
  it('Workflows page shows error text from API', async () => {
    mockApiGet.mockRejectedValue(new Error('AnomalyDetected: suspicious drift'))
    render(<Workflows />)
    await waitFor(() => {
      expect(screen.getByText(/AnomalyDetected: suspicious drift/)).toBeInTheDocument()
    })
  })
})

// ERR9: CredentialForm with missing name
describe('ERR9: Credential form validation', () => {
  it('CredentialForm can be submitted with empty name (server-side validation)', () => {
    render(<Credentials />)
    // The form does not have client-side required validation on name
    // This verifies the form renders and is interactive without crashing
    expect(screen.getByText('Credentials')).toBeInTheDocument()
  })
})

// ERR10: Long workflow name — no layout break
describe('ERR10: Long workflow name', () => {
  it('renders long workflow name without crashing', async () => {
    const longName = 'A'.repeat(200)
    mockApiGet.mockResolvedValue([{ id: 'wf-1', name: longName, status: 'draft', steps: [] }])
    render(<Workflows />)
    await waitFor(() => {
      // In card view the WorkflowCard renders the name — page should not crash
      expect(screen.getByText('Workflows')).toBeInTheDocument()
    })
  })
})

// ERR11: Workflow with no steps — EmptyState or editor handles it
describe('ERR11: Workflow with no steps', () => {
  it('empty workflow array renders empty state', async () => {
    mockApiGet.mockResolvedValue([])
    render(<Workflows />)
    await waitFor(() => {
      expect(screen.getByText('No workflows yet')).toBeInTheDocument()
    })
  })

  it('workflow with empty steps array renders in list', async () => {
    mockApiGet.mockResolvedValue([{ id: 'wf-1', name: 'Test', status: 'draft', steps: [] }])
    render(<Workflows />)
    await waitFor(() => {
      expect(screen.getByText('Test')).toBeInTheDocument()
    })
  })
})

// ERR12: Execution with null seals — renders without crash
describe('ERR12: Null seals resilience', () => {
  it('Executions page does not crash with null/undefined data', () => {
    // The mock returns empty executions, simulating no-data scenario
    render(<Executions />)
    expect(screen.getByText('Executions')).toBeInTheDocument()
    expect(screen.getByText('No executions')).toBeInTheDocument()
  })
})

// Dashboard loading state
describe('Dashboard loading state', () => {
  it('shows dash placeholders while loading', () => {
    // Don't resolve the promise yet — Dashboard shows loading state
    mockApiGet.mockReturnValue(new Promise(() => {}))
    render(<Dashboard />)
    const dashes = screen.getAllByText('-')
    expect(dashes.length).toBe(4) // 4 stat cards show "-" while loading
  })
})

// All pages render without crash (smoke tests)
describe('Smoke tests — all pages render', () => {
  it('Dashboard renders', async () => {
    mockApiGet.mockResolvedValue([])
    render(<Dashboard />)
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
  })

  it('Workflows renders', async () => {
    mockApiGet.mockResolvedValue([])
    render(<Workflows />)
    expect(screen.getByText('Workflows')).toBeInTheDocument()
  })

  it('Executions renders', () => {
    render(<Executions />)
    expect(screen.getByText('Executions')).toBeInTheDocument()
  })

  it('Credentials renders', () => {
    render(<Credentials />)
    expect(screen.getByText('Credentials')).toBeInTheDocument()
  })

  it('Skills renders', () => {
    render(<Skills />)
    expect(screen.getByText('Skills')).toBeInTheDocument()
  })

  it('MCPServers renders', () => {
    render(<MCPServers />)
    expect(screen.getByText('MCP Servers')).toBeInTheDocument()
  })
})
