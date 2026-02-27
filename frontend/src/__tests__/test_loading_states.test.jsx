import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { http, HttpResponse } from 'msw'
import { server } from '../mocks/server.js'

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
}))

vi.mock('../lib/auth', () => ({
  isAuthenticated: () => true,
  getToken: () => 'fake-token',
  clearAuth: vi.fn(),
}))

// Use a hanging API mock so components stay in loading state
vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn(() => new Promise(() => {})),
    post: vi.fn(() => new Promise(() => {})),
    put: vi.fn(() => new Promise(() => {})),
    patch: vi.fn(() => new Promise(() => {})),
    delete: vi.fn(() => new Promise(() => {})),
    setToken: vi.fn(),
    clearToken: vi.fn(),
  },
}))

vi.mock('../hooks/useSkills.js', () => ({
  useSkills: () => ({ data: null, isLoading: true }),
  useSkill: () => ({ data: null, isLoading: true }),
  useUpdateSkill: () => ({ mutate: vi.fn() }),
  useDeleteSkill: () => ({ mutate: vi.fn() }),
  useDuplicateSkill: () => ({ mutate: vi.fn() }),
  useCreateSkill: () => ({ mutate: vi.fn() }),
  useSkillInvocations: () => ({ data: [], isLoading: true }),
}))

vi.mock('../hooks/useCredentials.js', () => ({
  useCredentials: () => ({ data: null, isLoading: true }),
  useCreateCredential: () => ({ mutate: vi.fn() }),
  useDeleteCredential: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useMCPServers.js', () => ({
  useMCPServers: () => ({ data: null, isLoading: true }),
  useAddMCPServer: () => ({ mutate: vi.fn() }),
  useRemoveMCPServer: () => ({ mutate: vi.fn() }),
  useMCPServerTools: () => ({ data: [], isLoading: true }),
  useReconnectMCPServer: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useExecutions.js', () => ({
  useExecutions: () => ({ data: null, isLoading: true }),
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

import api from '../lib/api.js'
import Workflows from '../pages/Workflows.jsx'
import Dashboard from '../pages/Dashboard.jsx'

// Polyfill scrollIntoView for jsdom (used by LiveFeed)
Element.prototype.scrollIntoView = vi.fn()

describe('Loading states', () => {
  it('LD1: Workflows page shows skeleton cards while loading', () => {
    render(<Workflows />)
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(3)
  })

  it('LD2: Dashboard shows dash placeholders while loading', () => {
    render(<Dashboard />)
    // Dashboard stat cards show '-' when loading
    const dashes = screen.getAllByText('-')
    expect(dashes.length).toBeGreaterThanOrEqual(4)
  })

  it('LD3: Workflows page transitions from loading to content', async () => {
    let resolveApi
    api.get.mockImplementation(() => new Promise((resolve) => { resolveApi = resolve }))

    render(<Workflows />)

    // Initially shows skeletons
    expect(document.querySelectorAll('.animate-pulse').length).toBe(3)

    // Resolve the API call
    await waitFor(() => expect(resolveApi).toBeDefined())
    resolveApi({ workflows: [{ id: 'wf-1', name: 'Loaded WF', status: 'draft', steps: [] }], total: 1 })

    await waitFor(() => {
      expect(screen.getByText('Loaded WF')).toBeInTheDocument()
    })
    expect(document.querySelectorAll('.animate-pulse').length).toBe(0)
  })

  it('LD4: Workflows page shows error state on failure', async () => {
    api.get.mockRejectedValue(new Error('Network error'))

    render(<Workflows />)

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeInTheDocument()
    })
  })

  it('LD5: Dashboard shows error on fetch failure', async () => {
    api.get.mockRejectedValue(new Error('Server down'))

    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText(/Server down/)).toBeInTheDocument()
    })
  })
})
