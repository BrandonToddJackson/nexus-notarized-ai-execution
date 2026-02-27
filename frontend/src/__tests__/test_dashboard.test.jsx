import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'

// --- Mocks ---
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/' }),
  useParams: () => ({}),
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  BrowserRouter: ({ children }) => <div>{children}</div>,
}))

const mockSeals = [
  { id: 's1', chain_id: 'chain-1', status: 'executed', cost_usd: 0.05, created_at: '2026-02-26T10:00:00Z' },
  { id: 's2', chain_id: 'chain-1', status: 'executed', cost_usd: 0.03, created_at: '2026-02-26T10:01:00Z' },
  { id: 's3', chain_id: 'chain-2', status: 'blocked', cost_usd: 0.02, created_at: '2026-02-26T10:02:00Z' },
  { id: 's4', chain_id: 'chain-3', status: 'executed', cost_usd: 0.10, created_at: '2026-02-26T09:00:00Z' },
]

let mockApiGetResponse = mockSeals

vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn(() => Promise.resolve(mockApiGetResponse)),
    post: vi.fn(() => Promise.resolve({})),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
    delete: vi.fn(() => Promise.resolve({})),
    setToken: vi.fn(),
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

import Dashboard from '../pages/Dashboard.jsx'

beforeEach(() => {
  mockApiGetResponse = mockSeals
  vi.useFakeTimers({ shouldAdvanceTime: true })
  mockNavigate.mockClear()
})

afterEach(() => {
  vi.useRealTimers()
})

// DB1: Stats computed correctly
describe('DB1: Stats computation', () => {
  it('renders Dashboard heading', async () => {
    render(<Dashboard />)
    expect(screen.getByText('Dashboard')).toBeInTheDocument()
  })

  it('shows total seals count after load', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      // '4' appears in both the stat card and a chain badge — getAllByText is correct here
      expect(screen.getAllByText('4').length).toBeGreaterThanOrEqual(1)
    })
    // label
    expect(screen.getByText('Total Seals')).toBeInTheDocument()
  })

  it('shows total chains count after load', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument()
    })
    expect(screen.getByText('Total Chains')).toBeInTheDocument()
  })

  it('computes blocked percentage correctly', async () => {
    render(<Dashboard />)
    // 1 out of 4 blocked = 25.0%
    await waitFor(() => {
      expect(screen.getByText('25.0%')).toBeInTheDocument()
    })
    expect(screen.getByText('Blocked %')).toBeInTheDocument()
  })

  it('computes LLM cost correctly', async () => {
    render(<Dashboard />)
    // 0.05 + 0.03 + 0.02 + 0.10 = 0.20
    await waitFor(() => {
      expect(screen.getByText('$0.20')).toBeInTheDocument()
    })
    expect(screen.getByText('LLM Cost')).toBeInTheDocument()
  })
})

// DB2: LiveFeed/recent executions renders
describe('DB2: LiveFeed renders', () => {
  it('renders Live Feed section heading', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      // Dashboard h2 + LiveFeed h3 both say "Live Feed" — getAllByText handles both
      expect(screen.getAllByText('Live Feed').length).toBeGreaterThanOrEqual(1)
    })
  })

  it('renders Recent Chains section heading', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('Recent Chains')).toBeInTheDocument()
    })
  })

  it('shows chain IDs truncated', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      // chain_id.slice(0, 8) = 'chain-1' (7 chars, fits), 'chain-2', 'chain-3'
      expect(screen.getByText('chain-1')).toBeInTheDocument()
      expect(screen.getByText('chain-2')).toBeInTheDocument()
      expect(screen.getByText('chain-3')).toBeInTheDocument()
    })
  })

  it('shows seal count for chains', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('2 seals')).toBeInTheDocument() // chain-1 has 2 seals
      expect(screen.getAllByText('1 seal').length).toBe(2) // chain-2 and chain-3 each have 1
    })
  })
})

// DB3: Gate failure / blocked chain highlight
describe('DB3: Blocked chain indication', () => {
  it('shows "blocked" badge for chain with blocked seal', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('blocked')).toBeInTheDocument()
    })
  })

  it('shows "executed" badge for non-blocked chains', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getAllByText('executed').length).toBe(2)
    })
  })
})

// DB4: Navigation links
describe('DB4: Navigation', () => {
  it('recent chain buttons navigate to /ledger on click', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('chain-1')).toBeInTheDocument()
    })
    const chainBtn = screen.getByText('chain-1').closest('button')
    chainBtn.click()
    expect(mockNavigate).toHaveBeenCalledWith('/ledger')
  })
})

// DB5: Empty dashboard — no seals
describe('DB5: Empty dashboard', () => {
  it('shows zeros when no seals', async () => {
    mockApiGetResponse = []
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('0')).toBeInTheDocument() // totalSeals or totalChains
    })
    expect(screen.getByText('0.0%')).toBeInTheDocument()
    expect(screen.getByText('$0.00')).toBeInTheDocument()
  })

  it('shows "No chains recorded yet." when empty', async () => {
    mockApiGetResponse = []
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('No chains recorded yet.')).toBeInTheDocument()
    })
  })

  it('does not crash with empty array response', async () => {
    mockApiGetResponse = []
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument()
    })
  })
})

// Error handling
describe('Error state', () => {
  it('shows error message when API fails', async () => {
    const api = (await import('../lib/api')).default
    api.get.mockRejectedValueOnce(new Error('Network error'))
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText(/Failed to load data/)).toBeInTheDocument()
    })
  })
})

// Stat card labels
describe('Stat card labels', () => {
  it('renders all four stat labels', async () => {
    render(<Dashboard />)
    await waitFor(() => {
      expect(screen.getByText('Total Chains')).toBeInTheDocument()
    })
    expect(screen.getByText('Total Seals')).toBeInTheDocument()
    expect(screen.getByText('Blocked %')).toBeInTheDocument()
    expect(screen.getByText('LLM Cost')).toBeInTheDocument()
  })
})
