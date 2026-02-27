import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../mocks/server.js'
import { fixtures } from '../mocks/data.js'

// Mock react-router-dom
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/workflows' }),
  useParams: () => ({}),
  useSearchParams: () => [new URLSearchParams(), vi.fn()],
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  Navigate: () => null,
  BrowserRouter: ({ children }) => <div>{children}</div>,
}))

vi.mock('../lib/api', () => ({
  default: {
    get: vi.fn((path) => {
      if (path === '/v2/workflows') {
        return fetch('/v2/workflows').then(r => r.json())
      }
      return Promise.resolve([])
    }),
    post: vi.fn(() => Promise.resolve({})),
    put: vi.fn(() => Promise.resolve({})),
    patch: vi.fn(() => Promise.resolve({})),
    delete: vi.fn((path) => {
      return fetch(path, { method: 'DELETE' }).then(r => r.json())
    }),
    setToken: vi.fn(),
    clearToken: vi.fn(),
  },
}))

vi.mock('react-hot-toast', () => ({
  Toaster: () => null,
  default: { success: vi.fn(), error: vi.fn() },
}))

import Workflows from '../pages/Workflows.jsx'

beforeEach(() => {
  mockNavigate.mockReset()
})

describe('Workflow list page', () => {
  it('WL1: Empty state renders when no workflows', async () => {
    server.use(
      http.get('/v2/workflows', () => {
        return HttpResponse.json({ workflows: [], total: 0 })
      })
    )

    const api = (await import('../lib/api')).default
    api.get.mockImplementation(() =>
      fetch('/v2/workflows').then(r => r.json())
    )

    render(<Workflows />)

    await waitFor(() => {
      expect(screen.getByText('No workflows yet')).toBeInTheDocument()
    })
  })

  it('WL2: Workflows load and display names', async () => {
    const api = (await import('../lib/api')).default
    api.get.mockImplementation(() =>
      fetch('/v2/workflows').then(r => r.json())
    )

    render(<Workflows />)

    await waitFor(() => {
      expect(screen.getByText('Test Workflow')).toBeInTheDocument()
    })
  })

  it('WL3: View toggle between card and table', async () => {
    const api = (await import('../lib/api')).default
    api.get.mockImplementation(() =>
      fetch('/v2/workflows').then(r => r.json())
    )

    const user = userEvent.setup()
    render(<Workflows />)

    await waitFor(() => {
      expect(screen.getByText('Test Workflow')).toBeInTheDocument()
    })

    // Table view button (the list icon)
    const tableBtn = screen.getByText('☰')
    await user.click(tableBtn)

    // In table view we should see table headers
    await waitFor(() => {
      expect(screen.getByText('Name')).toBeInTheDocument()
      expect(screen.getByText('Status')).toBeInTheDocument()
      expect(screen.getByText('Steps')).toBeInTheDocument()
    })
  })

  it('WL4: Search input filters results', async () => {
    server.use(
      http.get('/v2/workflows', () => {
        return HttpResponse.json({
          workflows: [
            { ...fixtures.workflow, id: 'wf-1', name: 'Alpha Workflow' },
            { ...fixtures.workflow, id: 'wf-2', name: 'Beta Pipeline' },
          ],
          total: 2,
        })
      })
    )

    const api = (await import('../lib/api')).default
    api.get.mockImplementation(() =>
      fetch('/v2/workflows').then(r => r.json())
    )

    const user = userEvent.setup()
    render(<Workflows />)

    await waitFor(() => {
      expect(screen.getByText('Alpha Workflow')).toBeInTheDocument()
      expect(screen.getByText('Beta Pipeline')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/search workflows/i)
    await user.type(searchInput, 'Alpha')

    await waitFor(() => {
      expect(screen.getByText('Alpha Workflow')).toBeInTheDocument()
      expect(screen.queryByText('Beta Pipeline')).not.toBeInTheDocument()
    })
  })

  it('WL7: New Workflow button navigates to /workflows/new', async () => {
    const api = (await import('../lib/api')).default
    api.get.mockImplementation(() =>
      fetch('/v2/workflows').then(r => r.json())
    )

    const user = userEvent.setup()
    render(<Workflows />)

    const newBtn = screen.getByText('+ New Workflow')
    await user.click(newBtn)

    expect(mockNavigate).toHaveBeenCalledWith('/workflows/new')
  })

  it('WL15: API 500 shows error', async () => {
    server.use(
      http.get('/v2/workflows', () => {
        return HttpResponse.json({ detail: 'Internal Server Error' }, { status: 500 })
      })
    )

    const api = (await import('../lib/api')).default
    api.get.mockImplementation(() =>
      fetch('/v2/workflows').then(r => {
        if (!r.ok) throw new Error('Internal Server Error')
        return r.json()
      })
    )

    render(<Workflows />)

    await waitFor(() => {
      expect(screen.getByText(/Internal Server Error/)).toBeInTheDocument()
    })
  })

  it('WL16: Loading state shows skeleton', async () => {
    const api = (await import('../lib/api')).default
    // Make the API hang indefinitely so loading persists
    api.get.mockImplementation(() => new Promise(() => {}))

    render(<Workflows />)

    // The loading state renders 3 skeleton cards with animate-pulse
    const skeletons = document.querySelectorAll('.animate-pulse')
    expect(skeletons.length).toBe(3)
  })
})
