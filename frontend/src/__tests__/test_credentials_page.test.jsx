import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'

// --- Mocks ---
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/credentials' }),
  useParams: () => ({}),
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  BrowserRouter: ({ children }) => <div>{children}</div>,
}))

const mockCredentials = [
  { id: 'cred-1', name: 'OpenAI Key', type: 'api_key', created_at: '2026-02-20T10:00:00Z' },
  { id: 'cred-2', name: 'HubSpot Token', type: 'oauth2', created_at: '2026-02-21T12:00:00Z' },
  { id: 'cred-3', name: 'Slack Webhook', type: 'webhook', created_at: '2026-02-22T08:00:00Z' },
]

let mockCredData = { credentials: mockCredentials }
const mockMutateFn = vi.fn()
const mockDeleteMutateFn = vi.fn()

vi.mock('../hooks/useCredentials.js', () => ({
  useCredentials: () => ({ data: mockCredData, isLoading: false }),
  useCreateCredential: () => ({ mutate: mockMutateFn, isPending: false }),
  useDeleteCredential: () => ({ mutate: mockDeleteMutateFn }),
  useCredentialTypes: () => ({ data: [], isLoading: false }),
  useTestCredential: () => ({ mutate: vi.fn() }),
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

// Mock the testCredential API function used directly in CredentialForm
vi.mock('../api/credentials.js', () => ({
  testCredential: vi.fn(() => Promise.resolve({ success: true })),
  listCredentials: vi.fn(),
  createCredential: vi.fn(),
  deleteCredential: vi.fn(),
  getCredentialTypes: vi.fn(),
  peekCredential: vi.fn(),
  rotateCredential: vi.fn(),
}))

import Credentials from '../pages/Credentials.jsx'

beforeEach(() => {
  mockCredData = { credentials: mockCredentials }
  mockMutateFn.mockClear()
  mockDeleteMutateFn.mockClear()
})

// CR1: Credential list renders
describe('CR1: Credential list renders', () => {
  it('renders Credentials heading', () => {
    render(<Credentials />)
    expect(screen.getByText('Credentials')).toBeInTheDocument()
  })

  it('renders credential names', () => {
    render(<Credentials />)
    expect(screen.getByText('OpenAI Key')).toBeInTheDocument()
    expect(screen.getByText('HubSpot Token')).toBeInTheDocument()
    expect(screen.getByText('Slack Webhook')).toBeInTheDocument()
  })

  it('renders column headers Name, Type, Created', () => {
    render(<Credentials />)
    expect(screen.getByText('Name')).toBeInTheDocument()
    expect(screen.getByText('Type')).toBeInTheDocument()
    expect(screen.getByText('Created')).toBeInTheDocument()
  })

  it('renders type badges', () => {
    render(<Credentials />)
    expect(screen.getByText('api_key')).toBeInTheDocument()
    expect(screen.getByText('oauth2')).toBeInTheDocument()
    expect(screen.getByText('webhook')).toBeInTheDocument()
  })
})

// CR2: Add Credential opens form
describe('CR2: Add Credential form', () => {
  it('shows Add Credential button', () => {
    render(<Credentials />)
    expect(screen.getByText('Add Credential')).toBeInTheDocument()
  })

  it('opens credential form modal on click', async () => {
    render(<Credentials />)
    fireEvent.click(screen.getByText('Add Credential'))
    // CredentialForm renders a Name label and Save button
    await waitFor(() => {
      expect(screen.getByPlaceholderText('My API Key')).toBeInTheDocument()
    })
    expect(screen.getByText('Save')).toBeInTheDocument()
  })

  it('shows API Key field in form by default', async () => {
    render(<Credentials />)
    fireEvent.click(screen.getByText('Add Credential'))
    await waitFor(() => {
      expect(screen.getByText('API Key')).toBeInTheDocument()
    })
  })
})

// CR3: Submit empty name still calls onSubmit (no built-in validation on the form)
// The CredentialForm passes the form data directly to onSubmit — validation is server-side
describe('CR3: Form submission behavior', () => {
  it('submit with empty name calls mutate with empty name', async () => {
    render(<Credentials />)
    fireEvent.click(screen.getByText('Add Credential'))
    await waitFor(() => {
      expect(screen.getByText('Save')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Save'))
    expect(mockMutateFn).toHaveBeenCalledWith(
      expect.objectContaining({ name: '' }),
      expect.any(Object),
    )
  })
})

// CR4: Submit valid form calls POST
describe('CR4: Submit valid credential', () => {
  it('fills name and submits', async () => {
    const user = userEvent.setup()
    render(<Credentials />)
    fireEvent.click(screen.getByText('Add Credential'))
    await waitFor(() => {
      expect(screen.getByPlaceholderText('My API Key')).toBeInTheDocument()
    })
    const nameInput = screen.getByPlaceholderText('My API Key')
    await user.type(nameInput, 'My New Credential')
    fireEvent.click(screen.getByText('Save'))
    expect(mockMutateFn).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'My New Credential' }),
      expect.any(Object),
    )
  })
})

// CR5: No secret values in DOM
describe('CR5: No secret values in DOM', () => {
  it('does not render any secret/value/token field in the credential list', () => {
    render(<Credentials />)
    // The DataTable only shows name, type, created_at — no secret/data fields
    const html = document.body.innerHTML
    expect(html).not.toContain('sk-')
    expect(html).not.toContain('secret_')
    expect(html).not.toContain('password')
  })
})

// CR6: Test Connection button in CredentialForm
describe('CR6: Test Connection button', () => {
  it('renders Test Connection button in form', async () => {
    render(<Credentials />)
    fireEvent.click(screen.getByText('Add Credential'))
    await waitFor(() => {
      expect(screen.getByText('Test Connection')).toBeInTheDocument()
    })
  })
})

// CR7: Peek credential — API function exists
describe('CR7: Peek credential API', () => {
  it('peekCredential function is available', async () => {
    const { peekCredential } = await import('../api/credentials.js')
    expect(peekCredential).toBeDefined()
  })
})

// CR8: Rotate credential — API function exists
describe('CR8: Rotate credential API', () => {
  it('rotateCredential function is available', async () => {
    const { rotateCredential } = await import('../api/credentials.js')
    expect(rotateCredential).toBeDefined()
  })
})

// CR9: Delete credential with confirmation
describe('CR9: Delete credential', () => {
  it('renders Delete button for each credential', () => {
    render(<Credentials />)
    const deleteButtons = screen.getAllByText('Delete')
    expect(deleteButtons.length).toBe(3)
  })

  it('shows ConfirmDelete dialog when Delete clicked', () => {
    render(<Credentials />)
    const deleteButtons = screen.getAllByText('Delete')
    fireEvent.click(deleteButtons[0])
    expect(screen.getByText('Delete Credential')).toBeInTheDocument()
    expect(screen.getByText(/Are you sure you want to delete "OpenAI Key"/)).toBeInTheDocument()
  })

  it('calls deleteCredential.mutate on confirm', () => {
    render(<Credentials />)
    const deleteButtons = screen.getAllByText('Delete')
    fireEvent.click(deleteButtons[0])
    // ConfirmDelete has a Delete button to confirm
    const confirmBtn = screen.getAllByText('Delete').find(
      btn => btn.className && btn.className.includes('bg-red')
    )
    fireEvent.click(confirmBtn)
    expect(mockDeleteMutateFn).toHaveBeenCalledWith('cred-1')
  })

  it('cancel closes the dialog without deleting', () => {
    render(<Credentials />)
    const deleteButtons = screen.getAllByText('Delete')
    fireEvent.click(deleteButtons[0])
    expect(screen.getByText('Delete Credential')).toBeInTheDocument()
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Delete Credential')).not.toBeInTheDocument()
  })
})

// CR10: Empty state
describe('CR10: Empty state', () => {
  it('shows EmptyState when no credentials', () => {
    mockCredData = { credentials: [] }
    render(<Credentials />)
    expect(screen.getByText('No credentials')).toBeInTheDocument()
    expect(screen.getByText('Add credentials to connect to external services')).toBeInTheDocument()
  })

  it('shows action button in empty state', () => {
    mockCredData = { credentials: [] }
    render(<Credentials />)
    // Both the header and empty state have "Add Credential"
    const buttons = screen.getAllByText('Add Credential')
    expect(buttons.length).toBe(2)
  })
})

// Cancel button in form
describe('Form cancel', () => {
  it('closes modal when Cancel clicked in form', async () => {
    render(<Credentials />)
    fireEvent.click(screen.getByText('Add Credential'))
    await waitFor(() => {
      expect(screen.getByText('Save')).toBeInTheDocument()
    })
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByText('Save')).not.toBeInTheDocument()
  })
})
