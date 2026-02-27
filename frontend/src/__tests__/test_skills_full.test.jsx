import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'

// --- Mocks ---
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/skills' }),
  useParams: () => ({ skillId: 'skill-1' }),
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
  BrowserRouter: ({ children }) => <div>{children}</div>,
}))

const mockSkills = [
  {
    id: 'skill-1',
    name: 'code_review',
    display_name: 'Code Review',
    description: 'Automated code review with best practices',
    is_active: true,
    version: 3,
    tags: ['dev', 'review'],
    invocation_count: 42,
  },
  {
    id: 'skill-2',
    name: 'data_extraction',
    display_name: 'Data Extraction',
    description: 'Extract structured data from unstructured text',
    is_active: false,
    version: 1,
    tags: ['data'],
    invocation_count: 7,
  },
  {
    id: 'skill-3',
    name: 'email_drafting',
    display_name: 'Email Drafting',
    description: 'AI-assisted professional email composition',
    is_active: true,
    version: 2,
    tags: ['writing', 'email'],
    invocation_count: 120,
  },
]

let mockSkillsData = { skills: mockSkills }
let mockSkillsFilters = {}
const mockUpdateMutate = vi.fn()
const mockDeleteMutate = vi.fn()
const mockDuplicateMutate = vi.fn()
const mockCreateMutate = vi.fn()

vi.mock('../hooks/useSkills.js', () => ({
  useSkills: (filters) => {
    mockSkillsFilters = filters
    return { data: mockSkillsData, isLoading: false }
  },
  useSkill: () => ({ data: null, isLoading: false }),
  useUpdateSkill: () => ({ mutate: mockUpdateMutate, isPending: false }),
  useDeleteSkill: () => ({ mutate: mockDeleteMutate }),
  useDuplicateSkill: () => ({ mutate: mockDuplicateMutate }),
  useCreateSkill: () => ({ mutate: mockCreateMutate, isPending: false }),
  useSkillInvocations: () => ({ data: [], isLoading: false }),
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

vi.mock('@tanstack/react-query', () => ({
  QueryClient: vi.fn(() => ({})),
  QueryClientProvider: ({ children }) => <div>{children}</div>,
  useQuery: () => ({ data: null, isLoading: false }),
  useMutation: () => ({ mutate: vi.fn(), isPending: false }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}))

import Skills from '../pages/Skills.jsx'

beforeEach(() => {
  mockSkillsData = { skills: mockSkills }
  mockSkillsFilters = {}
  mockNavigate.mockClear()
  mockUpdateMutate.mockClear()
  mockDeleteMutate.mockClear()
  mockDuplicateMutate.mockClear()
  mockCreateMutate.mockClear()
})

// SK1: Skills list loads with name, active status, version
describe('SK1: Skills list display', () => {
  it('renders Skills heading', () => {
    render(<Skills />)
    expect(screen.getByText('Skills')).toBeInTheDocument()
  })

  it('renders skill display names via SkillCard', () => {
    render(<Skills />)
    expect(screen.getByText('Code Review')).toBeInTheDocument()
    expect(screen.getByText('Data Extraction')).toBeInTheDocument()
    expect(screen.getByText('Email Drafting')).toBeInTheDocument()
  })

  it('renders skill internal names', () => {
    render(<Skills />)
    expect(screen.getByText('code_review')).toBeInTheDocument()
    expect(screen.getByText('data_extraction')).toBeInTheDocument()
    expect(screen.getByText('email_drafting')).toBeInTheDocument()
  })

  it('renders version badges', () => {
    render(<Skills />)
    expect(screen.getByText('v3')).toBeInTheDocument()
    expect(screen.getByText('v1')).toBeInTheDocument()
    expect(screen.getByText('v2')).toBeInTheDocument()
  })

  it('renders invocation counts', () => {
    render(<Skills />)
    expect(screen.getByText('42 invocations')).toBeInTheDocument()
    expect(screen.getByText('7 invocations')).toBeInTheDocument()
    expect(screen.getByText('120 invocations')).toBeInTheDocument()
  })

  it('renders tags', () => {
    render(<Skills />)
    expect(screen.getByText('dev')).toBeInTheDocument()
    expect(screen.getByText('review')).toBeInTheDocument()
    expect(screen.getByText('data')).toBeInTheDocument()
  })
})

// SK2: Search filters by name
describe('SK2: Search filter', () => {
  it('renders search input', () => {
    render(<Skills />)
    expect(screen.getByPlaceholderText('Search skills...')).toBeInTheDocument()
  })

  it('passes search filter to useSkills', async () => {
    const user = userEvent.setup()
    render(<Skills />)
    const input = screen.getByPlaceholderText('Search skills...')
    await user.type(input, 'code')
    await waitFor(() => {
      expect(mockSkillsFilters.search).toBe('code')
    })
  })
})

// SK3: Active-only toggle
describe('SK3: Active-only toggle', () => {
  it('renders Active only toggle', () => {
    render(<Skills />)
    expect(screen.getByText('Active only')).toBeInTheDocument()
  })

  it('passes active filter to useSkills when toggled', () => {
    render(<Skills />)
    // Toggle component renders a checkbox-like element
    const toggle = screen.getByText('Active only').closest('label') || screen.getByText('Active only').parentElement
    fireEvent.click(toggle)
    // After clicking, the activeOnly state changes and re-renders
    expect(mockSkillsFilters.active).toBe(true)
  })
})

// SK4: Create new skill navigates to /skills/new
describe('SK4: Create new skill', () => {
  it('renders New Skill button', () => {
    render(<Skills />)
    expect(screen.getByText('New Skill')).toBeInTheDocument()
  })

  it('navigates to /skills/new on click', () => {
    render(<Skills />)
    fireEvent.click(screen.getByText('New Skill'))
    expect(mockNavigate).toHaveBeenCalledWith('/skills/new')
  })
})

// SK5: Edit skill navigates to /skills/:id
describe('SK5: Edit skill', () => {
  it('shows Edit option in card menu', () => {
    render(<Skills />)
    // Click the "..." menu on the first card
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    expect(screen.getByText('Edit')).toBeInTheDocument()
  })

  it('navigates to skill edit page', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    fireEvent.click(screen.getByText('Edit'))
    expect(mockNavigate).toHaveBeenCalledWith('/skills/skill-1')
  })
})

// SK6: Cancel — not applicable at list level (cancel is in SkillEditorPanel)
describe('SK6: No API call without explicit action', () => {
  it('opening and closing menu does not trigger mutations', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    // Close by clicking again
    fireEvent.click(menuButtons[0])
    expect(mockUpdateMutate).not.toHaveBeenCalled()
    expect(mockDeleteMutate).not.toHaveBeenCalled()
  })
})

// SK7: Duplicate
describe('SK7: Duplicate skill', () => {
  it('shows Duplicate option in card menu', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    expect(screen.getByText('Duplicate')).toBeInTheDocument()
  })

  it('calls duplicateSkill.mutate with skill id', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    fireEvent.click(screen.getByText('Duplicate'))
    expect(mockDuplicateMutate).toHaveBeenCalledWith('skill-1')
  })
})

// SK8: Delete
describe('SK8: Delete skill', () => {
  it('shows Delete option in card menu', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    expect(screen.getByText('Delete')).toBeInTheDocument()
  })

  it('calls deleteSkill.mutate with skill id', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    fireEvent.click(screen.getByText('Delete'))
    expect(mockDeleteMutate).toHaveBeenCalledWith('skill-1')
  })
})

// SK9: Import skill
describe('SK9: Import skill', () => {
  it('renders Import button in header', () => {
    render(<Skills />)
    expect(screen.getByText('Import')).toBeInTheDocument()
  })

  it('importSkill API function exists', async () => {
    const { importSkill } = await import('../api/skills.js')
    expect(importSkill).toBeDefined()
  })
})

// SK10: Export skill
describe('SK10: Export skill', () => {
  it('shows Export option in card menu', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    expect(screen.getByText('Export')).toBeInTheDocument()
  })

  it('exportSkill API function exists', async () => {
    const { exportSkill } = await import('../api/skills.js')
    expect(exportSkill).toBeDefined()
  })
})

// SK11: Version diff
describe('SK11: Version diff API', () => {
  it('diffSkillVersions API function exists', async () => {
    const { diffSkillVersions } = await import('../api/skills.js')
    expect(diffSkillVersions).toBeDefined()
  })
})

// SK12: Invocation history
describe('SK12: Invocation history', () => {
  it('shows View Invocations option in card menu', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    expect(screen.getByText('View Invocations')).toBeInTheDocument()
  })

  it('navigates to invocations tab', () => {
    render(<Skills />)
    const menuButtons = screen.getAllByText('...')
    fireEvent.click(menuButtons[0])
    fireEvent.click(screen.getByText('View Invocations'))
    expect(mockNavigate).toHaveBeenCalledWith('/skills/skill-1?tab=invocations')
  })
})

// Empty state
describe('Empty state', () => {
  it('shows EmptyState when no skills', () => {
    mockSkillsData = { skills: [] }
    render(<Skills />)
    expect(screen.getByText('No skills yet')).toBeInTheDocument()
    expect(screen.getByText('Create your first skill to get started')).toBeInTheDocument()
  })

  it('New Skill action in empty state navigates', () => {
    mockSkillsData = { skills: [] }
    render(<Skills />)
    // Both header and empty state have "New Skill"
    const buttons = screen.getAllByText('New Skill')
    fireEvent.click(buttons[buttons.length - 1])
    expect(mockNavigate).toHaveBeenCalledWith('/skills/new')
  })
})

// SkillCard toggle
describe('SkillCard active toggle', () => {
  it('calls updateSkill.mutate when toggle clicked', () => {
    render(<Skills />)
    // Each SkillCard has a Toggle component — Toggle renders role="switch" (not checkbox)
    // The sidebar may also render a switch, so we get >=3 and click the first skill toggle
    const toggles = screen.getAllByRole('switch')
    expect(toggles.length).toBeGreaterThanOrEqual(3)
    // Click the last switch (skill toggles come after any sidebar toggles in DOM order)
    fireEvent.click(toggles[toggles.length - 3])
    expect(mockUpdateMutate).toHaveBeenCalledWith({ id: 'skill-1', is_active: false })
  })
})

// Description truncation
describe('Description display', () => {
  it('renders skill descriptions', () => {
    render(<Skills />)
    expect(screen.getByText('Automated code review with best practices')).toBeInTheDocument()
  })
})
