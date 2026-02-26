import { vi, describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import '@testing-library/jest-dom'

// Mock react-router-dom
const mockNavigate = vi.fn()
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
  useLocation: () => ({ pathname: '/skills' }),
  Link: ({ children, to, ...props }) => <a href={to} {...props}>{children}</a>,
}))

// Mock hooks
const mockUpdateSkill = { mutate: vi.fn() }
const mockDeleteSkill = { mutate: vi.fn() }
const mockDuplicateSkill = { mutate: vi.fn() }

vi.mock('../hooks/useSkills.js', () => ({
  useSkills: () => ({ data: [], isLoading: false }),
  useSkill: () => ({ data: null, isLoading: false }),
  useUpdateSkill: () => mockUpdateSkill,
  useDeleteSkill: () => mockDeleteSkill,
  useDuplicateSkill: () => mockDuplicateSkill,
  useCreateSkill: () => ({ mutate: vi.fn() }),
  useSkillInvocations: () => ({ data: [], isLoading: false }),
}))

vi.mock('../hooks/useCredentials.js', () => ({
  useCredentials: () => ({ data: [], isLoading: false }),
  useCreateCredential: () => ({ mutate: vi.fn() }),
  useDeleteCredential: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useMCPServers.js', () => ({
  useMCPServers: () => ({ data: [], isLoading: false }),
  useAddMCPServer: () => ({ mutate: vi.fn() }),
  useRemoveMCPServer: () => ({ mutate: vi.fn() }),
  useMCPServerTools: () => ({ data: [], isLoading: false }),
  useReconnectMCPServer: () => ({ mutate: vi.fn() }),
}))

vi.mock('../hooks/useExecutions.js', () => ({
  useExecutions: () => ({ data: [], isLoading: false }),
}))

vi.mock('../hooks/useExecutionStream.js', () => ({
  useExecutionStream: () => {},
}))

// Mock diff for SkillVersionDiff
vi.mock('diff', () => ({
  createPatch: (name, oldStr, newStr) => {
    const lines = []
    lines.push(`--- ${name}`)
    lines.push(`+++ ${name}`)
    lines.push('@@ -1 +1 @@')
    if (oldStr) lines.push(`-${oldStr}`)
    if (newStr) lines.push(`+${newStr}`)
    return lines.join('\n')
  },
}))

import { SkillCard } from '../components/skills/SkillCard.jsx'
import { SkillVersionDiff } from '../components/skills/SkillVersionDiff.jsx'
import { SkillFilePicker } from '../components/skills/SkillFilePicker.jsx'

const SAMPLE_SKILL = {
  id: 'sk-1',
  name: 'my-skill',
  display_name: 'My Awesome Skill',
  description: 'This is a test skill that does amazing things for testing purposes and more text to check truncation behavior.',
  version: 3,
  tags: ['test', 'demo'],
  invocation_count: 42,
  is_active: true,
}

describe('SkillCard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders display_name', () => {
    render(<SkillCard skill={SAMPLE_SKILL} />)
    expect(screen.getByText('My Awesome Skill')).toBeInTheDocument()
  })

  it('renders description preview', () => {
    render(<SkillCard skill={SAMPLE_SKILL} />)
    // Description under 120 chars should render as-is
    expect(screen.getByText(SAMPLE_SKILL.description)).toBeInTheDocument()
  })

  it('truncates long description at 120 chars', () => {
    const longDesc = 'A'.repeat(150)
    const skill = { ...SAMPLE_SKILL, description: longDesc }
    render(<SkillCard skill={skill} />)
    const truncated = screen.getByText(`${'A'.repeat(120)}...`)
    expect(truncated).toBeInTheDocument()
  })

  it('renders version badge', () => {
    render(<SkillCard skill={SAMPLE_SKILL} />)
    expect(screen.getByText('v3')).toBeInTheDocument()
  })

  it('renders invocation count', () => {
    render(<SkillCard skill={SAMPLE_SKILL} />)
    expect(screen.getByText('42 invocations')).toBeInTheDocument()
  })

  it('renders kebab menu with options on click', () => {
    render(<SkillCard skill={SAMPLE_SKILL} />)
    const menuButton = screen.getByText('...')
    fireEvent.click(menuButton)
    expect(screen.getByText('Edit')).toBeInTheDocument()
    expect(screen.getByText('Duplicate')).toBeInTheDocument()
    expect(screen.getByText('Delete')).toBeInTheDocument()
  })

  it('renders tags', () => {
    render(<SkillCard skill={SAMPLE_SKILL} />)
    expect(screen.getByText('test')).toBeInTheDocument()
    expect(screen.getByText('demo')).toBeInTheDocument()
  })
})

describe('SkillVersionDiff', () => {
  it('renders added lines with green styling', () => {
    const { container } = render(
      <SkillVersionDiff oldText="old content" newText="new content" oldVersion={1} newVersion={2} />
    )
    const addedLine = container.querySelector('.text-green-700')
    expect(addedLine).toBeInTheDocument()
  })

  it('renders removed lines with red styling', () => {
    const { container } = render(
      <SkillVersionDiff oldText="old content" newText="new content" oldVersion={1} newVersion={2} />
    )
    const removedLine = container.querySelector('.text-red-700')
    expect(removedLine).toBeInTheDocument()
  })
})

describe('SkillFilePicker', () => {
  it('shows max file size text', () => {
    render(<SkillFilePicker files={[]} onChange={vi.fn()} />)
    expect(screen.getByText(/max 500KB/)).toBeInTheDocument()
  })

  it('shows file name and size', () => {
    const files = [{ name: 'test.txt', size: 1024, description: '' }]
    render(<SkillFilePicker files={files} onChange={vi.fn()} />)
    expect(screen.getByText('test.txt')).toBeInTheDocument()
    expect(screen.getByText('1.0 KB')).toBeInTheDocument()
  })

  it('shows Add File button', () => {
    render(<SkillFilePicker files={[]} onChange={vi.fn()} />)
    expect(screen.getByText('Add File')).toBeInTheDocument()
  })

  it('disables Add File when max files reached', () => {
    const files = Array.from({ length: 5 }, (_, i) => ({
      name: `file${i}.txt`, size: 100, description: '',
    }))
    render(<SkillFilePicker files={files} onChange={vi.fn()} />)
    const btn = screen.getByText('Add File')
    expect(btn).toBeDisabled()
  })
})
