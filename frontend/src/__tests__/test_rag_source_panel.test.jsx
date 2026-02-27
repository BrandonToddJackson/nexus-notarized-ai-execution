/**
 * RagSourcePanel — comprehensive test suite
 *
 * Coverage:
 *   RS1  Visibility / mount
 *   RS2  Tab switching
 *   RS3  Submit button disabled states
 *   RS4  Text tab — happy path (FormData body, onSuccess callback)
 *   RS5  URL tab — submit URL
 *   RS6  File tab — file selection and upload
 *   RS7  Auth token injected from localStorage
 *   RS8  Error handling (API error, network error, JSON parse fallback)
 *   RS9  Loading state
 *   RS10 Close behaviours (X button, Cancel, backdrop click)
 *   RS11 Error clears on retry
 *   RS12 Namespace default and override
 */

import React from 'react'
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import '@testing-library/jest-dom'
import { http, HttpResponse, delay } from 'msw'
import { server } from '../mocks/server.js'
import RagSourcePanel from '../components/knowledge/RagSourcePanel'

// ─── helpers ──────────────────────────────────────────────────────────────────

const onClose = vi.fn()
const onSuccess = vi.fn()

function renderOpen(props = {}) {
  return render(
    <RagSourcePanel open={true} onClose={onClose} onSuccess={onSuccess} {...props} />
  )
}

function getSubmitBtn() {
  return screen.getByRole('button', { name: /^Ingest$/i })
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
})

// ─── RS1: Visibility / mount ───────────────────────────────────────────────────

describe('RS1: Visibility', () => {
  it('renders nothing when open=false', () => {
    const { container } = render(
      <RagSourcePanel open={false} onClose={onClose} onSuccess={onSuccess} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders panel heading when open=true', () => {
    renderOpen()
    expect(screen.getByText('Add RAG Source')).toBeInTheDocument()
  })

  it('renders namespace input with default value "campaign"', () => {
    renderOpen()
    expect(screen.getByLabelText('namespace')).toHaveValue('campaign')
  })
})

// ─── RS2: Tab switching ────────────────────────────────────────────────────────

describe('RS2: Tab switching', () => {
  it('renders all three tab labels', () => {
    renderOpen()
    expect(screen.getByText('Text / Notion')).toBeInTheDocument()
    expect(screen.getByText('URL')).toBeInTheDocument()
    expect(screen.getByText('File Upload')).toBeInTheDocument()
  })

  it('shows textarea on text tab (default)', () => {
    renderOpen()
    expect(screen.getByLabelText('content')).toBeInTheDocument()
    expect(screen.queryByLabelText('url')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('file')).not.toBeInTheDocument()
  })

  it('shows url input after clicking URL tab', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('URL'))
    expect(screen.getByLabelText('url')).toBeInTheDocument()
    expect(screen.queryByLabelText('content')).not.toBeInTheDocument()
  })

  it('shows file input after clicking File Upload tab', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('File Upload'))
    expect(screen.getByLabelText('file')).toBeInTheDocument()
    expect(screen.queryByLabelText('content')).not.toBeInTheDocument()
  })
})

// ─── RS3: Submit button disabled states ───────────────────────────────────────

describe('RS3: Submit disabled states', () => {
  it('submit disabled on text tab when content is empty', () => {
    renderOpen()
    expect(getSubmitBtn()).toBeDisabled()
  })

  it('submit disabled on text tab when content is whitespace only', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.type(screen.getByLabelText('content'), '   ')
    expect(getSubmitBtn()).toBeDisabled()
  })

  it('submit enabled on text tab when content has non-whitespace', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.type(screen.getByLabelText('content'), 'hello')
    expect(getSubmitBtn()).toBeEnabled()
  })

  it('submit disabled on URL tab when url is empty', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('URL'))
    expect(getSubmitBtn()).toBeDisabled()
  })

  it('submit enabled on URL tab when url is filled', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('URL'))
    await user.type(screen.getByLabelText('url'), 'https://example.com')
    expect(getSubmitBtn()).toBeEnabled()
  })

  it('submit disabled on File tab when no file selected', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('File Upload'))
    expect(getSubmitBtn()).toBeDisabled()
  })
})

// ─── RS4: Text tab — happy path ────────────────────────────────────────────────

describe('RS4: Text tab happy path', () => {
  it('sends content and namespace as FormData fields', async () => {
    const user = userEvent.setup()
    let captured = null
    server.use(
      http.post('/v1/knowledge/multimodal', async ({ request }) => {
        captured = await request.formData()
        return HttpResponse.json({ document_id: 'doc_rs4', status: 'ingested' })
      })
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'ICP: SaaS founders')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(captured).not.toBeNull())
    expect(captured.get('content')).toBe('ICP: SaaS founders')
    expect(captured.get('namespace')).toBe('campaign')
    expect(captured.get('url')).toBeNull()  // url not appended on text tab
  })

  it('calls onSuccess with document_id from response', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', () =>
        HttpResponse.json({ document_id: 'doc_abc', status: 'ingested' })
      )
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'some content')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith('doc_abc'))
  })
})

// ─── RS5: URL tab ─────────────────────────────────────────────────────────────

describe('RS5: URL tab', () => {
  it('sends url field (not content) in FormData', async () => {
    const user = userEvent.setup()
    let captured = null
    server.use(
      http.post('/v1/knowledge/multimodal', async ({ request }) => {
        captured = await request.formData()
        return HttpResponse.json({ document_id: 'doc_url', status: 'ingested' })
      })
    )

    renderOpen()
    await user.click(screen.getByText('URL'))
    await user.type(screen.getByLabelText('url'), 'https://example.com/brief')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(captured).not.toBeNull())
    expect(captured.get('url')).toBe('https://example.com/brief')
    expect(captured.get('content')).toBeNull()  // content not appended on url tab
    expect(captured.get('namespace')).toBe('campaign')
  })
})

// ─── RS6: File tab ─────────────────────────────────────────────────────────────

describe('RS6: File tab', () => {
  it('shows selected filename after file chosen', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('File Upload'))

    const fileInput = screen.getByLabelText('file')
    const mockFile = new File(['pdf content'], 'campaign.pdf', { type: 'application/pdf' })
    fireEvent.change(fileInput, { target: { files: [mockFile] } })

    await waitFor(() => {
      expect(screen.getByText('campaign.pdf')).toBeInTheDocument()
    })
  })

  it('submit enabled after file selected', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByText('File Upload'))

    const fileInput = screen.getByLabelText('file')
    const mockFile = new File(['content'], 'brief.pdf', { type: 'application/pdf' })
    fireEvent.change(fileInput, { target: { files: [mockFile] } })

    await waitFor(() => expect(getSubmitBtn()).toBeEnabled())
  })

  it('sends file as FormData field', async () => {
    const user = userEvent.setup()
    let captured = null
    server.use(
      http.post('/v1/knowledge/multimodal', async ({ request }) => {
        captured = await request.formData()
        return HttpResponse.json({ document_id: 'doc_file', status: 'ingested' })
      })
    )

    renderOpen()
    await user.click(screen.getByText('File Upload'))
    const fileInput = screen.getByLabelText('file')
    const mockFile = new File(['pdf bytes'], 'test.pdf', { type: 'application/pdf' })
    fireEvent.change(fileInput, { target: { files: [mockFile] } })
    await waitFor(() => expect(getSubmitBtn()).toBeEnabled())
    await user.click(getSubmitBtn())

    await waitFor(() => expect(captured).not.toBeNull())
    const uploadedFile = captured.get('file')
    // jsdom serialises File objects as Blobs in multipart; assert presence and size
    expect(uploadedFile).toBeTruthy()
    expect(uploadedFile.size).toBeGreaterThan(0)
  })
})

// ─── RS7: Auth token injection ─────────────────────────────────────────────────

describe('RS7: Auth header', () => {
  it('sends Authorization header when nexus_token in localStorage', async () => {
    const user = userEvent.setup()
    localStorage.setItem('nexus_token', 'jwt-abc123')
    let authHeader = null
    server.use(
      http.post('/v1/knowledge/multimodal', ({ request }) => {
        authHeader = request.headers.get('authorization')
        return HttpResponse.json({ document_id: 'doc_auth', status: 'ingested' })
      })
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'secured content')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(authHeader).not.toBeNull())
    expect(authHeader).toBe('Bearer jwt-abc123')
  })

  it('sends no Authorization header when no token in localStorage', async () => {
    const user = userEvent.setup()
    let authHeader = 'not-checked'
    server.use(
      http.post('/v1/knowledge/multimodal', ({ request }) => {
        authHeader = request.headers.get('authorization')
        return HttpResponse.json({ document_id: 'doc_noauth', status: 'ingested' })
      })
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'open content')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(authHeader).toBe(null))
  })
})

// ─── RS8: Error handling ───────────────────────────────────────────────────────

describe('RS8: Error handling', () => {
  it('shows API error detail on 503 response', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', () =>
        HttpResponse.json(
          { detail: 'RAG-Anything not enabled. Set NEXUS_RAG_ANYTHING_ENABLED=true.' },
          { status: 503 }
        )
      )
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'some content')
    await user.click(getSubmitBtn())

    await waitFor(() => {
      expect(screen.getByText('RAG-Anything not enabled. Set NEXUS_RAG_ANYTHING_ENABLED=true.')).toBeInTheDocument()
    })
    expect(onSuccess).not.toHaveBeenCalled()
  })

  it('falls back to HTTP status text when response body is not JSON', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', () =>
        new HttpResponse('Bad Gateway', { status: 502, statusText: 'Bad Gateway' })
      )
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'content')
    await user.click(getSubmitBtn())

    await waitFor(() => {
      // Component catches json() failure and uses statusText or "HTTP 502"
      const error = screen.getByText(/Bad Gateway|HTTP 502/i)
      expect(error).toBeInTheDocument()
    })
  })

  it('shows error on network failure', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', () => HttpResponse.error())
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'content')
    await user.click(getSubmitBtn())

    await waitFor(() => {
      // Any non-empty error message rendered in the red div
      const errorDiv = document.querySelector('.bg-red-900\\/50')
      expect(errorDiv).toBeTruthy()
      expect(errorDiv.textContent.length).toBeGreaterThan(0)
    })
  })

  it('does not call onClose when submission fails', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', () =>
        HttpResponse.json({ detail: 'Server error' }, { status: 500 })
      )
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'content')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(screen.getByText('Server error')).toBeInTheDocument())
    expect(onClose).not.toHaveBeenCalled()
  })
})

// ─── RS9: Loading state ────────────────────────────────────────────────────────

describe('RS9: Loading state', () => {
  it('button shows "Ingesting..." while request is in flight', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', async () => {
        await delay(80)
        return HttpResponse.json({ document_id: 'doc_slow', status: 'ingested' })
      })
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'campaign brief')
    await user.click(getSubmitBtn())

    // Button should show loading text while response is delayed
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Ingesting\.\.\./i })).toBeInTheDocument()
    })

    // After response resolves, loading ends and onSuccess fires
    await waitFor(() => expect(onSuccess).toHaveBeenCalled(), { timeout: 500 })
  })

  it('submit button is disabled while loading', async () => {
    const user = userEvent.setup()
    server.use(
      http.post('/v1/knowledge/multimodal', async () => {
        await delay(80)
        return HttpResponse.json({ document_id: 'doc_slow2', status: 'ingested' })
      })
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'content')
    await user.click(getSubmitBtn())

    await waitFor(() => {
      const btn = screen.getByRole('button', { name: /Ingesting\.\.\./i })
      expect(btn).toBeDisabled()
    })

    await waitFor(() => expect(onSuccess).toHaveBeenCalled(), { timeout: 500 })
  })
})

// ─── RS10: Close behaviours ────────────────────────────────────────────────────

describe('RS10: Close behaviours', () => {
  it('X button calls onClose', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('Cancel button calls onClose', async () => {
    const user = userEvent.setup()
    renderOpen()
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('clicking backdrop overlay calls onClose', () => {
    const { container } = renderOpen()
    // container.firstChild is the overlay div (fixed inset-0)
    fireEvent.click(container.firstChild)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('clicking inside the panel card does NOT call onClose', () => {
    renderOpen()
    // Click the heading inside the modal — should not trigger onClose
    fireEvent.click(screen.getByText('Add RAG Source'))
    expect(onClose).not.toHaveBeenCalled()
  })
})

// ─── RS11: Error clears on retry ───────────────────────────────────────────────

describe('RS11: Error clears on retry', () => {
  it('error message disappears when user resubmits successfully', async () => {
    const user = userEvent.setup()

    // First call fails
    server.use(
      http.post('/v1/knowledge/multimodal', () =>
        HttpResponse.json({ detail: 'Temporary failure' }, { status: 500 })
      )
    )

    renderOpen()
    await user.type(screen.getByLabelText('content'), 'content')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(screen.getByText('Temporary failure')).toBeInTheDocument())

    // Fix the server response for retry
    server.use(
      http.post('/v1/knowledge/multimodal', () =>
        HttpResponse.json({ document_id: 'doc_retry', status: 'ingested' })
      )
    )

    await user.click(getSubmitBtn())

    // Error should be cleared on retry start
    await waitFor(() => expect(screen.queryByText('Temporary failure')).not.toBeInTheDocument())
    await waitFor(() => expect(onSuccess).toHaveBeenCalledWith('doc_retry'))
  })
})

// ─── RS12: Namespace ──────────────────────────────────────────────────────────

describe('RS12: Namespace', () => {
  it('default namespace is "campaign"', () => {
    renderOpen()
    expect(screen.getByLabelText('namespace')).toHaveValue('campaign')
  })

  it('changed namespace is sent in FormData', async () => {
    const user = userEvent.setup()
    let captured = null
    server.use(
      http.post('/v1/knowledge/multimodal', async ({ request }) => {
        captured = await request.formData()
        return HttpResponse.json({ document_id: 'doc_ns', status: 'ingested' })
      })
    )

    renderOpen()
    const nsInput = screen.getByLabelText('namespace')
    await user.clear(nsInput)
    await user.type(nsInput, 'cold_outreach')
    await user.type(screen.getByLabelText('content'), 'content')
    await user.click(getSubmitBtn())

    await waitFor(() => expect(captured).not.toBeNull())
    expect(captured.get('namespace')).toBe('cold_outreach')
  })
})
