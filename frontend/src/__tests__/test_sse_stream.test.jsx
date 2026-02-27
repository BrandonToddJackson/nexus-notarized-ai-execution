import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Track EventSource instances for testing
let mockEventSourceInstances = []
let MockEventSource

function createMockEventSource() {
  MockEventSource = vi.fn(function (url) {
    this.url = url
    this.listeners = {}
    this.addEventListener = vi.fn((event, handler) => {
      this.listeners[event] = handler
    })
    this.close = vi.fn()
    mockEventSourceInstances.push(this)
  })
  vi.stubGlobal('EventSource', MockEventSource)
}

// Mock auth to return a known token
vi.mock('../lib/auth.js', () => ({
  getToken: vi.fn(() => 'test-jwt-token'),
  setToken: vi.fn(),
  clearAuth: vi.fn(),
  isAuthenticated: vi.fn(() => true),
}))

// Mock react-hot-toast — use vi.hoisted to make the mock function available at hoist time
const { mockToastError } = vi.hoisted(() => ({ mockToastError: vi.fn() }))
vi.mock('react-hot-toast', () => ({
  default: { error: mockToastError, success: vi.fn() },
}))

import { useExecutionStream } from '../hooks/useExecutionStream.js'
import { useAppStore } from '../stores/appStore.js'

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return function Wrapper({ children }) {
    return React.createElement(QueryClientProvider, { client: queryClient }, children)
  }
}

beforeEach(() => {
  mockEventSourceInstances = []
  createMockEventSource()
  mockToastError.mockReset()
  // Reset zustand store
  useAppStore.setState({ gateFailureCount: 0 })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useExecutionStream SSE hook', () => {
  it('SSE1: EventSource URL includes ?token=jwt parameter', () => {
    const wrapper = createWrapper()
    renderHook(() => useExecutionStream(), { wrapper })

    expect(mockEventSourceInstances).toHaveLength(1)
    const es = mockEventSourceInstances[0]
    expect(es.url).toContain('/v2/events/stream?token=')
    expect(es.url).toContain('test-jwt-token')
  })

  it('SSE2: execution_update event triggers React Query invalidation', () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries')

    function Wrapper({ children }) {
      return React.createElement(QueryClientProvider, { client: queryClient }, children)
    }

    renderHook(() => useExecutionStream(), { wrapper: Wrapper })

    const es = mockEventSourceInstances[0]
    expect(es.addEventListener).toHaveBeenCalledWith('execution_update', expect.any(Function))

    // Fire the event
    const handler = es.listeners['execution_update']
    act(() => { handler({ data: '{}' }) })

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['executions'] })
  })

  it('SSE3: gate_failure event increments Zustand gateFailureCount', () => {
    const wrapper = createWrapper()
    expect(useAppStore.getState().gateFailureCount).toBe(0)

    renderHook(() => useExecutionStream(), { wrapper })

    const es = mockEventSourceInstances[0]
    const handler = es.listeners['gate_failure']
    act(() => { handler({ data: '{}' }) })

    expect(useAppStore.getState().gateFailureCount).toBe(1)
  })

  it('SSE4: gate_failure event shows toast error', () => {
    const wrapper = createWrapper()
    renderHook(() => useExecutionStream(), { wrapper })

    const es = mockEventSourceInstances[0]
    const handler = es.listeners['gate_failure']
    act(() => { handler({ data: '{}' }) })

    expect(mockToastError).toHaveBeenCalledWith('Gate failure detected', { duration: 4000 })
  })

  it('SSE5: EventSource is closed on unmount', () => {
    const wrapper = createWrapper()
    const { unmount } = renderHook(() => useExecutionStream(), { wrapper })

    const es = mockEventSourceInstances[0]
    expect(es.close).not.toHaveBeenCalled()

    unmount()

    expect(es.close).toHaveBeenCalled()
  })

  it('SSE6: No EventSource created when token is null', async () => {
    const { getToken } = await import('../lib/auth.js')
    getToken.mockReturnValueOnce(null)

    const wrapper = createWrapper()
    renderHook(() => useExecutionStream(), { wrapper })

    expect(mockEventSourceInstances).toHaveLength(0)
  })
})
