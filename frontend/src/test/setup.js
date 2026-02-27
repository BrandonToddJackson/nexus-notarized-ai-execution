import '@testing-library/jest-dom'
import { server } from '../mocks/server.js'

// jsdom doesn't implement scrollIntoView — mock it globally
window.HTMLElement.prototype.scrollIntoView = vi.fn()

// jsdom doesn't implement EventSource — mock it globally so components using SSE don't crash
global.EventSource = vi.fn(() => ({
  close: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  readyState: 1,
  CONNECTING: 0, OPEN: 1, CLOSED: 2,
}))

beforeAll(() => server.listen({ onUnhandledRequest: 'warn' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())
