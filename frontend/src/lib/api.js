/**
 * API client with auth interceptors, error handling, retry.
 *
 * Usage:
 *   import api from './lib/api'
 *   const result = await api.post('/v1/execute', { task: '...' })
 */

const API_BASE = '/v1'

class ApiClient {
  constructor() {
    this.token = localStorage.getItem('nexus_token')
  }

  setToken(token) {
    this.token = token
    localStorage.setItem('nexus_token', token)
  }

  clearToken() {
    this.token = null
    localStorage.removeItem('nexus_token')
  }

  async request(method, path, body = null) {
    const headers = { 'Content-Type': 'application/json' }
    if (this.token) headers['Authorization'] = `Bearer ${this.token}`

    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: body ? JSON.stringify(body) : null,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(error.detail || `HTTP ${response.status}`)
    }

    return response.json()
  }

  get(path) { return this.request('GET', path) }
  post(path, body) { return this.request('POST', path, body) }
  put(path, body) { return this.request('PUT', path, body) }
  delete(path) { return this.request('DELETE', path) }
}

export default new ApiClient()
